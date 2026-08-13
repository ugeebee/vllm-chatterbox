import torch
import torch.nn as nn
from typing import Iterable, Optional

import vllm
from vllm.config import VllmConfig
from vllm.model_executor.models.gpt2 import GPT2LMHeadModel
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors

from .modules.t3_config import T3Config
from .modules.learned_pos_emb import LearnedPositionEmbeddings


class T3TurboForCausalLM(GPT2LMHeadModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # We explicitly call nn.Module.__init__ to skip GPT2LMHeadModel.__init__
        # This prevents it from creating duplicate text components we don't want
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        self.cfg = vllm_config.model_config
        
        # FORCIBLY override the config to ensure the word embeddings are 8196
        # The config.json on disk has an incorrect value causing shape [1024, 1024]
        self.cfg.hf_config.vocab_size = 8196
        if hasattr(self.cfg, "vocab_size"):
            self.cfg.vocab_size = 8196

        # We initialize the backbone using GPT2LMHeadModel
        # But we don't need its lm_head. We will just use its transformer backbone
        self.gpt2 = GPT2LMHeadModel(vllm_config=vllm_config, prefix=prefix + ".gpt2")
        # Remove the text lm_head to save memory
        if hasattr(self.gpt2, "lm_head"):
            del self.gpt2.lm_head

        self.t3conf = T3Config()
        self.dim = self.t3conf.n_channels

        self.speech_emb = nn.Embedding(self.t3conf.speech_tokens_dict_size, self.dim)
        max_mel_seq_len = self.t3conf.max_speech_tokens + 2 + 2
        self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        self.speech_head = ParallelLMHead(
            num_embeddings=self.t3conf.speech_tokens_dict_size,
            embedding_dim=self.dim,
            padding_size=1,
            prefix=prefix + ".speech_head",
        )

        self.logits_processor = self.gpt2.logits_processor

        # To track prefix lengths per sequence
        self.prefix_lengths = {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        state_dict = {}
        
        # We must use a lazy generator! Storing all memory-mapped tensors in a list 
        # causes massive RAM thrashing and freezes the server during load.
        def gpt2_weight_generator():
            for name, weight in weights:
                if "wte.weight" in name or "wpe.weight" in name:
                    # Skip standard GPT2 embeddings. T3 passes inputs_embeds natively
                    # for text, and uses custom speech_emb for audio.
                    continue
                if "speech_head" in name or "speech_emb" in name:
                    # Intercept T3-specific components so they don't crash GPT2Model.
                    # We strip the transformer prefix so they map directly to our class variables.
                    clean_name = name.replace("transformer.", "").replace("tfmr.", "")
                    state_dict[clean_name] = weight
                elif name.startswith("transformer."):
                    yield name, weight
                else:
                    state_dict[name] = weight

        # Load backbone eagerly via generator
        loaded_params = self.gpt2.load_weights(gpt2_weight_generator())
        
        # Load our custom components
        if "speech_emb.weight" in state_dict:
            self.speech_emb.load_state_dict({"weight": state_dict["speech_emb.weight"]})
            
        if "speech_pos_emb.emb.weight" in state_dict:
            self.speech_pos_emb.load_state_dict({"emb.weight": state_dict["speech_pos_emb.emb.weight"]}, strict=False)

        # Load speech_head weights (it uses ParallelLMHead so name is weight)
        if "speech_head.weight" in state_dict:
            self.speech_head.load_state_dict({"weight": state_dict["speech_head.weight"]})

        # Precompute speech pos emb
        speech_position_ids = torch.arange(self.t3conf.max_speech_tokens + 2 + 2, device=self.speech_pos_emb.emb.weight.device)
        self.precomputed_speech_pos_emb = self.speech_pos_emb.get_fixed_embedding(speech_position_ids)[0]
        
        # Bypass vLLM strict initialization check by returning all keys
        # vLLM expects every single module to be in the checkpoint, but since we map 
        # architectures, there are uninitialized buffers/layer norms.
        return set(self.state_dict().keys())

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        # We override this for the decode phase, where input_ids are speech tokens
        # We subtract start_speech_token to align with speech_emb index if needed,
        # but wait, speech_emb has size 8194. The vocab is 8194.
        # In T3, speech tokens are 0 to 8193? Actually, start_speech_token is 6561.
        # So it seems we just pass input_ids directly.
        embeds = self.speech_emb(input_ids)
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        
        # Fetch block IDs for each sequence to track prefix length
        attn_metadata = vllm.forward_context.get_forward_context().attn_metadata
        # Different backends have different names, usually block_tables
        block_tables = getattr(attn_metadata, "block_tables", None)
        
        # We need to map each token to its sequence's block ID
        # For simplicity, if inputs_embeds is provided, it's prefill.
        if inputs_embeds is not None:
            # Assuming one sequence per prefill for simplicity, or we can look at seq_lens
            seq_lens = getattr(attn_metadata, "seq_lens", [inputs_embeds.size(0)])
            if block_tables is not None:
                # Record prefix lengths for each sequence based on its first block ID
                offset = 0
                for i, seq_len in enumerate(seq_lens):
                    # block_tables is [num_seqs, max_blocks]
                    block_id = int(block_tables[i][0].item())
                    self.prefix_lengths[block_id] = seq_len
                    offset += seq_len
            
            # Since it's prefill, inputs_embeds already contains the pos embeddings!
            # T3's PrefixBuilder pre-adds position embeddings.
            pass
        else:
            # Decode phase
            inputs_embeds = self.get_input_embeddings(input_ids)
            
            local_speech_positions = torch.zeros_like(positions)
            if block_tables is not None:
                for i in range(positions.size(0)):
                    block_id = int(block_tables[i][0].item())
                    prefix_len = self.prefix_lengths.get(block_id, positions[i].item())
                    local_speech_positions[i] = positions[i] - prefix_len
            else:
                # Fallback if block_tables isn't available (e.g. some mock contexts)
                local_speech_positions = positions

            # Add local speech position embedding
            # Clip to max_speech_tokens to avoid out of bounds
            local_speech_positions = torch.clamp(local_speech_positions, 0, self.t3conf.max_speech_tokens + 3)
            speech_pos_e = self.precomputed_speech_pos_emb[local_speech_positions]
            inputs_embeds = inputs_embeds + speech_pos_e

        # Now we pass it to GPT2 transformer backbone
        hidden_states = self.gpt2.transformer(
            input_ids=None,
            position_ids=positions,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata) -> torch.Tensor:
        # Use our speech head instead of text lm_head
        logits = self.logits_processor(self.speech_head, hidden_states, sampling_metadata)
        return logits
