import torch
import torch.nn as nn
from typing import Optional

from .modules.t3_config import T3Config
from .modules.cond_enc import T3Cond, T3CondEnc
from .modules.learned_pos_emb import LearnedPositionEmbeddings

class PrefixBuilder(nn.Module):
    def __init__(self, config: T3Config, text_tokens_dict_size: int = 50276):
        super().__init__()
        self.config = config
        self.dim = config.n_channels
        
        self.cond_enc = T3CondEnc(config)
        self.text_emb = nn.Embedding(text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(config.speech_tokens_dict_size, self.dim)
        
        max_text_seq_len = config.max_text_tokens + 2
        self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)
        
        max_mel_seq_len = config.max_speech_tokens + 2 + 2
        self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

    def load_weights(self, state_dict: dict):
        self.load_state_dict(state_dict, strict=False)
        
        text_position_ids = torch.arange(self.config.max_text_tokens + 2, device=self.text_pos_emb.emb.weight.device)
        self.precomputed_text_pos_emb = self.text_pos_emb.get_fixed_embedding(text_position_ids)[0]
        
        speech_position_ids = torch.arange(self.config.max_speech_tokens + 2 + 2, device=self.speech_pos_emb.emb.weight.device)
        self.precomputed_speech_pos_emb = self.speech_pos_emb.get_fixed_embedding(speech_position_ids)[0]

    def forward(self, text_ids: torch.Tensor, cond: Optional[T3Cond] = None) -> torch.Tensor:
        """
        Builds the prefix embeddings for a request.
        Format: <| cond | text | start_speech |>
        """
        device = self.text_emb.weight.device
        
        # Text embeddings
        text_ids = text_ids.to(device)
        text_e = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)]
        
        # Start speech token
        start_speech_token = torch.tensor([self.config.start_speech_token], device=device)
        start_speech_e = self.speech_emb(start_speech_token) + self.precomputed_speech_pos_emb[0:1]
        
        if cond is not None:
            cond = cond.to(device)
            cond_e = self.cond_enc(cond)
            # Turbo has no CFG, so we just return the direct concatenation
            return torch.cat([cond_e, text_e, start_speech_e], dim=0)
        else:
            return torch.cat([text_e, start_speech_e], dim=0)
