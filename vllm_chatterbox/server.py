from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import uvicorn
import torch
import json
import asyncio
from typing import Optional
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType

from .modules.t3_config import T3Config
from .modules.cond_enc import T3Cond
from .prefix import PrefixBuilder

app = FastAPI(title="T3 Turbo API")

engine = None
prefix_builder = None
config = T3Config()

@app.on_event("startup")
async def startup_event():
    global engine, prefix_builder
    
    # Prefix builder logic
    prefix_builder = PrefixBuilder(config)
    # We would normally load weights into prefix_builder here
    # e.g., prefix_builder.load_weights(torch.load("path_to_t3_weights.pt"))
    prefix_builder.eval()

    # Initialize vLLM AsyncLLMEngine
    engine_args = AsyncEngineArgs(
        model="dummy_path_since_weights_are_loaded_custom", # This needs to point to a valid HF format dir if needed
        # We can also register our plugin explicitly if not picked up via entry points
        worker_use_ray=False,
    )
    # For now we assume engine loads the registered T3TurboForCausalLM
    engine = AsyncLLMEngine.from_engine_args(engine_args)


@app.post("/generate_speech")
async def generate_speech(
    text: str = Form(...),
    reference_audio: Optional[UploadFile] = None
):
    # Dummy processing of reference audio/cond
    # In a real setup, we would extract speaker_emb, emotion, etc.
    if reference_audio:
        cond = T3Cond(speaker_emb=torch.randn(1, 256)) # Mock condition
    else:
        cond = None

    # Text tokenization (mock using config defaults)
    # text_ids should come from an EnTokenizer or similar
    text_ids = torch.randint(1, config.max_text_tokens, (len(text.split()),))

    # Precompute prefix
    with torch.no_grad():
        prefix_embeds = prefix_builder(text_ids, cond=cond)

    # Convert to standard vLLM PromptType format
    # The prompt_token_ids length must match the length of prompt_embeds 
    # to pass vLLM's internal length checks, even if the ids are ignored.
    prompt = {
        "prompt_token_ids": [0] * len(prefix_embeds),
        "prompt_embeds": prefix_embeds
    }

    from vllm.sampling_params import SamplingParams
    sampling_params = SamplingParams(
        temperature=1.0,
        top_k=50,
        stop_token_ids=[config.stop_speech_token],
        max_tokens=config.max_speech_tokens
    )

    request_id = f"req-{torch.randint(0, 1000000, (1,)).item()}"
    
    # Generate tokens via AsyncLLMEngine
    async def token_generator():
        results_generator = engine.generate(prompt, sampling_params, request_id)
        
        last_yielded_len = 0
        async for request_output in results_generator:
            for output in request_output.outputs:
                new_tokens = output.token_ids[last_yielded_len:]
                if new_tokens:
                    yield json.dumps({"tokens": new_tokens}) + "\n"
                    last_yielded_len += len(new_tokens)

    return StreamingResponse(token_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
