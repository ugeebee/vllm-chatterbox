import sys
if not hasattr(sys, 'get_int_max_str_digits'):
    def get_int_max_str_digits() -> int: return 4300
    def set_int_max_str_digits(maxdigits: int) -> None: pass
    sys.get_int_max_str_digits = get_int_max_str_digits
    sys.set_int_max_str_digits = set_int_max_str_digits

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
    
    # Load the raw weights into prefix builder
    from safetensors.torch import load_file
    raw_weights = load_file("/home/ssm-user/models/chatterbox-turbo/t3_turbo_v1.safetensors")
    prefix_builder.load_weights(raw_weights)
    
    prefix_builder.eval()

    # Initialize vLLM AsyncLLMEngine
    engine_args = AsyncEngineArgs(
        model="/home/ssm-user/models/chatterbox-turbo/t3-hf-format",
        enforce_eager=True, # Bypasses CUDAGraph capture which deadlocks custom architectures
    )
    # For now we assume engine loads the registered T3TurboForCausalLM
    engine = AsyncLLMEngine.from_engine_args(engine_args)


@app.post("/generate_speech")
async def generate_speech(
    text: str = Form(...)
):
    # Load the default condition from the conds.pt file you downloaded
    default_cond_dict = torch.load("/home/ssm-user/models/chatterbox-turbo/conds.pt", map_location="cpu", weights_only=True)
    cond = T3Cond(**default_cond_dict["default"])

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
