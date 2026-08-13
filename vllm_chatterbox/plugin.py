import vllm

def register():
    # Register the model class with vLLM
    from .model import T3TurboForCausalLM
    vllm.ModelRegistry.register_model("T3TurboForCausalLM", T3TurboForCausalLM)
