import vllm

def register():
    # Register the model class with vLLM
    from .model import T3TurboForCausalLM
    vllm.ModelRegistry.register_model("GPT2LMHeadModel", T3TurboForCausalLM)

# Call it immediately when the module is imported
register()
