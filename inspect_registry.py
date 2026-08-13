import inspect
from vllm.model_executor.models.registry import _RegisteredModel
print(inspect.getsource(_RegisteredModel.from_model_cls))
print(inspect.getsource(_RegisteredModel))
