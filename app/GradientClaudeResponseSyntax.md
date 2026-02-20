Gradient Claude Response Syntax

'''python
cd ~/code_assist
pipenv run python3
from gradient import Gradient
from dotenv import load_dotenv
import os
load_dotenv('.env')
GRADIENT_MODEL_ACCESS_KEY = os.environ.get("GRADIENT_MODEL_ACCESS_KEY")
client = Gradient(access_token=GRADIENT_MODEL_ACCESS_KEY)
response = client.chat.completions.create(
	messages=[{"role": "user", "content": "Just quick test; plz respond briefly.",}],
	model="anthropic-claude-opus-4.6",
	max_tokens=1024,
	temperature=0.2
)
generated_text = response.choices[0].message.content
print(generated_text)  # Hello! I'm here and ready to help. What's up? 😊
print(type(response))  # <class 'gradient.types.chat.completion_create_response.CompletionCreateResponse'>
print(dir(response))  # ['__abstractmethods__', '__annotations__', '__class__', '__class_getitem__', '__class_vars__', '__copy__', '__deepcopy__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__fields__', '__fields_set__', '__format__', '__ge__', '__get_pydantic_core_schema__', '__get_pydantic_json_schema__', '__getattr__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__pretty__', '__private_attributes__', '__pydantic_complete__', '__pydantic_computed_fields__', '__pydantic_core_schema__', '__pydantic_custom_init__', '__pydantic_decorators__', '__pydantic_extra__', '__pydantic_fields__', '__pydantic_fields_set__', '__pydantic_generic_metadata__', '__pydantic_init_subclass__', '__pydantic_on_complete__', '__pydantic_parent_namespace__', '__pydantic_post_init__', '__pydantic_private__', '__pydantic_root_model__', '__pydantic_serializer__', '__pydantic_setattr_handlers__', '__pydantic_validator__', '__reduce__', '__reduce_ex__', '__replace__', '__repr__', '__repr_args__', '__repr_name__', '__repr_recursion__', '__repr_str__', '__rich_repr__', '__setattr__', '__setstate__', '__signature__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl', '_calculate_keys', '_copy_and_set_values', '_get_value', '_iter', '_setattr_handler', 'choices', 'construct', 'copy', 'created', 'dict', 'from_orm', 'id', 'json', 'model', 'model_computed_fields', 'model_config', 'model_construct', 'model_copy', 'model_dump', 'model_dump_json', 'model_extra', 'model_fields', 'model_fields_set', 'model_json_schema', 'model_parametrized_name', 'model_post_init', 'model_rebuild', 'model_validate', 'model_validate_json', 'model_validate_strings', 'object', 'parse_file', 'parse_obj', 'parse_raw', 'schema', 'schema_json', 'to_dict', 'to_json', 'update_forward_refs', 'usage', 'validate']
usage = response.usage
print(usage)  # CompletionUsage(completion_tokens=20, prompt_tokens=16, total_tokens=36, cache_created_input_tokens=0, cache_creation={'ephemeral_1h_input_tokens': 0, 'ephemeral_5m_input_tokens': 0}, cache_read_input_tokens=0)
if response.usage:
    print({
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    })
# {'input_tokens': 16, 'output_tokens': 20, 'total_tokens': 36}
full = response.model_dump()
print(full)  # {'id': '', 'choices': [{'finish_reason': 'stop', 'index': 0, 'logprobs': None, 'message': {'content': "Hello! I'm here and ready to help. What's up? 😊", 'reasoning_content': None, 'refusal': None, 'role': 'assistant', 'tool_calls': None}}], 'created': 1771099688, 'model': 'anthropic-claude-opus-4.6', 'object': 'chat.completion', 'usage': {'completion_tokens': 20, 'prompt_tokens': 16, 'total_tokens': 36, 'cache_created_input_tokens': 0, 'cache_creation': {'ephemeral_1h_input_tokens': 0, 'ephemeral_5m_input_tokens': 0}, 'cache_read_input_tokens': 0}}
import json
print(json.dumps(response.model_dump(), indent=2))
'''
{
  "id": "",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "Hello! I'm here and ready to help. What's up? \ud83d\ude0a",
        "reasoning_content": null,
        "refusal": null,
        "role": "assistant",
        "tool_calls": null
      }
    }
  ],
  "created": 1771099688,
  "model": "anthropic-claude-opus-4.6",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 20,
    "prompt_tokens": 16,
    "total_tokens": 36,
    "cache_created_input_tokens": 0,
    "cache_creation": {
      "ephemeral_1h_input_tokens": 0,
      "ephemeral_5m_input_tokens": 0
    },
    "cache_read_input_tokens": 0
  }
}

'''python
estimated_cost = (
    response.usage.prompt_tokens * INPUT_RATE
  + response.usage.completion_tokens * OUTPUT_RATE
) / 1_000_000
'''
