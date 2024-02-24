import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig

# 设置自定义的缓存目录
# os.environ["TRANSFORMERS_CACHE"] = "/mnt/mydisk/"
model_path = '/mnt/mydisk/xverse-XVERSE-7B-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model.generation_config = GenerationConfig.from_pretrained(model_path)
model = model.eval()
history = [{"role": "user", "content": "1955年谁是美国总统？他是什么党派？"}]
response = model.chat(tokenizer, history)
print(response)
history.append({"role": "assistant", "content": response})
history.append({"role": "user", "content": "他任职了多少年"})
response = model.chat(tokenizer, history)
print(response)
