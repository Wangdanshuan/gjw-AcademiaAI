import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = '/mnt/mydisk/xverse-XVERSE-7B'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')
model = model.eval()
inputs = tokenizer('北京的景点：故宫、天坛、万里长城等。\n深圳的景点：', return_tensors='pt').input_ids
inputs = inputs.cuda()
generated_ids = model.generate(inputs, max_new_tokens=64, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.1)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
