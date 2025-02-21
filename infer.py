import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modules import ABLinear
from utils import inject_context

#raw_model
#model_id='drlidj/llama-7b-hf'
#ab_model
model_id='drlidj/llama-7b-hf_c0.2'
#model_id='drlidj/llama-7b-hf_c0.4'
#model_id='drlidj/llama-7b-hf_c0.6'
#model_id='drlidj/llama-7b-hf_c0.8'

device='cuda:0'
with inject_context(model_id, ABLinear if hasattr(AutoConfig.from_pretrained(model_id),'linear_info') else None):
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,trust_remote_code=True).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
print(f'model: {model}')

inputs = tokenizer('What is the responsibility of the AI assistant?', return_tensors="pt").to(device)
generate_ids = model.generate(input_ids=inputs.input_ids, do_sample=True, top_k=50, max_length=128, top_p=0.95, temperature=0.97)
answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f'{answer}')
