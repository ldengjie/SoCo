import os
import torch
from lm_eval import evaluator
from lm_eval.utils import make_table
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoConfig
from modules import ABLinear
from utils import inject_context

#raw_model
#model_id='drlidj/llama-7b-hf'
#ab_model
model_id='drlidj/llama-7b-hf_c0.2'
#model_id='drlidj/llama-7b-hf_c0.4'
#model_id='drlidj/llama-7b-hf_c0.6'
#model_id='drlidj/llama-7b-hf_c0.8'

with inject_context(model_id, ABLinear if hasattr(AutoConfig.from_pretrained(model_id),'linear_info') else None):
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,trust_remote_code=True).to('cuda:0')
print(f'model: {model}')

results = evaluator.simple_evaluate(
    model=HFLM(pretrained=model,trust_remote_code=True,use_fast_tokenizer=False),
    tasks=['arc_easy','openbookqa','winogrande','hellaswag','piqa','mathqa','truthfulqa'],
    bootstrap_iters=1,
)

print(make_table(results))
