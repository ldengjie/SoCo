import sys
from contextlib import contextmanager
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights

def inject_model(model_dir,linear_cls,linear_info):
    #can't use from_pretrained inside the init_empty_weights. https://github.com/huggingface/accelerate/issues/1298#issuecomment-1499458279
    with init_empty_weights():
        original_model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_dir),torch_dtype=torch.bfloat16)
    original_model_cls=original_model.__class__
    model_name = original_model_cls.__name__
    print(f'--- inject_model original_model_cls {original_model_cls}, model_name {model_name}, linear_cls {nn.Linear} -> {linear_cls}')

    if linear_info is None and hasattr(original_model.config,'linear_info'): linear_info = original_model.config.linear_info
    class InjectedModel(original_model_cls):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.linear_name=[]
            self.old_linear_num=self.new_linear_num=self.rank_num=0.
            module_dict=dict(self.named_modules())
            for full_name,original_module in module_dict.items():
                if 'layers' in full_name and isinstance(original_module, nn.Linear):
                    rank=0
                    if linear_cls is not None:
                        rank = linear_info[full_name]['rank']
                        if rank>0:
                            new_linear = linear_cls(linear_info[full_name]['in_features'], linear_info[full_name]['out_features'], rank=rank, bias=linear_info[full_name]['bias'], dtype=original_module.weight.dtype)
                        else:
                            new_linear = nn.Linear(linear_info[full_name]['in_features'], linear_info[full_name]['out_features'], bias=linear_info[full_name]['bias'], dtype=original_module.weight.dtype)
                        parent_name,name = full_name.rsplit('.',1)
                        setattr(module_dict[parent_name], name, new_linear)
                    self.linear_name.append(full_name)
                    self.old_linear_num += original_module.in_features * original_module.out_features
                    self.new_linear_num += (original_module.in_features*rank + rank + rank*original_module.out_features) if rank>0 else (original_module.in_features * original_module.out_features)
                    self.rank_num       += rank*2
            if linear_info is not None: self.config.linear_info = linear_info
            print(f'*** compression ratio {1-self.new_linear_num/self.old_linear_num: .2%},  old_linear_num {self.old_linear_num/10**9: 6.2f}B, new_linear_num {self.new_linear_num/10**9: 6.2f}B, rank_num {self.rank_num/10**6: 6.2f}M, rank/old {self.rank_num/self.old_linear_num: 6.2e}, rank/new {self.rank_num/(self.new_linear_num+1e-5): 6.2e}')

    InjectedModel.__name__     = original_model_cls.__name__
    InjectedModel.__qualname__ = original_model_cls.__qualname__
    #必须,否则保存不了modeling代码
    InjectedModel.__module__   = original_model_cls.__module__
    return model_name,InjectedModel,original_model_cls

@contextmanager
def inject_context(model_dir,linear_cls=None,linear_info=None):
    model_name,injected_model_cls,original_model_cls=inject_model(model_dir,linear_cls,linear_info)
    for _,v in sys.modules.items():
        if hasattr(v,'__file__') and hasattr(v,model_name):
            setattr(v,model_name,injected_model_cls)
    yield
    for _,v in sys.modules.items():
        if hasattr(v,'__file__') and hasattr(v,model_name):
            setattr(v,model_name,original_model_cls)
