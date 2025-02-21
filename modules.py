import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ABLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=False, dtype=None):
        super().__init__()
        self.info_str     = f'ABLinear(in_features={in_features}, out_features={out_features}, rank={rank}, bias=True, dtype={dtype})'
        self.out_features = out_features
        self.in_features  = in_features
        self.A            = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.B            = nn.Linear(rank, out_features, bias=True, dtype=dtype)
    def forward(self, x):
        if max(self.out_features,self.in_features)<math.prod(x.shape[:-1]):
            return F.linear(x, self.B.weight@self.A.weight, self.B.bias)
        else:
            return self.B(self.A(x))
    def __repr__(self):
        return self.info_str
