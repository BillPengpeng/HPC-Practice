# TORCH_LOGS="output_code" python3 pytorch_square_compiler.py

import torch

def square_2(a):
    return a * a

opt_square_2 = torch.compile(square_2)
opt_square_2(torch.randn(100, 100).cuda())