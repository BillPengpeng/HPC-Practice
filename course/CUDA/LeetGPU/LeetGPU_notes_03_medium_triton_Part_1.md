本文记录medium challenges的优化过程。

## 1. Reduction

### Basic

```python
import triton
import triton.language as tl

@triton.jit
def reduction(input, output, N, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * block_size + tl.arange(0, block_size)
    mask = offset < N
    data = tl.load(input + offset, mask)
    result = tl.sum(data)
    tl.atomic_add(output, result)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    reduction[grid](input, output, N, block_size=BLOCK_SIZE)
```

## 2. Softmax

### Basic

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_max_kernel(
    input, max_val_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    input = input.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data = tl.load(input + offset, mask)
    result = tl.max(data)
    tl.atomic_max(max_val_ptr, result)

@triton.jit
def softmax_sum_kernel(
    input, output, max_val_ptr, sum_val_ptr, N,
    BLOCK_SIZE: tl.constexpr
):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data = tl.load(input + offset, mask)
    max_val = tl.load(max_val_ptr)
    data = data - max_val
    exp_result = tl.exp(data)
    # tips tl.where
    exp_result = tl.where(mask, exp_result, 0)
    tl.store(output + offset, exp_result, mask)

    sum_result = tl.sum(exp_result)
    tl.atomic_add(sum_val_ptr, sum_result)

@triton.jit
def softmax_kernel(
    input, output, sum_val_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    data = tl.load(input + offset, mask)
    sum_val = tl.load(sum_val_ptr)
    result = data / sum_val
    tl.store(output + offset, result, mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    # max_val: torch.Tensor = 0
    # sum_val: torch.Tensor = 0
    max_val = torch.full((1,), float("-inf"), dtype=torch.float32, device='cuda') 
    sum_val = torch.zeros((1,), dtype=torch.float32, device='cuda')
    softmax_max_kernel[grid](input, max_val, N, BLOCK_SIZE)
    softmax_sum_kernel[grid](input, output, max_val, sum_val, N, BLOCK_SIZE)
    softmax_kernel[grid](output, output, sum_val, N, BLOCK_SIZE)
```