本文记录medium challenges的优化过程。

## 5. Batch Normalization

### Basic

```python
import torch
import triton
import triton.language as tl

@triton.jit
def bn_kernel(input, 
    gamma, 
    beta, 
    output, 
    N_POWER_2:tl.constexpr,
    C_POWER_2:tl.constexpr,
    N:tl.constexpr,
    C:tl.constexpr,
    eps,
    block_size:tl.constexpr
):
    input = input.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    n_offset = tl.arange(0, N_POWER_2)
    c_offset = pid
    data_offset = n_offset[:, None] * C + c_offset
    mask = (n_offset[:, None] < N) & (c_offset[None, :] < C)
    data = tl.load(input + data_offset, mask, other=0)
    mean = tl.div_rn(tl.sum(data), N)
    mean_exp = tl.div_rn(tl.sum(data * data), N)
    var = tl.rsqrt(mean_exp - mean * mean + eps)

    cur_gamma = tl.load(gamma + pid)
    cur_beta = tl.load(beta + pid)
    result = cur_gamma * (data - mean) * var + cur_beta;
    tl.store(output + data_offset, result, mask)


# input, gamma, beta, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, 
          output: torch.Tensor, N: int, C: int, eps: float):
    
    block_size = 256
    C_POWER_2 = triton.next_power_of_2(C)
    N_POWER_2 = triton.next_power_of_2(N)
    grid = (C_POWER_2, 1)
    bn_kernel[grid](input, 
        gamma, 
        beta, 
        output, 
        N_POWER_2,
        C_POWER_2,
        N,
        C,
        eps,
        block_size
    )
```

### 增加Block_size_C

```python
import torch
import triton
import triton.language as tl

@triton.jit
def bn_kernel(input, 
    gamma, 
    beta, 
    output, 
    N_POWER_2:tl.constexpr,
    # C_POWER_2:tl.constexpr,
    N:tl.constexpr,
    C:tl.constexpr,
    eps,
    block_size_c:tl.constexpr
):
    input = input.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    n_offset = tl.arange(0, N_POWER_2)
    c_offset = pid * block_size_c + tl.arange(0, block_size_c)
    data_offset = n_offset[:, None] * C + c_offset
    mask = (n_offset[:, None] < N) & (c_offset[None, :] < C)
    data = tl.load(input + data_offset, mask, other=0)
    mean = tl.div_rn(tl.sum(data, axis=0, keep_dims=True), N)
    mean_exp = tl.div_rn(tl.sum(data * data, axis=0, keep_dims=True), N)
    var = tl.rsqrt(mean_exp - mean * mean + eps)

    mask_C = c_offset < C
    cur_gamma = tl.load(gamma + c_offset, mask_C, other=0)
    cur_beta = tl.load(beta + c_offset, mask_C, other=0)
    result = cur_gamma * (data - mean) * var + cur_beta;
    tl.store(output + data_offset, result, mask)


# input, gamma, beta, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, 
          output: torch.Tensor, N: int, C: int, eps: float):
    block_size_c = 4 #256
    # C_POWER_2 = triton.next_power_of_2(C)
    N_POWER_2 = triton.next_power_of_2(N)
    grid = (triton.cdiv(C, block_size_c), 1)
    bn_kernel[grid](input, 
        gamma, 
        beta, 
        output, 
        N_POWER_2,
        # C_POWER_2,
        N,
        C,
        eps,
        block_size_c
    )
```

## 6. RMS Normalization

### Basic

```python
import torch
import triton
import triton.language as tl

@triton.jit
def rms_sum_kernel(input, 
    gamma, 
    beta, 
    output, 
    N,
    eps,
    rms,
    block_size:tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * block_size + tl.arange(0, block_size)
    mask = offset < N
    data = tl.load(input + offset, mask, other=0)
    sum = tl.sum(data * data)
    tl.atomic_add(rms, sum)

@triton.jit
def rms_kernel(input, 
    gamma, 
    beta, 
    output, 
    N,
    eps,
    rms_ptr,
    block_size:tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * block_size + tl.arange(0, block_size)
    mask = offset < N
    data = tl.load(input + offset, mask, other=0)
    rms = tl.load(rms_ptr)
    data = data * tl.rsqrt(rms / N + eps)
    # data = data * rms
    result = gamma * data + beta
    tl.store(output + offset, result, mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: float, beta: float, 
          output: torch.Tensor, N: int, eps: float):
    block_size = 2048
    grid=(triton.cdiv(N, block_size),)
    rms = torch.zeros((1,), dtype=torch.float32, device='cuda')
    rms_sum_kernel[grid](input, gamma, beta, output, N, eps, rms, block_size)
    rms_kernel[grid](input, gamma, beta, output, N, eps, rms, block_size)
```