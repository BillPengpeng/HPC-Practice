本文记录medium challenges的优化过程。

## 7. Histogramming

### Basic

```python
import torch
import triton
import triton.language as tl

@triton.jit
def hist(input,
    histogram,
    N,
    num_bins,
    block_size:tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * block_size + tl.arange(0, block_size)
    mask = offset < N
    data = tl.load(input + offset, mask)
    tl.atomic_add(histogram + data, 1, mask)


# input, histogram are tensors on the GPU
def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    block_size = 1024
    grid = (triton.cdiv(N, block_size), )
    hist[grid](input, histogram, N, num_bins, block_size)
```

## 8. Dot Product

### Basic

```python
import torch
import triton
import triton.language as tl

@triton.jit
def dot_product(
    a,
    b,
    result,
    n,
    block_size:tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * block_size + tl.arange(0, block_size)
    mask = offset < n
    data_a = tl.load(a + offset, mask)
    data_b = tl.load(b + offset, mask)
    tl.atomic_add(result, tl.sum(data_a * data_b))

# a, b, result are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int):
    block_size = 1024
    grid = (triton.cdiv(n, block_size), )
    dot_product[grid](a, b, result, n, block_size)
```

## 9. Mean Squared Error

### Basic

```python
import torch
import triton
import triton.language as tl

@triton.jit
def dot_product(
    predictions,
    targets,
    mse,
    N,
    block_size:tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * block_size + tl.arange(0, block_size)
    mask = offset < N
    data_a = tl.load(predictions + offset, mask)
    data_b = tl.load(targets + offset, mask)
    result = (data_a - data_b) * (data_a - data_b) / N
    tl.atomic_add(mse, tl.sum(result))

# predictions, targets, mse are tensors on the GPU
def solve(predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
    block_size = 1024
    grid = (triton.cdiv(N, block_size), )
    dot_product[grid](predictions, targets, mse, N, block_size)
```