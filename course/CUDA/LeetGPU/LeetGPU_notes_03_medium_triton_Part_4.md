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

