本文记录easy challenges的优化过程。

## 1. ReLU

```python
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(input + offset, mask)
    y = max(x, 0)
    tl.store(output + offset, y, mask)
    
# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE)
```

## 2. Leaky ReLU

```python
import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(
    input,
    output,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(input + offset, mask)
    y = tl.where(x > 0, x, 0.01*x)
    tl.store(output + offset, y, mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](
        input,
        output,
        N,
        BLOCK_SIZE
    )
```

## 3. Color Inversion

```python
import torch
import triton
import triton.language as tl

@triton.jit
def invert_kernel(
    image,
    width, height,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE * 4 + tl.arange(0, BLOCK_SIZE * 4)
    mask = (offset < width * height * 4) & (offset % 4 != 3)
    x = tl.load(image + offset, mask)
    y = 255 - x
    tl.store(image + offset, y, mask)

# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    
    invert_kernel[grid](
        image,
        width, height,
        BLOCK_SIZE
    ) 
```

## 4. Matrix Copy

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matrix_copy(a, b, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < N*N
    x = tl.load(a + block_start, mask)
    tl.store(b + block_start, x, mask)


# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N*N, BLOCK_SIZE),)
    
    matrix_copy[grid](
        a,
        b,
        N,
        BLOCK_SIZE
    )  
```

## 5. Matrix Transpose

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    input, output,
    rows, cols,
    stride_ir, stride_ic,  
    stride_or, stride_oc
):
    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)
    block_start_x = pid_y * stride_ir + pid_x * stride_ic
    mask = block_start_x < rows * cols
    block_start_y = pid_x * stride_or + pid_y * stride_oc
    x = tl.load(input + block_start_x, mask)
    tl.store(output + block_start_y, x, mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1  
    stride_or, stride_oc = rows, 1
    
    grid = (rows, cols)
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc
    ) 
```

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    input, output,
    rows, cols,
    stride_ir, stride_ic,  
    stride_or, stride_oc,
    BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(axis=0) * BLOCK_SIZE
    j = tl.program_id(axis=1) * BLOCK_SIZE
    # [:, None]：将一维张量转为列向量（形状 (BLOCK_SIZE, 1)）​
    # [None, :]：将一维张量转为行向量（形状 (1, BLOCK_SIZE)）
    inp_offs = input + stride_ir * (i + tl.arange(0, BLOCK_SIZE)[:, None]) + \
               stride_ic * (j + tl.arange(0, BLOCK_SIZE)[None, :])
    inp_mask = ((i+tl.arange(0, BLOCK_SIZE)) < rows)[:, None] & \
               ((j+tl.arange(0, BLOCK_SIZE)) < cols)[None, :] 
    inp_vals = tl.load(inp_offs, mask=inp_mask)
    transposed = tl.trans(inp_vals)
    # Store
    out_offs = output + stride_or * (j + tl.arange(0, BLOCK_SIZE)[:, None]) + \
               stride_oc * (i+tl.arange(0, BLOCK_SIZE)[None, :])
    out_mask = ((j+tl.arange(0, BLOCK_SIZE)) < cols)[:, None] & \
               ((i+tl.arange(0, BLOCK_SIZE)) < rows)[None, :]
    tl.store(out_offs, transposed, mask=out_mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1  
    stride_or, stride_oc = rows, 1
    BLOCK_SIZE = 32
    grid = (rows+BLOCK_SIZE-1)//BLOCK_SIZE, (cols+BLOCK_SIZE-1)//BLOCK_SIZE

    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc,
        BLOCK_SIZE
    ) 
```

## 6. Count 2D Array Element

```python
import torch
import triton
import triton.language as tl

@triton.jit
def count_2d_array(input, output, N, M, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (block_start < M*N)
    x = tl.load(input + block_start, mask)
    y = tl.sum(x == K)
    tl.atomic_add(output, y)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M*N, BLOCK_SIZE),)
    
    count_2d_array[grid](
        input,
        output,
        N,
        M,
        K,
        BLOCK_SIZE
    )  
```

## 7. Reverse Array

```python
import torch
import triton
import triton.language as tl

@triton.jit
def reverse_kernel(
    input,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start_0 = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_start_1 = N - 1 - (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    mask_0 = block_start_0 < N
    mask_1 = block_start_1 >= 0
    x_0 = tl.load(input + block_start_0, mask_0)
    x_1 = tl.load(input + block_start_1, mask_1)
    tl.store(input + block_start_0, x_1, mask_0)
    tl.store(input + block_start_1, x_0, mask_1)

# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)
    
    reverse_kernel[grid](
        input,
        N,
        BLOCK_SIZE
    ) 
```