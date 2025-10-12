本文记录easy challenges的优化过程。

## 8. Count Array Element

```python
import torch
import triton
import triton.language as tl

@triton.jit
def count_equal_kernel(input_ptr, output_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < N
    x = tl.load(input_ptr + block_start, mask)
    y = tl.sum(x == K)
    tl.atomic_add(output_ptr, y)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    count_equal_kernel[grid](input, output, N, K, BLOCK_SIZE=BLOCK_SIZE)
```

## 9. Sigmoid Linear Unit

```python
import torch
import triton
import triton.language as tl

@triton.jit
def silu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(input + offset, mask)
    y = x / (1 + tl.exp(-x))
    tl.store(output + offset, y, mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    silu_kernel[grid](input, output, N, BLOCK_SIZE)
```

## 10. Swish-Gated Linear Unit

```python
import torch
import triton
import triton.language as tl

@triton.jit
def swiglu(
    input, output, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < (N // 2)
    x1 = tl.load(input + offset, mask)
    x2 = tl.load(input + offset + N // 2, mask)
    y = x1 / (1 + tl.exp(-x1)) * x2
    tl.store(output + offset, y, mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    swiglu[grid](
        input, output, N, BLOCK_SIZE=BLOCK_SIZE
    )
```

## 11. 1D Convolution

```python
import torch
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    input, kernel, output,
    input_size, kernel_size,
    BLOCK_SIZE: tl.constexpr
):
    output_size = input_size - kernel_size + 1
    pid = tl.program_id(axis=0)
    results = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for i in range(kernel_size):
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + i
        mask = offset < input_size
        data = tl.load(input + offset, mask)
        weight = tl.load(kernel + i)
        results += data * weight
    write_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    write_mask = write_offset < output_size
    tl.store(output + write_offset, results, write_mask)

# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)
    
    conv1d_kernel[grid](
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE
    )
```

## 12. Rainbow Table

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fnv1a_hash(x):
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261
    
    hash_val = tl.full(x.shape, OFFSET_BASIS, tl.uint32)
    
    for byte_pos in range(4):
        byte = (x >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME
    
    return hash_val

@triton.jit
def fnv1a_hash_kernel(
    input,
    output,
    n_elements,
    n_rounds,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(input + offset, mask)
    y = tl.cast(x, tl.uint32)
    for i in range(n_rounds):
        y = fnv1a_hash(y)
    tl.store(output + offset, y, mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fnv1a_hash_kernel[grid](
        input,
        output,
        N,
        R,
        BLOCK_SIZE
    )
```

## 13. Matrix Multiplication

### Basic

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a, b, c, 
    M, N, K,
    stride_am, stride_an, 
    stride_bn, stride_bk, 
    stride_cm, stride_ck
):
    tile: tl.constexpr = 16
    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)
    a = a.to(tl.pointer_type(tl.float32))
    b = b.to(tl.pointer_type(tl.float32))
    c = c.to(tl.pointer_type(tl.float32))

    sum = 0.0
    for idx in range((N + tile - 1) // tile):
        a_offset = pid_y * stride_am + idx * tile + tl.arange(0, tile)
        a_mask = a_offset < M * N
        b_offset = pid_x + idx * tile * K + tl.arange(0, tile) * K
        b_mask = b_offset < N * K
        x1 = tl.load(a + a_offset, a_mask)
        x2 = tl.load(b + b_offset, b_mask)
        sum += tl.sum(x1 * x2)

    c_offset = pid_y * stride_cm + pid_x * stride_ck
    tl.store(c + c_offset, sum)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1 
    stride_bn, stride_bk = K, 1  
    stride_cm, stride_ck = K, 1
    
    grid = (M, K) 
    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck
    )
```

### Tile

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a, b, c, 
    M, N, K,
    stride_am, stride_an, 
    stride_bn, stride_bk, 
    stride_cm, stride_ck,
    tite: tl.constexpr
):
    pid_y = tl.program_id(axis=0) 
    pid_x = tl.program_id(axis=1) 
    a = a.to(tl.pointer_type(tl.float32))
    b = b.to(tl.pointer_type(tl.float32))
    c = c.to(tl.pointer_type(tl.float32))

    sum = tl.zeros((tite, tite), dtype=tl.float32) 
    offset_a = pid_y * tite + tl.arange(0, tite)
    offset_b = pid_x * tite + tl.arange(0, tite)
    a_ptr = a + offset_a[:, None] * stride_am
    b_ptr = b + offset_b[None, :] * stride_bk
    a_mask = offset_a[:, None] < M
    b_mask = offset_b[None, :] < K
    for idx in range(N):
        x1 = tl.load(a_ptr + idx * stride_an, a_mask)
        x2 = tl.load(b_ptr + idx * stride_bn, b_mask)
        sum += x1 * x2;

    c_ptrs = c + offset_a[:, None] * stride_cm + offset_b[None, :] * stride_ck
    offset_c = (offset_a[:, None] < M) & (offset_b[None, :] < K)
    tl.store(c_ptrs, sum, offset_c)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1 
    stride_bn, stride_bk = K, 1  
    stride_cm, stride_ck = K, 1
    tite = 16
    
    # grid = (M, K) 
    grid = ((M + tite - 1) // tite, (K + tite - 1) // tite)
    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        tite
    )
```

### Tile v2

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a, b, c, 
    M, N, K,
    stride_am, stride_an, 
    stride_bn, stride_bk, 
    stride_cm, stride_ck,
    tile_mk: tl.constexpr,
    tile_n: tl.constexpr
):
    pid_y = tl.program_id(axis=0) 
    pid_x = tl.program_id(axis=1) 

    sum = tl.zeros((tile_mk, tile_mk), dtype=tl.float32) 
    offset_am = pid_y * tile_mk + tl.arange(0, tile_mk) 
    offset_bk = pid_x * tile_mk + tl.arange(0, tile_mk)
    a_ptr = a + offset_am[:, None] * stride_am 
    b_ptr = b + offset_bk[None, :] * stride_bk
    # for idx in range((N + tile_n - 1) // tile_n):
    for idx in tl.range(0, tl.cdiv(N, tile_n)):
        offset_an_bn = tl.arange(0, tile_n) + idx * tile_n
        a_mask = (offset_am[:, None] < M) & (offset_an_bn[None, :] < N)
        b_mask = (offset_bk[None, :] < K) & (offset_an_bn[:, None] < N)
        x1 = tl.load(a_ptr + stride_an * offset_an_bn[None, :], a_mask, other=0.0)
        x2 = tl.load(b_ptr + stride_bn * offset_an_bn[:, None], b_mask, other=0.0)
        # sum = tl.dot(x1, x2, sum);
        sum = tl.dot(x1, x2, sum, input_precision="ieee");

    c_ptrs = c + offset_am[:, None] * stride_cm + offset_bk[None, :] * stride_ck
    c_mask = (offset_am[:, None] < M) & (offset_bk[None, :] < K)
    tl.store(c_ptrs, sum, c_mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1 
    stride_bn, stride_bk = K, 1  
    stride_cm, stride_ck = K, 1
    tile_mk = 64 #32
    tile_n = 32 #16
    
    # grid = (M, K) 
    grid = ((M + tile_mk - 1) // tile_mk, (K + tile_mk - 1) // tile_mk)
    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        tile_mk,
        tile_n
    )
```

### 为提高L2 缓存命中率而进行程序重排序

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a, b, c, 
    M, N, K,
    stride_am, stride_an, 
    stride_bn, stride_bk, 
    stride_cm, stride_ck,
    tile_mk: tl.constexpr,
    tile_n: tl.constexpr,
    group_size_m: tl.constexpr
):
    # pid_y = tl.program_id(axis=0) 
    # pid_x = tl.program_id(axis=1) 
    pid = tl.program_id(axis=0) 
    grid_m = tl.cdiv(M, tile_mk)
    grid_k = tl.cdiv(K, tile_mk)
    # pid_y = pid // grid_n
    # pid_x = pid % grid_n
    num_pid_in_group = grid_k * group_size_m
    group_id = pid // num_pid_in_group 
    cur_pid_m = group_id * group_size_m
    cur_group_size = min(grid_m - cur_pid_m, group_size_m)
    # m方向分组，此处改为列优先
    # pid_y = cur_pid_m + ((pid % num_pid_in_group) // group_size_m)
    # pid_x = (pid % num_pid_in_group) % group_size_m
    # 此处cur_group_size比较关键
    pid_y = cur_pid_m + ((pid % num_pid_in_group) % cur_group_size)
    pid_x = (pid % num_pid_in_group) // cur_group_size


    sum = tl.zeros((tile_mk, tile_mk), dtype=tl.float32) 
    offset_am = pid_y * tile_mk + tl.arange(0, tile_mk) 
    offset_bk = pid_x * tile_mk + tl.arange(0, tile_mk)
    a_ptr = a + offset_am[:, None] * stride_am 
    b_ptr = b + offset_bk[None, :] * stride_bk
    for idx in tl.range(0, tl.cdiv(N, tile_n)):
        offset_an_bn = tl.arange(0, tile_n) + idx * tile_n
        a_mask = (offset_am[:, None] < M) & (offset_an_bn[None, :] < N)
        b_mask = (offset_bk[None, :] < K) & (offset_an_bn[:, None] < N)
        x1 = tl.load(a_ptr + stride_an * offset_an_bn[None, :], a_mask, other=0.0)
        x2 = tl.load(b_ptr + stride_bn * offset_an_bn[:, None], b_mask, other=0.0)
        sum = tl.dot(x1, x2, sum);

    c_ptrs = c + offset_am[:, None] * stride_cm + offset_bk[None, :] * stride_ck
    c_mask = (offset_am[:, None] < M) & (offset_bk[None, :] < K)
    tl.store(c_ptrs, sum, c_mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1 
    stride_bn, stride_bk = K, 1  
    stride_cm, stride_ck = K, 1
    tile_mk = 64 #32
    tile_n = 32 #16
    group_size_m = 4 #2
    
    # grid = (M, K) 
    # grid = ((M + tile_mk - 1) // tile_mk, (K + tile_mk - 1) // tile_mk)
    grid = (triton.cdiv(M, tile_mk) * triton.cdiv(K, tile_mk), 1)
    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        tile_mk,
        tile_n,
        group_size_m
    )
```