本文记录medium challenges的优化过程。

## 3. 2D Convolution

### Basic

```python
import torch
import triton
import triton.language as tl

@triton.jit
def conv(
    input,
    kernel,
    output,
    input_row,
    input_cols,
    output_rows,
    output_cols,
    kernel_rows:tl.constexpr,
    kernel_cols:tl.constexpr,
    KERNEL_BLOCK_ROWS:tl.constexpr,
    KERNEL_BLOCK_COLS:tl.constexpr

):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    # KERNEL_BLOCK_ROWS = triton.next_power_of_2(kernel_rows)
    # KERNEL_BLOCK_COLS = triton.next_power_of_2(kernel_cols)

    offset_x = pid_x + tl.arange(0, KERNEL_BLOCK_ROWS)
    offset_y = pid_y + tl.arange(0, KERNEL_BLOCK_COLS)
    data_ptr = input + offset_x[:, None] * input_cols + offset_y[None, :]
    mask = (offset_x[:, None] < input_row) & (offset_y[None, :] < input_cols)
    data = tl.load(data_ptr, mask, other=0)

    weight_offset_x = tl.arange(0, KERNEL_BLOCK_ROWS)
    weight_offset_y = tl.arange(0, KERNEL_BLOCK_COLS)
    weight_ptr = kernel + weight_offset_x[:, None] * kernel_cols + weight_offset_y[None, :]
    weight_mask = (weight_offset_x[:, None] < kernel_rows) & (weight_offset_y[None, :] < kernel_cols)
    weight = tl.load(weight_ptr, weight_mask, other=0)

    result = tl.sum(data * weight)
    result_offset = pid_x * output_cols + pid_y
    tl.store(output + result_offset, result)


# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    grid = (output_rows, output_cols)
    KERNEL_BLOCK_ROWS = triton.next_power_of_2(kernel_rows)
    KERNEL_BLOCK_COLS = triton.next_power_of_2(kernel_cols)
    conv[grid](
        input,
        kernel,
        output,
        input_rows,
        input_cols,
        output_rows,
        output_cols,
        kernel_rows,
        kernel_cols,
        KERNEL_BLOCK_ROWS,
        KERNEL_BLOCK_COLS
    )
```

### 简化mask & 引入num_warps=1

```python
import torch
import triton
import triton.language as tl

@triton.jit
def conv(
    input,
    kernel,
    output,
    input_row,
    input_cols,
    output_rows,
    output_cols,
    kernel_rows:tl.constexpr,
    kernel_cols:tl.constexpr,
    KERNEL_BLOCK_ROWS:tl.constexpr,
    KERNEL_BLOCK_COLS:tl.constexpr

):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    weight_offset_x = tl.arange(0, KERNEL_BLOCK_ROWS)
    weight_offset_y = tl.arange(0, KERNEL_BLOCK_COLS)
    weight_ptr = kernel + weight_offset_x[:, None] * kernel_cols + weight_offset_y[None, :]
    weight_mask = (weight_offset_x[:, None] < kernel_rows) & (weight_offset_y[None, :] < kernel_cols)
    weight = tl.load(weight_ptr, weight_mask, other=0)

    offset_x = pid_x + weight_offset_x #tl.arange(0, KERNEL_BLOCK_ROWS)
    offset_y = pid_y + weight_offset_y #tl.arange(0, KERNEL_BLOCK_COLS)
    data_ptr = input + offset_x[:, None] * input_cols + offset_y[None, :]
    # mask = (offset_x[:, None] < input_row) & (offset_y[None, :] < input_cols)
    # data = tl.load(data_ptr, mask, other=0)
    data = tl.load(data_ptr, weight_mask, other=0)

    result = tl.sum(data * weight)
    result_offset = pid_x * output_cols + pid_y
    tl.store(output + result_offset, result)


# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    grid = (output_rows, output_cols)
    KERNEL_BLOCK_ROWS = triton.next_power_of_2(kernel_rows)
    KERNEL_BLOCK_COLS = triton.next_power_of_2(kernel_cols)
    conv[grid](
        input,
        kernel,
        output,
        input_rows,
        input_cols,
        output_rows,
        output_cols,
        kernel_rows,
        kernel_cols,
        KERNEL_BLOCK_ROWS,
        KERNEL_BLOCK_COLS,
        # tips num_warps=1是​​控制线程块（Thread Block）内 Warp 数量的关键参数​​，直接影响并行度和硬件资源利用率
        num_warps=1
    )
```

