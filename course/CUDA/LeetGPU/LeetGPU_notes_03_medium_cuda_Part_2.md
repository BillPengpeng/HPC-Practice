本文记录medium challenges的优化过程。

## 3. 2D Convolution

### Basic

```c
#include <cuda_runtime.h>

__global__ void  convolution(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int output_rows, int output_cols, 
           int kernel_rows, int kernel_cols, int shared_mem_row, int shared_mem_col)
{
    extern __shared__ float arr[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    for (int x = threadIdx.x; x < shared_mem_row; x += blockDim.x)
    {
        for (int y = threadIdx.y; y < shared_mem_col; y += blockDim.y)
        {
            int dst_idx = x * shared_mem_col + y;
            int src_idx = blockDim.x * blockIdx.x + x; // - kernel_rows / 2;
            int src_idy = blockDim.y * blockIdx.y + y; // - kernel_cols / 2;
            if (src_idx < input_rows && src_idy < input_cols)
                arr[dst_idx] = input[src_idx * input_cols + src_idy];
        }
    }
    __syncthreads();

    if (idx < output_rows && idy < output_cols)
    {
        float sum = 0;
        for (int i = 0; i < kernel_rows; i++)
        {
            int arr_i = threadIdx.x + i;
            for (int j = 0; j < kernel_cols; j++)
            {
                int arr_j = threadIdx.y + j;
                int kernel_idx = i * kernel_cols + j;
                int arr_idx = arr_i * shared_mem_col + arr_j;
                sum += kernel[kernel_idx] * arr[arr_idx];
            }
        }
        output[idx * output_cols + idy] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int tile_size = 32; //256;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    dim3 threadsPerBlock(tile_size, tile_size);
    dim3 blocksPerGrid((output_rows + tile_size - 1) / tile_size, \
                       (output_cols + tile_size - 1) / tile_size);
    int shared_mem_row = tile_size + kernel_rows - 1;
    int shared_mem_col = tile_size + kernel_cols - 1;
    int shared_mem_size = shared_mem_row * shared_mem_col * sizeof(float);
    convolution<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(input, kernel, output, 
                                                                   input_rows, input_cols, 
                                                                   output_rows, output_cols, 
                                                                   kernel_rows, kernel_cols,
                                                                   shared_mem_row, shared_mem_col);
    cudaDeviceSynchronize();
}
```

### Constant Memory

```c
#include <cuda_runtime.h>

__constant__ float F[1024];

__global__ void  convolution(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int output_rows, int output_cols, 
           int kernel_rows, int kernel_cols, int shared_mem_row, int shared_mem_col)
{
    extern __shared__ float arr[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    for (int x = threadIdx.x; x < shared_mem_row; x += blockDim.x)
    {
        for (int y = threadIdx.y; y < shared_mem_col; y += blockDim.y)
        {
            int dst_idx = x * shared_mem_col + y;
            int src_idx = blockDim.x * blockIdx.x + x; // - kernel_rows / 2;
            int src_idy = blockDim.y * blockIdx.y + y; // - kernel_cols / 2;
            if (src_idx < input_rows && src_idy < input_cols)
                arr[dst_idx] = input[src_idx * input_cols + src_idy];
        }
    }
    __syncthreads();

    if (idx < output_rows && idy < output_cols)
    {
        float sum = 0;
        for (int i = 0; i < kernel_rows; i++)
        {
            int arr_i = threadIdx.x + i;
            for (int j = 0; j < kernel_cols; j++)
            {
                int arr_j = threadIdx.y + j;
                int kernel_idx = i * kernel_cols + j;
                int arr_idx = arr_i * shared_mem_col + arr_j;
                sum += F[kernel_idx] * arr[arr_idx];
            }
        }
        output[idx * output_cols + idy] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int tile_size = 32; //256;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    dim3 threadsPerBlock(tile_size, tile_size);
    dim3 blocksPerGrid((output_rows + tile_size - 1) / tile_size, \
                       (output_cols + tile_size - 1) / tile_size);
    int shared_mem_row = tile_size + kernel_rows - 1;
    int shared_mem_col = tile_size + kernel_cols - 1;
    int shared_mem_size = shared_mem_row * shared_mem_col * sizeof(float);
    cudaMemcpyToSymbol(F, kernel, sizeof(float) * kernel_rows * kernel_cols);
    convolution<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(input, kernel, output, 
                                                                   input_rows, input_cols, 
                                                                   output_rows, output_cols, 
                                                                   kernel_rows, kernel_cols,
                                                                   shared_mem_row, shared_mem_col);
    cudaDeviceSynchronize();
}
```

### 简化For循环

```c
#include <cuda_runtime.h>

__constant__ float F[1024];

__global__ void  convolution(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int output_rows, int output_cols, 
           int kernel_rows, int kernel_cols, int shared_mem_row, int shared_mem_col)
{
    extern __shared__ float arr[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    for (int x = threadIdx.x, src_idx = idx; x < shared_mem_row; x += blockDim.x, src_idx += blockDim.x)
    {
        for (int y = threadIdx.y, src_idy = idy; y < shared_mem_col; y += blockDim.y, src_idy += blockDim.y)
        {
            int dst_idx = x * shared_mem_col + y;
            if (src_idx < input_rows && src_idy < input_cols)
                arr[dst_idx] = input[src_idx * input_cols + src_idy];
        }
    }
    __syncthreads();

    if (idx < output_rows && idy < output_cols)
    {
        float sum = 0;
        for (int i = 0, arr_i = threadIdx.x; i < kernel_rows; i++, arr_i++)
        {
            for (int j = 0, arr_j = threadIdx.y; j < kernel_cols; j++, arr_j++)
            {
                int kernel_idx = i * kernel_cols + j;
                int arr_idx = arr_i * shared_mem_col + arr_j;
                sum += F[kernel_idx] * arr[arr_idx];
            }
        }
        output[idx * output_cols + idy] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int tile_size = 32; //256;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    dim3 threadsPerBlock(tile_size, tile_size);
    dim3 blocksPerGrid((output_rows + tile_size - 1) / tile_size, \
                       (output_cols + tile_size - 1) / tile_size);
    int shared_mem_row = tile_size + kernel_rows - 1;
    int shared_mem_col = tile_size + kernel_cols - 1;
    int shared_mem_size = shared_mem_row * shared_mem_col * sizeof(float);
    cudaMemcpyToSymbol(F, kernel, sizeof(float) * kernel_rows * kernel_cols);
    convolution<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(input, kernel, output, 
                                                                   input_rows, input_cols, 
                                                                   output_rows, output_cols, 
                                                                   kernel_rows, kernel_cols,
                                                                   shared_mem_row, shared_mem_col);
    cudaDeviceSynchronize();
}
```