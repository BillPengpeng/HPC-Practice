本文记录medium challenges的优化过程。

## 1. Reduction

### Basic

```c
#include <cuda_runtime.h>

#define tile_size 256
#define num_per_thread 32 //4

__global__ void reduction(const float* input, float* output, int N) {
    int start_idx = (blockDim.x * blockIdx.x + threadIdx.x) * num_per_thread;
    __shared__ float sum[tile_size];
    sum[threadIdx.x] = 0.0f;

    for (int i = 0; i < num_per_thread && start_idx + i < N; i++)
    {
        sum[threadIdx.x] += input[start_idx + i];
    }
    // __syncthreads();
    for (int stride = tile_size / 2; stride >= 1; stride >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < stride)
        {
            sum[threadIdx.x] += sum[threadIdx.x + stride];
        }
    }
    if (0 == threadIdx.x)
        atomicAdd(output, sum[0]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = tile_size;
    int num_per_block = num_per_thread * threadsPerBlock;
    int blocksPerGrid = (N + num_per_block - 1) / num_per_block;
    reduction<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

### 控制内存发散（Memory Divergence）

```c
#include <cuda_runtime.h>

#define tile_size 256
#define num_per_thread 32 //4

__global__ void reduction(const float* input, float* output, int N) {
    // int start_idx = (blockDim.x * blockIdx.x + threadIdx.x) * num_per_thread;
    int start_idx = blockDim.x * blockIdx.x * num_per_thread + threadIdx.x;
    __shared__ float sum[tile_size];
    sum[threadIdx.x] = 0.0f;

    // for (int i = 0; i < num_per_thread && start_idx + i < N; i++)
    for (int i = 0; i < num_per_thread && start_idx + i * tile_size < N; i++)
    {
        // sum[threadIdx.x] += input[start_idx + i];
        sum[threadIdx.x] += input[start_idx + i * tile_size];
    }
    // __syncthreads();
    for (int stride = tile_size / 2; stride >= 1; stride >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < stride)
        {
            sum[threadIdx.x] += sum[threadIdx.x + stride];
        }
    }
    if (0 == threadIdx.x)
        atomicAdd(output, sum[0]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = tile_size;
    int num_per_block = num_per_thread * threadsPerBlock;
    int blocksPerGrid = (N + num_per_block - 1) / num_per_block;
    reduction<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

### 增加Warp Reduce

```c
#include <cuda_runtime.h>

#define tile_size 256
#define num_per_thread 32 //4

__device__ void warpRedue(volatile float*sum, int tid)
{
    sum[tid] += sum[tid + 32];
    sum[tid] += sum[tid + 16];
    sum[tid] += sum[tid + 8];
    sum[tid] += sum[tid + 4];
    sum[tid] += sum[tid + 2];
    sum[tid] += sum[tid + 1];
}

__global__ void reduction(const float* input, float* output, int N) {
    // int start_idx = (blockDim.x * blockIdx.x + threadIdx.x) * num_per_thread;
    int start_idx = blockDim.x * blockIdx.x * num_per_thread + threadIdx.x;
    __shared__ float sum[tile_size];
    sum[threadIdx.x] = 0.0f;

    // for (int i = 0; i < num_per_thread && start_idx + i < N; i++)
    for (int i = 0; i < num_per_thread && start_idx + i * tile_size < N; i++)
    {
        // sum[threadIdx.x] += input[start_idx + i];
        sum[threadIdx.x] += input[start_idx + i * tile_size];
    }
    __syncthreads();
    // for (int stride = tile_size / 2; stride >= 1; stride >>= 1)
    for (int stride = tile_size / 2; stride > 32; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            sum[threadIdx.x] += sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // warp reduce
    if (threadIdx.x < 32) 
    {
        // sum[threadIdx.x] += sum[threadIdx.x + 32];
        // sum[threadIdx.x] += sum[threadIdx.x + 16];
        // sum[threadIdx.x] += sum[threadIdx.x + 8];
        // sum[threadIdx.x] += sum[threadIdx.x + 4];
        // sum[threadIdx.x] += sum[threadIdx.x + 2];
        // sum[threadIdx.x] += sum[threadIdx.x + 1];
        warpRedue(sum, threadIdx.x);
    }
    if (0 == threadIdx.x)
        atomicAdd(output, sum[0]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = tile_size;
    int num_per_block = num_per_thread * threadsPerBlock;
    int blocksPerGrid = (N + num_per_block - 1) / num_per_block;
    reduction<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

### 两次Warp Reduce

```c
#include <cuda_runtime.h>

#define tile_size 256
#define warp_size 32
#define num_per_thread 32 //4

__device__ void warpRedue(volatile float*sum, int tid)
{
    sum[tid] += sum[tid + 32];
    sum[tid] += sum[tid + 16];
    sum[tid] += sum[tid + 8];
    sum[tid] += sum[tid + 4];
    sum[tid] += sum[tid + 2];
    sum[tid] += sum[tid + 1];
}

__global__ void reduction(const float* input, float* output, int N) {
    // int start_idx = (blockDim.x * blockIdx.x + threadIdx.x) * num_per_thread;
    int start_idx = blockDim.x * blockIdx.x * num_per_thread + threadIdx.x;
    __shared__ float warp_sum[tile_size >> 5]; /// warp_size];

    float sum = 0.0f;
    for (int i = 0; i < num_per_thread && start_idx + i * tile_size < N; i++)
    {
        sum += input[start_idx + i * tile_size];
    }
    // __syncthreads();

    // warp reduce
    #pragma unroll
    for (int stride = 16; stride >= 1; stride >>= 1)
    {
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    }
    int lane = threadIdx.x & (warp_size - 1);
    int warpId = threadIdx.x >> 5; /// 32;
    if (0 == lane) warp_sum[warpId] = sum;
    __syncthreads();

    // warp reduce
    if (0 == warpId) {
        // float block_sum = ((lane < tile_size / warp_size) ? warp_sum[lane] : 0.0f);
        float block_sum = (lane < (tile_size >> 5) ? warp_sum[lane] : 0.0f);
        #pragma unroll
        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, stride);
        }
        if (0 == lane)
            atomicAdd(output, block_sum);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = tile_size;
    int num_per_block = num_per_thread * threadsPerBlock;
    int blocksPerGrid = (N + num_per_block - 1) / num_per_block;
    reduction<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

## 2. Softmax

### Basic

```c
#include <cuda_runtime.h>
#include <cfloat>

__device__ float atomicMaxFloat(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float max_arr[32]; // = { -FLT_MAX };

    float val = idx < N ? input[idx] : -FLT_MAX;
    for (int stride = 16; stride >= 1; stride >>= 1)
    {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, stride));
    }
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;
    if (0 == lane_id)
        max_arr[warp_id] = val;
    __syncthreads();

    float block_max = 0;
    if (0 == warp_id)
    {
        block_max = (lane_id < (blockDim.x >> 5) ? max_arr[lane_id] : -FLT_MAX);
        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, stride));
        }
        if (0 == threadIdx.x)
            atomicMaxFloat(output, block_max);
    }
}

__global__ void sum_kernel(const float* input, float* output, float* max_val, float* result_sum, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sum_arr[32]; // = { 0 };

    float val = idx < N ? expf(input[idx] - *max_val) : 0;
    if (idx < N) output[idx] = val;
    float sum = val;
    for (int stride = 16; stride >= 1; stride >>= 1)
    {
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    }
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;
    if (0 == lane_id)
        sum_arr[warp_id] = sum;
    __syncthreads();

    float block_sum = 0;
    if (0 == warp_id)
    {
        block_sum = (lane_id < (blockDim.x >> 5) ? sum_arr[lane_id] : 0);
        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, stride);
        }
    }
    if (0 == threadIdx.x)
        atomicAdd(result_sum, block_sum);
}

__global__ void softmax_kernel(const float* input, float* output, float* max_val, float* result_sum, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) output[idx] /= *result_sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float *result_sum, *max_val;
    cudaMalloc(&result_sum, 1 * sizeof(float));
    cudaMalloc(&max_val, 1 * sizeof(float));

    max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, max_val, N);
    sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, max_val, result_sum, N);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, max_val, result_sum, N);
    cudaFree(result_sum);
    cudaFree(max_val);

    cudaDeviceSynchronize();
}
```

### 增加coarse

```c
#include <cuda_runtime.h>
#include <cfloat>

#define coarse_size 32

__device__ float atomicMaxFloat(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(const float* input, float* output, int N) {
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = blockDim.x * blockIdx.x * coarse_size + threadIdx.x;
    __shared__ float max_arr[32];
    float val = -FLT_MAX; 
    for (int i = 0; i < coarse_size; i++, idx += blockDim.x)
    {
        if (idx < N) val = fmaxf(val, input[idx]);
    }

    for (int stride = 16; stride >= 1; stride >>= 1)
    {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, stride));
    }
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;
    if (0 == lane_id)
        max_arr[warp_id] = val;
    __syncthreads();

    float block_max = 0;
    if (0 == warp_id)
    {
        block_max = (lane_id < (blockDim.x >> 5) ? max_arr[lane_id] : -FLT_MAX);
        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, stride));
        }
    }
    if (0 == threadIdx.x)
        atomicMaxFloat(output, block_max);
}

__global__ void sum_kernel(const float* input, float* output, float* max_val, float* result_sum, int N) {
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = blockDim.x * blockIdx.x * coarse_size + threadIdx.x;
    __shared__ float sum_arr[32]; // = { 0 };

    float val = 0;
    float sum = 0;
    for (int i = 0; i < coarse_size; i++, idx += blockDim.x)
    {
        if (idx < N) 
        {
            val = expf(input[idx] - *max_val);
            output[idx] = val;
            sum += val;
        }
    }

    for (int stride = 16; stride >= 1; stride >>= 1)
    {
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    }
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;
    if (0 == lane_id)
        sum_arr[warp_id] = sum;
    __syncthreads();

    float block_sum = 0;
    if (0 == warp_id)
    {
        block_sum = (lane_id < (blockDim.x >> 5) ? sum_arr[lane_id] : 0);
        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, stride);
        }
    }
    if (0 == threadIdx.x)
        atomicAdd(result_sum, block_sum);
}

__global__ void softmax_kernel(const float* input, float* output, float* max_val, float* result_sum, int N) {
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = blockDim.x * blockIdx.x * coarse_size + threadIdx.x;
    float sum = *result_sum;
    for (int i = 0; i < coarse_size; i++, idx += blockDim.x)
    {
        if (idx < N) output[idx] /= sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = (N + threadsPerBlock*coarse_size - 1) / (threadsPerBlock*coarse_size);
    float *result_sum, *max_val;
    cudaMalloc(&result_sum, 1 * sizeof(float));
    cudaMalloc(&max_val, 1 * sizeof(float));

    max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, max_val, N);
    sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, max_val, result_sum, N);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, max_val, result_sum, N);
    cudaFree(result_sum);
    cudaFree(max_val);

    cudaDeviceSynchronize();
}
```