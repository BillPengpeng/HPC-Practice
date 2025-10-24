本文记录medium challenges的优化过程。

## 5. Batch Normalization

### Basic

```c
#include <cuda_runtime.h>

__global__ void mean_sum_kernel(const float* input, float* sum, int N, int C)
{
    // N
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // if (idx >= N) return;

    __shared__ float mean_sum[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;

    for (int k = 0; k < C; k++)
    {
        // tip: zero
        float cur_sum = (idx < N ? input[idx * C + k] : 0);
        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            cur_sum += __shfl_down_sync(0xffffffff, cur_sum, stride);
        }
        if (0 == lane_id)
        {
            mean_sum[warp_id] = cur_sum;
        }
        __syncthreads();
        if (0 == warp_id)
        {
            float final_sum = lane_id < (blockDim.x >> 5) ? mean_sum[lane_id] : 0;
            for (int stride = 16; stride >= 1; stride >>= 1)
            {
                final_sum += __shfl_down_sync(0xffffffff, final_sum, stride);
            }
            if (0 == lane_id)
                atomicAdd(&sum[k], final_sum);
        }
        if (k < C - 1) __syncthreads();
    }
}

__global__ void mean_kernel(float* sum, int N, int C)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= C) return;
    sum[idx] /= N;
}

__global__ void var_sum_kernel(const float* input, const float* mean, float* output, float* sum, int N, int C)
{
    __shared__ float var_sum[32];

    // N
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // if (idx >= N) return;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;

    for (int k = 0; k < C; k++)
    {
        // tip: zero
        float dif = (idx < N ? input[idx * C + k] - mean[k] : 0);
        if (idx < N) output[idx * C + k] = dif;
        float cur_sum = (dif * dif);

        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            cur_sum += __shfl_down_sync(0xffffffff, cur_sum, stride);
        }
        
        if (0 == lane_id)
        {
            var_sum[warp_id] = cur_sum;
        }
        __syncthreads();
        if (0 == warp_id)
        {
            float final_sum = lane_id < (blockDim.x >> 5) ? var_sum[lane_id] : 0;
            for (int stride = 16; stride >= 1; stride >>= 1)
            {
                final_sum += __shfl_down_sync(0xffffffff, final_sum, stride);
            }
            if (0 == lane_id)
                atomicAdd(&sum[k], final_sum);
        }
        if (k < C - 1) __syncthreads();
    }
}

__global__ void bn_kernel(float* mean, float* var, const float* gamma, const float* beta, float* output, int N, int C, float eps)
{
    // N
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // C
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx >= C || idy >= N) return;
    float result = gamma[idx] * (output[idy * C + idx] / sqrtf(var[idx] + eps)) + beta[idx];
    output[idy * C + idx] = result;
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, 
                     float* output, int N, int C, float eps) {
    int threadsPerBlock = 256;
    int blocksPerGrid_N = (N + threadsPerBlock - 1) / threadsPerBlock;

    // mean_sum
    float *mean, *var;
    cudaMalloc(&mean, sizeof(float) * C);
    cudaMalloc(&var, sizeof(float) * C);
    mean_sum_kernel<<<blocksPerGrid_N, threadsPerBlock>>>(input, mean, N, C);

    // mean
    int blocksPerGrid_C = (C + threadsPerBlock - 1) / threadsPerBlock;
    mean_kernel<<<blocksPerGrid_C, threadsPerBlock>>>(mean, N, C);

    // var_sum
    var_sum_kernel<<<blocksPerGrid_N, threadsPerBlock>>>(input, mean, output, var, N, C);

    // mean
    mean_kernel<<<blocksPerGrid_C, threadsPerBlock>>>(var, N, C);

    // final
    int tile_size = 32;
    dim3 threadsPerBlock2D(tile_size, tile_size);
    dim3 blocksPerGrid2D((C + tile_size - 1) / tile_size, (N + tile_size - 1) / tile_size);
    bn_kernel<<<blocksPerGrid2D, threadsPerBlock2D>>>(mean, var, gamma, beta, output, N, C, eps);

    cudaDeviceSynchronize();
    cudaFree(mean);  
    cudaFree(var);        
}
```

### BN计算公式简化

```c
#include <cuda_runtime.h>

__global__ void mean_sum_kernel(const float* input, float* sum, float* sum_exp, int N, int C)
{
    // N
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // if (idx >= N) return;

    __shared__ float mean_sum[32];
    __shared__ float mean_sum_exp[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;

    for (int k = 0; k < C; k++)
    {
        // tip: zero
        float cur_sum = (idx < N ? input[idx * C + k] : 0);
        float cur_sum_exp = cur_sum * cur_sum;
        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            cur_sum += __shfl_down_sync(0xffffffff, cur_sum, stride);
            cur_sum_exp += __shfl_down_sync(0xffffffff, cur_sum_exp, stride);
        }
        if (0 == lane_id)
        {
            mean_sum[warp_id] = cur_sum;
            mean_sum_exp[warp_id] = cur_sum_exp;
        }
        __syncthreads();
        if (0 == warp_id)
        {
            float final_sum = lane_id < (blockDim.x >> 5) ? mean_sum[lane_id] : 0;
            float final_sum_exp = lane_id < (blockDim.x >> 5) ? mean_sum_exp[lane_id] : 0;
            for (int stride = 16; stride >= 1; stride >>= 1)
            {
                final_sum += __shfl_down_sync(0xffffffff, final_sum, stride);
                final_sum_exp += __shfl_down_sync(0xffffffff, final_sum_exp, stride);
            }
            if (0 == lane_id)
            {
                atomicAdd(&sum[k], final_sum);
                atomicAdd(&sum_exp[k], final_sum_exp);
            }
        }
        if (k < C - 1) __syncthreads();
    }
}

__global__ void calc_mean_var(const float* sum, const float* sum_exp, float* mean, float* var, int N, int C, float eps)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= C) return;
    mean[idx] = sum[idx] / N;
    var[idx] = rsqrtf(sum_exp[idx] / N - mean[idx] * mean[idx] + eps);
}
__global__ void bn_kernel(float* mean, float* var, const float* gamma, const float* beta, const float* input, float* output, int N, int C, float eps)
{
    // N
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // C
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx >= C || idy >= N) return;
    float result = gamma[idx] * (input[idy * C + idx] - mean[idx]) * var[idx] + beta[idx];
    output[idy * C + idx] = result;
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, 
                     float* output, int N, int C, float eps) {
    int threadsPerBlock = 256;
    int blocksPerGrid_N = (N + threadsPerBlock - 1) / threadsPerBlock;

    // mean_sum
    float *sum, *sum_exp, *mean, *var;
    cudaMalloc(&sum, sizeof(float) * C);
    cudaMalloc(&sum_exp, sizeof(float) * C);
    cudaMalloc(&mean, sizeof(float) * C);
    cudaMalloc(&var, sizeof(float) * C);
    mean_sum_kernel<<<blocksPerGrid_N, threadsPerBlock>>>(input, sum, sum_exp, N, C);

    // mean
    int blocksPerGrid_C = (C + threadsPerBlock - 1) / threadsPerBlock;
    calc_mean_var<<<blocksPerGrid_C, threadsPerBlock>>>(sum, sum_exp, mean, var, N, C, eps);

    // final
    int tile_size = 32;
    dim3 threadsPerBlock2D(tile_size, tile_size);
    dim3 blocksPerGrid2D((C + tile_size - 1) / tile_size, (N + tile_size - 1) / tile_size);
    bn_kernel<<<blocksPerGrid2D, threadsPerBlock2D>>>(mean, var, gamma, beta, input, output, N, C, eps);

    cudaDeviceSynchronize();
    cudaFree(sum);  
    cudaFree(sum_exp);      
    cudaFree(mean);  
    cudaFree(var);        
}
```

### BN合并计算过程

```c
#include <cuda_runtime.h>

__global__ void bn_kernel(const float* input, const float* gamma, const float* beta, 
                                float* output, int N, int C, float eps)
{
    // N
    int k = blockIdx.x;
    int idx = threadIdx.x;

    __shared__ float mean_sum[32];
    __shared__ float mean_sum_exp[32];
    __shared__ float mean, var;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;

    float cur_sum = 0;
    float cur_sum_exp = 0;
    
    for (int i = idx; i < N; i += blockDim.x)
    {
        float val = input[i * C + k];
        cur_sum += val;
        cur_sum_exp += val * val;
    }
    for (int stride = 16; stride >= 1; stride >>= 1)
    {
        cur_sum += __shfl_down_sync(0xffffffff, cur_sum, stride);
        cur_sum_exp += __shfl_down_sync(0xffffffff, cur_sum_exp, stride);
    }
    if (0 == lane_id)
    {
        mean_sum[warp_id] = cur_sum;
        mean_sum_exp[warp_id] = cur_sum_exp;
    }
    __syncthreads();
    
    if (0 == warp_id)
    {
        float final_sum = lane_id < (blockDim.x >> 5) ? mean_sum[lane_id] : 0;
        float final_sum_exp = lane_id < (blockDim.x >> 5) ? mean_sum_exp[lane_id] : 0;
        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            final_sum += __shfl_down_sync(0xffffffff, final_sum, stride);
            final_sum_exp += __shfl_down_sync(0xffffffff, final_sum_exp, stride);
        }
        if (0 == lane_id)
        {
            mean = final_sum / N;
            var = rsqrtf(final_sum_exp / N - mean * mean + eps);
        }
    }
    __syncthreads();
    float cur_gamma = gamma[k];
    float cur_beta = beta[k];
    for (int i = idx; i < N; i += blockDim.x)
    {
        output[i * C + k] = cur_gamma * (input[i * C + k] - mean) * var + cur_beta;
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, 
                     float* output, int N, int C, float eps) {
    int threadsPerBlock = 1024; //256;
    bn_kernel<<<C, threadsPerBlock>>>(input, gamma, beta, output, N, C, eps);
    cudaDeviceSynchronize();
      
}
```

## 6. RMS Normalization

### Basic

```c
#include <cuda_runtime.h>

__global__ void var_sum_kernel(const float* input, float* sum_exp, int N)
{
    // N
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // if (idx >= N) return;

    __shared__ float mean_sum_exp[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;

    // tip: zero
    float cur_sum = (idx < N ? input[idx] : 0);
    float cur_sum_exp = cur_sum * cur_sum;
    for (int stride = 16; stride >= 1; stride >>= 1)
    {
        cur_sum_exp += __shfl_down_sync(0xffffffff, cur_sum_exp, stride);
    }
    if (0 == lane_id)
    {
        mean_sum_exp[warp_id] = cur_sum_exp;
    }
    __syncthreads();
    if (0 == warp_id)
    {
        float final_sum_exp = lane_id < (blockDim.x >> 5) ? mean_sum_exp[lane_id] : 0;
        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            final_sum_exp += __shfl_down_sync(0xffffffff, final_sum_exp, stride);
        }
        if (0 == lane_id)
        {
            atomicAdd(sum_exp, final_sum_exp);
        }
    }
}

__global__ void rms_kernel(float* sum_exp, float gamma, float beta, const float* input, float* output, int N, float eps)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    float rms = rsqrtf(*sum_exp / N + eps);
    float result = gamma * input[idx] * rms + beta;
    output[idx] = result;
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, 
                     float* output, int N, float eps) {
    int threadsPerBlock = 256;
    int blocksPerGrid_N = (N + threadsPerBlock - 1) / threadsPerBlock;

    // mean_sum
    float *sum_exp;
    cudaMalloc(&sum_exp, sizeof(float));
    var_sum_kernel<<<blocksPerGrid_N, threadsPerBlock>>>(input, sum_exp, N);

    // final
    rms_kernel<<<blocksPerGrid_N, threadsPerBlock>>>(sum_exp, gamma, beta, input, output, N, eps);

    cudaDeviceSynchronize();
    cudaFree(sum_exp);                                  
}
```

### RMS合并计算过程

```c
#include <cuda_runtime.h>

__global__ void var_sum_kernel(const float* input, float gamma, float beta, float* output, int N, float eps)
{
    // N
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // if (idx >= N) return;

    __shared__ float mean_sum_exp[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;

    // tip: zero
    float cur_sum_exp = 0;
    for (int i = idx; i < N; i += blockDim.x) 
    {
        float val = input[i];
        cur_sum_exp += val * val;
    }
    for (int stride = 16; stride >= 1; stride >>= 1)
    {
        cur_sum_exp += __shfl_down_sync(0xffffffff, cur_sum_exp, stride);
    }
    if (0 == lane_id)
    {
        mean_sum_exp[warp_id] = cur_sum_exp;
    }
    __syncthreads();
    __shared__ float rms;
    if (0 == warp_id)
    {
        float final_sum_exp = lane_id < (blockDim.x >> 5) ? mean_sum_exp[lane_id] : 0;
        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            final_sum_exp += __shfl_down_sync(0xffffffff, final_sum_exp, stride);
        }
        if (0 == lane_id)
            rms = rsqrtf(final_sum_exp / N + eps);
    }
    __syncthreads();
    for (int i = idx; i < N; i += blockDim.x) 
    {
        output[i] = gamma * input[i] * rms + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, 
                     float* output, int N, float eps) {
    int threadsPerBlock = 1024; //512; //256;
    // int blocksPerGrid_N = (N + threadsPerBlock - 1) / threadsPerBlock;

    // mean_sum
    var_sum_kernel<<<1, threadsPerBlock>>>(input, gamma, beta, output, N, eps);
    cudaDeviceSynchronize();                                  
}
```