本文记录medium challenges的优化过程。

## 7. Histogramming

### Basic

```c
#include <cuda_runtime.h>

__global__ void hist(const int* input, int* histogram, int N, int num_bins) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ int nums[];
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    {
        nums[i] = 0;
    }
    __syncthreads();
    if (idx < N)
    {
        atomicAdd(&nums[input[idx]], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    {
        if (nums[i] > 0)
            atomicAdd(&histogram[i], nums[i]);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int size = num_bins * sizeof(int);
    hist<<<blocksPerGrid, threadsPerBlock, size>>>(input, histogram, N, num_bins);
}
```

### COARSE_SIZE = 32

```c
#include <cuda_runtime.h>

#define COARSE_SIZE 32 //16 //8 //4

__global__ void hist(const int* input, int* histogram, int N, int num_bins) {
    int idx = blockDim.x * blockIdx.x * COARSE_SIZE + threadIdx.x;
    extern __shared__ int nums[];
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    {
        nums[i] = 0;
    }
    __syncthreads();
    if (idx < N)
    {
        for (int i = 0, j = idx; i < COARSE_SIZE && j < N; i++, j += blockDim.x)
            atomicAdd(&nums[input[j]], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    {
        if (nums[i] > 0)
            atomicAdd(&histogram[i], nums[i]);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock = 1024; //256;
    int blocksPerGrid = (N + threadsPerBlock*COARSE_SIZE - 1) / (threadsPerBlock*COARSE_SIZE);
    int size = num_bins * sizeof(int);
    hist<<<blocksPerGrid, threadsPerBlock, size>>>(input, histogram, N, num_bins);
}
```

## 8. Dot Product

### Basic

```c
#include <cuda_runtime.h>

__global__ void dot_product(const float* A, const float* B, float* result, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float cur_sum = 0;
    if (idx < N) cur_sum = A[idx] * B[idx];

    __shared__ float sum_arr[32];
    for (int stride = 16; stride >= 1; stride >>= 1)
    {
        cur_sum += __shfl_down_sync(0xffffffff, cur_sum, stride);
    }

    int cur_warp_id = threadIdx.x / 32;
    int cur_lane_id = threadIdx.x & 31;
    if (0 == cur_lane_id) sum_arr[cur_warp_id] = cur_sum;
    __syncthreads();

    if (0 == cur_warp_id)
    {   
        float block_sum = (cur_lane_id < (blockDim.x >> 5) ? sum_arr[cur_lane_id] : 0);
        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, stride);
        }
        if (0 == cur_lane_id) atomicAdd(result, block_sum);
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dot_product<<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
```

## 9. Mean Squared Error

### Basic (error)

```c
#include <cuda_runtime.h>

__global__ void mean_squared_error(const float* predictions, const float* targets, float* mse, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float cur_sum = 0;
    if (idx < N) cur_sum = powf(predictions[idx] - targets[idx], 2);

    __shared__ float sum_arr[32];
    for (int stride = 16; stride >= 1; stride >>= 1)
    {
        cur_sum += __shfl_down_sync(0xffffffff, cur_sum, stride);
    }

    int cur_warp_id = threadIdx.x / 32;
    int cur_lane_id = threadIdx.x & 31;
    if (0 == cur_lane_id) sum_arr[cur_warp_id] = cur_sum;
    __syncthreads();

    if (0 == cur_warp_id)
    {   
        float block_sum = (cur_lane_id < (blockDim.x >> 5) ? sum_arr[cur_lane_id] : 0);
        for (int stride = 16; stride >= 1; stride >>= 1)
        {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, stride);
        }
        if (0 == cur_lane_id) atomicAdd(mse, block_sum / N);
    }
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    mean_squared_error<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, mse, N);
    cudaDeviceSynchronize();
}
```