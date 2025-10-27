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
````