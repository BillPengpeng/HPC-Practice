本文记录easy challenges的优化过程。

## 1. ReLU

```c
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        output[idx] = fmaxf(input[idx], 0);
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

## 2. Leaky ReLU

```c
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        float x = input[idx];
        output[idx] = x > 0 ? x : (0.01*x);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

## 3. Color Inversion

```c
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int N = width * height;
    if (idx < N)
    {
        image[4 * idx + 0] = 255 - image[4 * idx + 0];
        image[4 * idx + 1] = 255 - image[4 * idx + 1];
        image[4 * idx + 2] = 255 - image[4 * idx + 2];
    }

}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
```

## 4. Matrix Copy

```c
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N * N)
    {
        B[idx] = A[idx];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
} 
```

## 5. Matrix Transpose

```c
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < cols && idy < rows)
    {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
```

## 6. Count 2D Array Element

```c
#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < M && idy < N)
    {
        int pos = idx * N + idy;
        atomicAdd(output, input[pos] == K);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}
```

## 7. Reverse Array

```c
#include <cuda_runtime.h>

const int BLOCK_SIZE = 256;

__global__ void reverse_array(float* input, int N) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    __shared__ float s_data[BLOCK_SIZE];
    if (idx < N)
    {
        int idx_1 = idx / 2;
        int idx_2 = N - 1 - idx_1;
        if (tid % 2 == 0)
        {
            s_data[tid] = input[idx_1];
        }
        else
        {
            s_data[tid] = input[idx_2];
        }
        __syncthreads();  // 等待所有线程完成写入共享内存
        if (tid % 2 == 1)
        {
            input[idx_1] = s_data[tid];
        }
        else
        {
            input[idx_2] = s_data[tid];
        }
    }
}
```

```c
#include <cuda_runtime.h>

const int BLOCK_SIZE = 256;

__global__ void reverse_array(float* input, int N) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < N / 2) {
        float a = input[id];
        float b = input[N - 1 - id];
a
        input[N - 1 - id] = a;
        input[id] = b;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
```