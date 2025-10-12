本文记录easy challenges的优化过程。

## 8. Count Array Element

```c
#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N && input[idx] == K)
    {
        atomicAdd(output, 1);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}
```

## 9. Sigmoid Linear Unit

```c
#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        float x = input[idx];
        output[idx] = x / (1 + expf(-x));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

## 10. Swish-Gated Linear Unit

```c
#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < halfN) 
    {
        float x1 = input[idx];
        float x2 = input[idx + halfN];
        output[idx] = x1 / (1 + expf(-x1)) * x2;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
```

## 11. 1D Convolution

### Basic

```c
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;
    if (idx < output_size) 
    {
        float sum = 0;
        for (int i = 0; i < kernel_size; i++)
        {
            sum += input[idx + i] * kernel[i];
        }
        output[idx] = sum;
    }                                   
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
```


### Tile

```c
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int tile_size = blockDim.x + kernel_size - 1;
    extern __shared__ float M[];
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x)
    {
        int pos = blockDim.x * blockIdx.x + i;
        if (pos < input_size)
            M[i] = input[pos];
    }
    __syncthreads();

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;
    if (idx < output_size) 
    {
        float sum = 0;
        for (int i = 0; i < kernel_size; i++)
        {
            // sum += input[idx + i] * kernel[i];
            sum += M[threadIdx.x + i] * kernel[i];
        }
        output[idx] = sum;
    }                                   
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    int shared_mem_size = (threadsPerBlock + kernel_size - 1) * sizeof(float);

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
```

### Constant Memory

```c
#include <cuda_runtime.h>

__constant__ float F[2048];

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int tile_size = blockDim.x + kernel_size - 1;
    extern __shared__ float M[];
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x)
    {
        int pos = blockDim.x * blockIdx.x + i;
        if (pos < input_size)
            M[i] = input[pos];
    }
    __syncthreads();

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;
    if (idx < output_size) 
    {
        float sum = 0;
        for (int i = 0; i < kernel_size; i++)
        {
            // sum += input[idx + i] * kernel[i];
            sum += M[threadIdx.x + i] * F[i]; //kernel[i];
        }
        output[idx] = sum;
    }                                   
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    int shared_mem_size = (threadsPerBlock + kernel_size - 1) * sizeof(float);

    cudaMemcpyToSymbol(F, kernel, sizeof(float) * kernel_size);
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
```

## 12. Rainbow Table

```c
#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    
    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        unsigned int result = input[idx];
        for (int i = 0; i < R; i++)
            result = fnv1a_hash(result);
        output[idx] = result;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}
```

## 13. Matrix Multiplication

### Basic

```c
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;  // B:N * K
    int idy = blockDim.y * blockIdx.y + threadIdx.y;  // A:M * N
    if (idx < K && idy < M)
    {
        float sum = 0;
        for (int i = 0; i < N; i++)
        {
            sum += A[idy * N + i] * B[i * K + idx];
        }
        C[idy * K + idx] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

### TILE

```c
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;  // B:N * K
    int idy = blockDim.y * blockIdx.y + threadIdx.y;  // A:M * N

    // extern __shared__ float S_B[]; // tile_size_n * tile_size_k
    // extern __shared__ float S_A[]; // tile_size_m * tile_size_n
    __shared__ float S_B[TILE_SIZE][TILE_SIZE];
    __shared__ float S_A[TILE_SIZE][TILE_SIZE];
    int tile_size = blockDim.x;

    float sum = 0;
    for (int i = 0; i < N; i += tile_size)
    {
        if (threadIdx.y + i < N && idx < K)
        {
            S_B[threadIdx.y][threadIdx.x] = B[(threadIdx.y + i) * K + idx];
        }

        if (idy < M && threadIdx.x + i < N)
        {
            S_A[threadIdx.y][threadIdx.x] = A[idy * N + threadIdx.x + i];
        }
        __syncthreads();

        if (idx < K && idy < M)
        {
            for (int j = 0; j < tile_size; j++)
            {
                if (i + j < N)
                    sum += (S_A[threadIdx.y][j] * S_B[j][threadIdx.x]);
            }
           
        }
        __syncthreads();
    }
    if (idx < K && idy < M)
    {
        C[idy * K + idx] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // int shared_mem_size = (2 * 16 * 16) * sizeof(float);
    // matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(A, B, C, M, N, K);
     matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

### 双缓冲技术


```c
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;  // B:N * K
    int idy = blockDim.y * blockIdx.y + threadIdx.y;  // A:M * N

    __shared__ float S_B[2][TILE_SIZE][TILE_SIZE];
    __shared__ float S_A[2][TILE_SIZE][TILE_SIZE];
    int tile_size = blockDim.x;
    
    int load_idx = 0;
    int load_shared_idx = 0;
    if (threadIdx.y + load_idx < N && idx < K)
    {
        S_B[load_shared_idx][threadIdx.y][threadIdx.x] = B[(threadIdx.y + load_idx) * K + idx];
    }

    if (idy < M && threadIdx.x + load_idx < N)
    {
        S_A[load_shared_idx][threadIdx.y][threadIdx.x] = A[idy * N + threadIdx.x + load_idx];
    }
    load_idx += tile_size;
    __syncthreads();


    float sum = 0;
    for (int i = 0; i < N; i += tile_size)
    {
        
        int calc_shared_idx = load_shared_idx;
        load_shared_idx = 1 - load_shared_idx;
        if (load_idx < N)
        {
            if (threadIdx.y + load_idx < N && idx < K)
            {
                S_B[load_shared_idx][threadIdx.y][threadIdx.x] = B[(threadIdx.y + load_idx) * K + idx];
            }

            if (idy < M && threadIdx.x + load_idx < N)
            {
                S_A[load_shared_idx][threadIdx.y][threadIdx.x] = A[idy * N + threadIdx.x + load_idx];
            }
            load_idx += tile_size;
        }

        if (idx < K && idy < M)
        {
            for (int j = 0; j < tile_size; j++)
            {
                if (i + j < N)
                    sum += (S_A[calc_shared_idx][threadIdx.y][j] * S_B[calc_shared_idx][j][threadIdx.x]);
            }
           
        }       
        __syncthreads();
    }
    if (idx < K && idy < M)
    {
        C[idy * K + idx] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // int shared_mem_size = (2 * 16 * 16) * sizeof(float);
    // matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(A, B, C, M, N, K);
     matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```


### 双缓冲技术 + 非对称TILE


```c
#include <cuda_runtime.h>

#define TILE_SIZE     16
#define TILE_NUM_MK   8 //4 //2 //1 
#define TILE_SIZE_MK (TILE_SIZE * TILE_NUM_MK)
#define TILE_SIZE_N   TILE_SIZE

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;  // B:N * K
    // int idy = blockDim.y * blockIdx.y + threadIdx.y;  // A:M * N
    int idx = TILE_SIZE_MK * blockIdx.x + threadIdx.x;  // B:N * K
    int idy = TILE_SIZE_MK * blockIdx.y + threadIdx.y;  // A:M * N

    __shared__ float S_B[2][TILE_SIZE_N][TILE_SIZE_MK];
    __shared__ float S_A[2][TILE_SIZE_MK][TILE_SIZE_N];
    
    int load_idx = 0;
    int load_shared_idx = 0;

    for (int offset = 0; offset < TILE_SIZE_MK; offset += TILE_SIZE)
    {
        // if (load_idx + threadIdx.y < N && idx + offset < K)
        // {
        //     S_B[load_shared_idx][threadIdx.y][threadIdx.x + offset] = B[(load_idx + threadIdx.y) * K + idx + offset];
        // }
        S_B[load_shared_idx][threadIdx.y][threadIdx.x + offset] = (load_idx + threadIdx.y < N && idx + offset < K) ? 
                                                                  (B[(load_idx + threadIdx.y) * K + idx + offset]) : 0;
    }
    for (int offset = 0; offset < TILE_SIZE_MK; offset += TILE_SIZE)
    {
        // if (idy + offset < M && load_idx + threadIdx.x < N)
        // {
        //     S_A[load_shared_idx][threadIdx.y + offset][threadIdx.x] = A[(idy + offset) * N + load_idx + threadIdx.x];
        // }    
        S_A[load_shared_idx][threadIdx.y + offset][threadIdx.x] = (idy + offset < M && load_idx + threadIdx.x < N) ? 
                                                                  (A[(idy + offset) * N + load_idx + threadIdx.x]) : 0;
    }
    load_idx += TILE_SIZE_N;
    __syncthreads();

    float sum_A[TILE_NUM_MK] = {0};
    float sum_B[TILE_NUM_MK] = {0};
    float sum[TILE_NUM_MK][TILE_NUM_MK] = {0};

    for (int i = 0; i < N; i += TILE_SIZE_N)
    {  
        int calc_shared_idx = load_shared_idx;
        load_shared_idx = 1 - load_shared_idx;
        if (load_idx < N)
        {
            for (int offset = 0; offset < TILE_SIZE_MK; offset += TILE_SIZE)
            {
                // if (load_idx + threadIdx.y < N && idx + offset < K)
                // {
                //     S_B[load_shared_idx][threadIdx.y][threadIdx.x + offset] = B[(load_idx + threadIdx.y) * K + idx + offset];
                // }
                S_B[load_shared_idx][threadIdx.y][threadIdx.x + offset] = (load_idx + threadIdx.y < N && idx + offset < K) ? 
                                                                          (B[(load_idx + threadIdx.y) * K + idx + offset]) : 0;
            }
            for (int offset = 0; offset < TILE_SIZE_MK; offset += TILE_SIZE)
            {
                // if (idy + offset < M && load_idx + threadIdx.x < N)
                // {
                //     S_A[load_shared_idx][threadIdx.y + offset][threadIdx.x] = A[(idy + offset) * N + load_idx + threadIdx.x];
                // }    
                S_A[load_shared_idx][threadIdx.y + offset][threadIdx.x] = (idy + offset < M && load_idx + threadIdx.x < N) ? 
                                                                          (A[(idy + offset) * N + load_idx + threadIdx.x]) : 0;
            }
            load_idx += TILE_SIZE_N;
        }

        // for (int k1 = 0; k1 < TILE_NUM_MK; k1++)
        // {
        //     for (int k2 = 0; k2 < TILE_NUM_MK; k2++)
        //     {
        //         int x_add = k1 * TILE_SIZE;
        //         int y_add = k2 * TILE_SIZE;
        //         if (idx + x_add < K && idy + y_add < M)
        //         {
        //             for (int j = 0; j < TILE_SIZE_N && i + j < N; j++)
        //                 sum[k2][k1] += (S_A[calc_shared_idx][threadIdx.y + y_add][j] * S_B[calc_shared_idx][j][threadIdx.x + x_add]);
        //         }
        //     }
        // }    
        for (int j = 0; j < TILE_SIZE_N && i + j < N; j++)
        {
            for (int k1 = 0; k1 < TILE_NUM_MK; k1++)
            {
                int x_add = k1 * TILE_SIZE;
                sum_B[k1] = S_B[calc_shared_idx][j][threadIdx.x + x_add];
            }
            for (int k2 = 0; k2 < TILE_NUM_MK; k2++)
            {
                int y_add = k2 * TILE_SIZE;
                sum_A[k2] = S_A[calc_shared_idx][threadIdx.y + y_add][j];
            }
            for (int k1 = 0; k1 < TILE_NUM_MK; k1++)
            {
                for (int k2 = 0; k2 < TILE_NUM_MK; k2++)
                {
                    sum[k2][k1] += sum_A[k2] * sum_B[k1];
                }
            }
        }
        __syncthreads();
    }
    for (int k1 = 0; k1 < TILE_NUM_MK; k1++)
    {
        int x_add = k1 * TILE_SIZE;
        for (int k2 = 0; k2 < TILE_NUM_MK; k2++)
        {
            int y_add = k2 * TILE_SIZE;
            if (idx + x_add < K && idy + y_add < M)
            {
                C[(idy + y_add) * K + idx + x_add] = sum[k2][k1];
            }
        }
    }   
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    // dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 blocksPerGrid((K + TILE_SIZE_MK - 1) / TILE_SIZE_MK,
                       (M + TILE_SIZE_MK - 1) / TILE_SIZE_MK);
    // int shared_mem_size = (2 * 16 * 16) * sizeof(float);
    // matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(A, B, C, M, N, K);
     matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```