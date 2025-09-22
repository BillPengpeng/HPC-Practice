本文记录Vector Addition的优化过程。

Implement a program that performs element-wise addition of two vectors containing 32-bit floating point numbers on a GPU. The program should take two input vectors of equal length and produce a single output vector containing their sum. 

## 1. Basic

```c
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

## 2. threadsPerBlock

**问题**：固定线程块大小为256可能不是所有GPU架构的最优选择。不同GPU的SM（流式多处理器）对线程块大小的支持不同（如部分GPU在512或1024线程/块时占用率更高）。  
**优化方法**：根据目标GPU架构调整线程块大小。例如：  
- 对于计算能力（Compute Capability）≥7.0的GPU（如Volta、Ampere），可尝试512或1024线程/块；  
- 使用CUDA Occupancy Calculator工具计算不同块大小下的SM占用率（建议目标占用率≥75%）。  

| threadsPerBlock | time(ms) | Device |
| :--: | :--: | :--: |
| 256 | 1.57692 | T4 | 
| 512 | **1.57281** | T4 |
| 1024 | 1.57914 | T4 |
| 256 | 0.24197 | A100-80G | 
| 512 | **0.24137** | A100-80G |
| 1024 | 0.25351 | A100-80G |

## 3. Thread Coarsening

```c
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
    if (idx + 1 < N)
    {
        C[idx + 1] = A[idx + 1] + B[idx + 1];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + 2*threadsPerBlock - 1) / (2*threadsPerBlock);

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

| threadsPerBlock | Thread Coarsening Number | time(ms) | Device | 
| :--: | :--: | :--: | :--: |
| 256 | 2 | 1.91419 | T4 | 
| 128 | 2 | 1.91706 | T4 |
| 256 | 2 | **0.2378**  | A100-80G | 
| 128 | 2 | 0.239   | A100-80G | 
| 512 | 2 | 0.24563 | A100-80G | 
| 256 | 4 | 0.27715 | A100-80G |

## 4. cudaMemcpyAsync异步传输

**问题**：若主机需要将数据从CPU传输到GPU（或反向），默认流（`cudaStreamDefault`）会顺序执行传输和计算，无法利用GPU的并行能力。  
**优化方法**：使用非默认CUDA流并行执行内存传输与核函数计算。例如：  

```c
// cudaMemcpy
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *d_A;
    float *d_B;
    float *d_C;
    int size = N * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// cudaMemcpyAsync
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *d_A;
    float *d_B;
    float *d_C;
    int size = N * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpyAsync(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_B, B, size, cudaMemcpyHostToDevice);
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaMemcpyAsync(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// cudaMemcpyAsync + cudaStreamSynchronize
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *d_A;
    float *d_B;
    float *d_C;
    int size = N * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpyAsync(d_A, A, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, B, size, cudaMemcpyHostToDevice, stream);
    vector_add<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
    
    cudaMemcpyAsync(C, d_C, size, cudaMemcpyDeviceToHost, stream);
    // cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream);
}
```

| threadsPerBlock | Memcpy | time(ms) | Device | 
| :--: | :--: | :--: | :--: |
| 256 | cudaMemcpy | 3.69951  | A100-80G | 
| 256 | cudaMemcpyAsync | 3.72505  | A100-80G | 
| 256 | cudaMemcpyAsync + cudaStreamSynchronize | 3.69809  | A100-80G | 

## 5. f32x4向量化

- 问题​​：当前核函数每个线程仅访问1个float（4字节），全局内存访问次数较多，未能充分利用GPU的宽内存总线（如PCIe/CUDA内存的128位或更宽带宽）。
​​- 优化方法​​：使用向量化数据类型（如float4）让每个线程一次性加载/存储4个连续的float元素，减少全局内存访问次数，提升内存带宽利用率。

```
// float4
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 reg_a = ((float4*)(&A[idx]))[0];
        float4 reg_b = ((float4*)(&B[idx]))[0];
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        ((float4*)(&C[idx]))[0] = reg_c;
    }
    else 
    {
        if (idx < N) C[idx] = A[idx] + B[idx];
        if (idx + 1 < N) C[idx + 1] = A[idx + 1] + B[idx + 1];
        if (idx + 2 < N) C[idx + 2] = A[idx + 2] + B[idx + 2];
    }
}

// float4 + float2
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 reg_a = ((float4*)(&A[idx]))[0];
        float4 reg_b = ((float4*)(&B[idx]))[0];
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        ((float4*)(&C[idx]))[0] = reg_c;
    }
    else if (idx + 2 < N) {
        // C[idx] = A[idx] + B[idx];
        // C[idx + 1] = A[idx + 1] + B[idx + 1];
        float2 reg_a = ((float2*)(&A[idx]))[0];
        float2 reg_b = ((float2*)(&B[idx]))[0];
        float2 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        ((float2*)(&C[idx]))[0] = reg_c;
        C[idx + 2] = A[idx + 2] + B[idx + 2];
    }
    else if (idx + 1 < N) {
        // C[idx] = A[idx] + B[idx];
        // C[idx + 1] = A[idx + 1] + B[idx + 1];
        float2 reg_a = ((float2*)(&A[idx]))[0];
        float2 reg_b = ((float2*)(&B[idx]))[0];
        float2 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        ((float2*)(&C[idx]))[0] = reg_c;
    } 
    else if (idx < N) 
    {
        C[idx] = A[idx] + B[idx];
    }
}

// 传参float4
__global__ void vector_add(const float4* A, const float4* B, float* C, int N) {
    int idx_float_4 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = idx_float_4 * 4;
    if (idx < N) {
        float4 cur_A = A[idx_float_4];
        float4 cur_B = B[idx_float_4];
        float4 reg_c;
        reg_c.x = cur_A.x + cur_B.x;
        reg_c.y = cur_A.y + cur_B.y;
        reg_c.z = cur_A.z + cur_B.z;
        reg_c.w = cur_A.w + cur_B.w;

        if (idx + 3 < N) ((float4*)(&C[idx]))[0] = reg_c;
        else {
            if (idx + 2 < N) C[idx + 2] = reg_c.z;
            if (idx + 1 < N) C[idx + 1] = reg_c.y;
            if (idx + 0 < N) C[idx + 0] = reg_c.x;
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + 4*threadsPerBlock - 1) / (4*threadsPerBlock);
    // float4 *A4 = (float4*)(&A[0]);
    // float4 *B4 = (float4*)(&B[0]);
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A4, B4, C, N);
    cudaDeviceSynchronize();
}
```

| threadsPerBlock | type | time(ms) | Device | 
| :--: | :--: | :--: | :--: |
| 256 | float4 | 0.23521  | A100-80G | 
| 256 | float4 + float2 | 0.23346  | A100-80G | 
| 256 | 传参float4 | 0.23661  | A100-80G | 