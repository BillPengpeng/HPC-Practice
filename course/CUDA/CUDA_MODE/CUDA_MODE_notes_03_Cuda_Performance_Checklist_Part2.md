本文主要整理CUDA MODE lecture_008 Cuda Performance Checklist的要点。

## 8. Ok last one: Matmul

![Matmul](https://pic1.zhimg.com/v2-93fb3341b426f1c0721f7b0714419c74_1440w.jpg)

### 矩阵乘法（Matmul）核心要点总结  

#### 1. 矩阵维度与运算逻辑  
- 输入矩阵维度：矩阵 $ A $ 为 $ [M, N] $（$ M $ 行 $ N $ 列），矩阵 $ B $ 为 $ [N, K] $（$ N $ 行 $ K $ 列）。  
- 输出矩阵维度：乘积矩阵 $ C = A \times B $ 为 $ [M, K] $（$ M $ 行 $ K $ 列）。  
- 元素级运算：$ C $ 的**每个元素**由 $ A $ 的**一行**与 $ B $ 的**一列**做**点积**得到。单个点积需 $ N $ 次乘法 + $ N $ 次加法，即单个元素对应 $ 2N $ 次浮点运算（FLOPs）。  


#### 2. 浮点运算次数（FLOPS）  
$ C $ 共有 $ M \times K $ 个元素，每个元素对应 $ 2N $ 次浮点运算，因此**总 FLOPS** 为：  
$$ \text{FLOPS} = M \times K \times 2N $$  


#### 3. 内存访问量（字节）  
矩阵运算的内存开销分为“加载输入”和“写入输出”：  
- 加载 $ A $：需 $ M \times N $ 字节；  
- 加载 $ B $：需 $ N \times K $ 字节；  
- 写入 $ C $：需 $ M \times K $ 字节；  
因此，**总内存访问量**为：  
$$ \text{Bytes} = MN + NK + MK $$  


#### 4. 算术强度（Arithmetic Intensity, AI）  
算术强度定义为“浮点运算次数 / 内存访问量”，用于衡量计算对内存带宽的利用效率，公式为：  
$$ \text{AI} = \frac{2MNK}{MN + NK + MK} $$  


#### 5. 性能瓶颈判定  
- 当矩阵规模**较大**时，计算耗时（浮点运算）主导整体性能，称为 **计算受限（Compute Bound）**；  
- 当矩阵规模**较小**时，内存读写耗时主导整体性能，称为 **带宽受限（Bandwidth Bound）**。  

## 9. TL;DR

这段内容是对两类内核性能优化方法的**要点速览（TL;DR）**，核心信息如下：  

- 针对 **带宽受限型内核（Bandwidth Bound Kernels）**：  
  优化手段为「融合（Fuse）、量化（quantize）、编译（compile）」；  

- 针对 **计算受限型内核（Compute Bound Kernels）**：  
  优化关键是「设计/编写更优的算法（Write a better algorithm）」。  

## 10. Tiling of reused data

![Tiling](https://pic1.zhimg.com/v2-2f51c1c3c774444e546e47f88c5212b2_1440w.jpg)

## 11. Minimize control divergence

```c
__global__ void processArrayWithDivergence(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (data[idx] % 2 == 0) {
            data[idx] = data[idx] * 2;
        } else {
            data[idx] = data[idx] + 1;
        }
    }
}

__global__ void processArrayWithoutDivergence(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int isEven = !(data[idx] % 2);
        data[idx] = isEven * (data[idx] * 2) + (!isEven) * (data[idx] + 1);
    }
}
```

![Minimize control divergence](https://pica.zhimg.com/v2-4144d7df3900661d7cf326c953c0e134_1440w.jpg)

## 12. Thread Coarsening

```c
// Original vector addition kernel without coarsening
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Vector addition kernel with thread coarsening
// Assuming a coarsening factor of 2
__global__ void VecAddCoarsened(float* A, float* B, float* C)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // Coarsening factor applied here
    if (i < N)
        C[i] = A[i] + B[i];
    if (i + 1 < N) // Handle the additional element due to coarsening
        C[i + 1] = A[i + 1] + B[i + 1];
}
```

![Thread Coarsening](https://pic4.zhimg.com/v2-f51646b7453ab2b89207c7bc6594b2af_1440w.jpg)

## 13. Privatization

```c
// Kernel without privatization: Direct global memory access
__global__ void windowSumDirect(const float *input, float *output, int n, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfWindow = windowSize / 2;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = -halfWindow; i <= halfWindow; ++i) {
            int accessIdx = idx + i;
            if (accessIdx >= 0 && accessIdx < n) {
                sum += input[accessIdx];
            }
        }
        output[idx] = sum;
    }
}

// Kernel with privatization: Preload window elements into registers
__global__ void windowSumPrivatized(const float *input, float *output, int n, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfWindow = windowSize / 2;
    __shared__ float sharedData[1024]; // Assuming blockDim.x <= 1024

    // Load input into shared memory (for demonstration, assuming window fits into shared memory)
    if (idx < n) {
        sharedData[threadIdx.x] = input[idx];
        __syncthreads(); // Ensure all loads are complete

        float sum = 0.0f;
        for (int i = -halfWindow; i <= halfWindow; ++i) {
            int accessIdx = threadIdx.x + i;
            // Check bounds within shared memory
            if (accessIdx >= 0 && accessIdx < blockDim.x && (idx + i) < n && (idx + i) >= 0) {
                sum += sharedData[accessIdx];
            }
        }
        output[idx] = sum;
    }
}
```

![Privatization](https://picx.zhimg.com/v2-2c99013a97123e82bbe17e604738a251_1440w.jpg)

## 14. Softmax系列

参考博文[一心二用的Online Softmax](https://zhuanlan.zhihu.com/p/638788074)

![softmax](https://pica.zhimg.com/v2-e3faadd469d50a775c3edc0d06dacd80_1440w.jpg)

![safe softmax](https://picx.zhimg.com/v2-f23f6030b6bac109501f6c61091676d9_1440w.jpg)

![online softmax](https://pica.zhimg.com/v2-7c4693a02c65d266968cb4b91374ced6_1440w.jpg)
