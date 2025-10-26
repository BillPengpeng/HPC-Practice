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

```ncu
    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.17
    Branch Instructions              inst        98304
    Branch Efficiency                   %            0
    Avg. Divergent Branches                          0
    ------------------------- ----------- ------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.12
    Branch Instructions              inst        65536
    Branch Efficiency                   %            0
    Avg. Divergent Branches                          0
    ------------------------- ----------- ------------
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

```ncu
  VecAdd(float *, float *, float *) (4, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.72
    SM Frequency                    Ghz         1.52
    Elapsed Cycles                cycle         3353
    Memory Throughput                 %         3.04
    DRAM Throughput                   %         3.04
    Duration                         us         2.21
    L1/TEX Cache Throughput           %         4.26
    L2 Cache Throughput               %         1.91
    SM Active Cycles              cycle       300.25
    Compute (SM) Throughput           %         0.39
    ----------------------- ----------- ------------

  VecAddCoarsened(float *, float *, float *) (2, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.72
    SM Frequency                    Ghz         1.52
    Elapsed Cycles                cycle         3362
    Memory Throughput                 %         2.45
    DRAM Throughput                   %         2.45
    Duration                         us         2.21
    L1/TEX Cache Throughput           %        11.65
    L2 Cache Throughput               %         2.29
    SM Active Cycles              cycle       168.30
    Compute (SM) Throughput           %         0.33
    ----------------------- ----------- ------------
```

![Thread Coarsening](https://pic4.zhimg.com/v2-f51646b7453ab2b89207c7bc6594b2af_1440w.jpg)

### **线程粗化两倍对 GPU 吞吐量指标的影响分析**  
线程粗化（Thread Coarsening）是通过**减少线程数量、增加单个线程的工作量**来优化 GPU 资源利用率的技术。结合用户提供的 `VecAdd`（普通版）和 `VecAddCoarsened`（粗化两倍版）性能数据，以下从 **线程配置、内存访问、计算资源** 三个维度，解释粗化对各项吞吐量的影响：  

### **线程配置与核心目标**  
- **普通版 VecAdd**：线程块配置为 `(4, 1, 1)`，每个线程块 256 线程，总线程数 $4 \times 256 = 1024$。假设每个线程处理 1 个元素（如向量化加法的单个分量），总计算量为 1024 次加法。  
- **粗化版 VecAddCoarsened**：线程块配置为 `(2, 1, 1)`，每个线程块 256 线程，总线程数 $2 \times 256 = 512$。线程粗化两倍意味着**每个线程处理 2 个元素**（总计算量仍为 $512 \times 2 = 1024$ 次加法，与普通版一致）。  

### **各吞吐量指标的变化与原因**  

#### **(1) 执行时间（Duration）与周期数（Elapsed Cycles）**  
- 普通版：Duration=2.21μs，Elapsed Cycles=3353。  
- 粗化版：Duration=2.21μs，Elapsed Cycles=3362（几乎无变化）。  
**原因**：总计算量相同（1024 次加法），且 GPU 的时钟频率（SM Frequency=1.52GHz）未变，因此执行时间由计算量决定，粗化未改变总计算负载，故周期数和时长基本一致。  

#### **(2) 内存吞吐量（Memory/DRAM Throughput）**  
- 普通版：Memory Throughput=3.04%，DRAM Throughput=3.04%。  
- 粗化版：Memory Throughput=2.45%（下降 19%），DRAM Throughput=2.45%（下降 19%）。  
**关键理解**：  
  - 内存吞吐量（Memory Throughput）是内存子系统（L1/L2/DRAM）的总有效带宽占峰值的比值。粗化后，总线程数减半（1024→512），内存控制器需处理的线程请求减少，导致总内存带宽需求下降。  
  - DRAM Throughput（DRAM 实际带宽占比）同步下降，说明 DRAM 未被充分激活，资源利用率降低。  

#### **(3) 缓存吞吐量（L1/TEX & L2 Cache Throughput）**  
- 普通版：L1/TEX=4.26%，L2=1.91%。  
- 粗化版：L1/TEX=11.65%（上升 173%），L2=2.29%（上升 20%）。  
**原因**：  
  - 粗化后，每个线程处理 2 个元素，访问的内存地址更集中（如连续访问 `in[i]` 和 `in[i+1]`），L1 缓存的空间局部性增强，命中率提升，因此 L1/TEX 缓存吞吐量显著上升。  
  - L2 缓存因 L1 命中率提高，需处理的缓存缺失请求减少，但仍需支持部分未命中的访问，故吞吐量小幅上升。  

#### **(4) 计算吞吐量（Compute SM Throughput）**  
- 普通版：Compute Throughput=0.39%（计算单元利用率）。  
- 粗化版：Compute Throughput=0.33%（下降 15%）。  
**矛盾点与解释**：  
  - 粗化后，每个线程处理更多数据（2 个元素 vs 1 个），理论上计算单元应更忙碌。但 Compute Throughput 下降，可能因：  
    - **线程调度开销减少**：线程数减半（1024→512），SM 调度线程的负担降低，SM Active Cycles（SM 活跃周期）从 300.25 降至 168.30（减少 44%），SM 空闲时间减少，但计算单元的“有效工作占比”（占峰值的百分比）因总计算量固定而下降。  
    - **计算密度未提升**：每个线程的计算量仅翻倍（从 1 次加法到 2 次），未达到 SM 计算单元的饱和需求（如 FP32 单元需更多并行计算任务），导致利用率未同步提升。  

### **3. 总结：线程粗化两倍的核心影响**  
线程粗化两倍（减少线程数、增加单线程工作量）对吞吐量的影响可归纳为：  
- **内存访问**：总内存带宽需求下降（Memory/DRAM Throughput 降低），但缓存局部性增强（L1/TEX Throughput 显著提升）。  
- **计算资源**：SM 调度开销减少（SM Active Cycles 降低），但计算单元利用率（Compute Throughput）因总计算量固定而小幅下降。  
- **执行效率**：总执行时间不变（因计算量固定），但资源分配更集中（缓存更高效，内存更闲置）。  

**关键结论**：线程粗化适用于**计算密度低、内存访问分散**的场景（如本例中的简单向量加法）。通过减少线程数、增加单线程工作量，可提升缓存利用率并降低内存控制器负担，但需注意避免因计算量未同步增加导致的计算单元利用率下降。对于计算密集型任务，粗化可能需配合其他优化（如增加计算负载）以充分利用 SM 资源。

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
