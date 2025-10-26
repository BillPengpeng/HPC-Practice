本文主要整理CUDA MODE lecture_009 Reduction的要点。

## 1.0 浮点数求和精度问题

```python
numbers = [1e-20] * 10 + [1e20, -1e20]  # 10个极小值 + 大正数 + 大负数

# 从左到右求和
sum_left_to_right_adjusted = sum(numbers)  # 结果为0.0

# 从右到左求和
sum_right_to_left_adjusted = sum(reversed(numbers))  # 结果约为9.999999999999997e-20

print(sum_left_to_right_adjusted, sum_right_to_left_adjusted)
```

### 问题分析：
1. **浮点数精度限制**：计算机使用二进制浮点数表示实数，存在精度限制
2. **大数吃小数现象**：
   - 从左到右求和：前10个1e-20相加≈1e-19，加上1e20后变为1e20（1e-19被舍入），再加-1e20结果为0
   - 从右到左求和：1e20 + (-1e20) = 0，再加10个1e-20≈1e-19
3. **结合律失效**：浮点数加法不满足结合律，求和顺序影响结果

## 1.1 浮点数范围对精度的影响​​

```python
import torch
large_value = torch.tensor([1000.0], dtype=torch.float32)  # Using float32 for initial value

# Define a smaller value that is significant for float32 but not for float16
small_value = torch.tensor([1e-3], dtype=torch.float32)  # Small value in float32

# Add small value to large value in float32
result_float32 = large_value + small_value

# Convert large value to float16 and add the small value (also converted to float16)
result_float16 = large_value.to(torch.float16) + small_value.to(torch.float16)

# Convert results back to float32 for accurate comparison
result_float32 = result_float32.item()
result_float16_converted = result_float16.to(torch.float32).item()

# Print results
# 1000.0009765625 1000.0
print(result_float32, result_float16_converted)
```

### 浮点数的基础格式
所有浮点数（包括`float32`/`float16`）都遵循**IEEE 754标准**，格式为：  
`符号位（1位） + 指数位（E位） + 尾数位（M位）`  

- **符号位**：表示正负（0正1负）。  
- **指数位**：决定数值的**范围**（能表示多大/多小的数），通过偏移量（Bias）计算实际指数（`e = E - Bias`）。  
- **尾数位**：决定数值的**精度**（能表示多少位有效数字），隐含最高位的`1.`（归一化数），实际尾数为`1.b₁b₂...b_M`。  


### 例子中的数值拆解
我们先明确例子中关键数值的浮点表示：
1. **`large_value = 1000.0`**  
   - `float32`：指数`e = log₂(1000) ≈ 9.96`→ 存储指数`E = 9 + 127 = 136`（Bias=127），尾数`1.1111010000`（23位，隐含`1.`）。  
   - `float16`：指数`e = 9.96`→ 存储指数`E = 9 + 15 = 24`（Bias=15），尾数`1.1111010000`（10位，隐含`1.`）。  
   结论：`1000.0`可被`float16`**精确表示**。

2. **`small_value = 1e-3（0.001）`**  
   - `float32`：指数`e = log₂(0.001) ≈ -9.96`→ 存储指数`E = -9 + 127 = 118`，尾数`1.024`（23位，隐含`1.`）。  
   - `float16`：指数`e = -9.96`→ 存储指数`E = -10 + 15 = 5`，尾数`1.0`（10位，隐含`1.`）。  
   结论：`1e-3`在`float16`中是**近似值**，但核心是它的指数为`-10`。


### 三、加法运算的核心规则：指数对齐
浮点数相加时，**必须将小数（指数更小的数）的尾数左移，使其指数与大数一致**。公式为：  
`a + b = 2^e_a × (m_a + m_b × 2^(e_b - e_a))`  
其中：
- `e_a`：大数的指数（`1000.0`的`e=9`）；  
- `m_a`：大数的尾数（`1000.0`的`m=1.1111010000`）；  
- `e_b`：小数的指数（`1e-3`的`e=-10`）；  
- `m_b`：小数的尾数（`1e-3`的`m=1.0`）。  


### 四、精度损失的根本原因：尾数位数不足
当小数的指数与大数的指数差**超过尾数位数**时，小数的尾数左移后会被**移出尾数范围**，导致其贡献被舍入为零。我们对比`float32`和`float16`的情况：


#### 1. `float32`加法：精度保留
- 指数差：`e_a - e_b = 9 - (-10) = 19`；  
- 尾数位数：`float32`有23位尾数；  
- 小数对齐后的尾数：`m_b × 2^(e_b - e_a) = 1.0 × 2^(-19) ≈ 1.9×10^-6`；  
- 结果：`m_a + 对齐后的尾数 = 1.1111010000 + 1.9×10^-6 ≈ 1.1111029`，仍有有效增量。  
最终结论：`float32`相加结果为`1000.0009765625`（可感知小数的影响）。


#### 2. `float16`加法：精度丢失
- 指数差：同样是`19`；  
- 尾数位数：`float16`仅10位尾数；  
- 小数对齐后的尾数：`1.9×10^-6`，远小于`float16`尾数的最小有效位（`2^-10 ≈ 9.77×10^-4`）；  
- 结果：`m_a + 对齐后的尾数 ≈ m_a`（小数的贡献被完全舍入）。  
最终结论：`float16`相加结果仍为`1000.0`（小数的影响消失）。


### 五、浮点数范围对精度的本质影响
浮点数的**范围**（由指数位数决定）和**精度**（由尾数位数决定）是一对矛盾：
- 数值越大（指数越高），能表示的小数部分的精度**越低**——因为尾数位数固定，需要更多位表示指数，剩余位表示小数部分的空间更少。  
- 当大数和小数相加时，若小数的**指数与大数的指数差超过尾数位数**，小数的贡献会被舍入为零，导致**大数“吃掉”小数**，精度丢失。


### 六、案例结论
例子中：
- `float16`的范围足够大（能表示`1000.0`），但**尾数位数太少**（10位），导致`1e-3`的贡献被完全舍入，结果不变；  
- `float32`的尾数位数足够多（23位），能保留`1e-3`的部分贡献，结果更精确。


### 总结建议
- 避免用**半精度（float16）**处理**混合数量级**的数据（大数+小数），优先用`float32`或更高精度；  
- 若必须用低精度，可通过**归一化数据**（将数值缩放至相近范围）或**分块求和**（减少指数差）降低精度损失；  
- 关键计算（如梯度累加）建议用`float32`甚至`float64`，避免低精度导致的数值不稳定。


**一句话总结**：浮点数的范围越大（指数越高），能表示的小数精度越低；当大数和小数的指数差超过尾数位数时，小数的贡献会被完全舍入，导致精度丢失。

## 2.0 simple_reduce

```c
__global__ void SimpleSumReductionKernel(float* input, float* output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();

    }
    if (threadIdx.x == 0) {
    *output = input[0];
    }
}
```

```ncu
  SimpleSumReductionKernel(float *, float *) (1, 1, 1)x(1024, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.81
    SM Frequency                    Ghz         1.55
    Elapsed Cycles                cycle         9318
    Memory Throughput                 %         1.59
    DRAM Throughput                   %         1.59
    Duration                         us         6.02
    L1/TEX Cache Throughput           %        19.41
    L2 Cache Throughput               %         1.57
    SM Active Cycles              cycle       381.20
    Compute (SM) Throughput           %         1.50
    ----------------------- ----------- ------------

    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Gbyte/s         3.94
    Mem Busy                               %         1.57
    Max Bandwidth                          %         1.83
    L1/TEX Hit Rate                        %        91.63
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Compression Input Sectors      sector            0
    L2 Hit Rate                            %        77.56
    Mem Pipes Busy                         %         0.75
    ---------------------------- ----------- ------------

    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.12
    Branch Instructions              inst         1312
    Branch Efficiency                   %        74.05
    Avg. Divergent Branches                       2.39
    ------------------------- ----------- ------------
```

### 1. Branch Instructions Ratio（分支指令比例）  
- **定义**：分支指令数占总指令数的百分比（`分支指令数 ÷ 总指令数 × 100%`）。  
- **数值**：0.12%（极低）。  
- **意义**：反映程序中**分支指令的频繁程度**。0.12%意味着程序中99.88%的指令是无分支的顺序/并行指令，分支非常少。


### 2. Branch Instructions（分支指令数）  
- **定义**：程序执行过程中遇到的**分支指令绝对数量**（如`if`、`switch`、循环中的条件跳转）。  
- **数值**：1,312条（很少）。  
- **意义**：结合“分支比例”看，进一步确认程序的分支行为极少——即使总指令数很大（如100万条），分支也仅占千分之一左右。


### 3. Branch Efficiency（分支效率）  
- **定义**：**Warp内线程执行分支时的一致性比例**（或“无分化的Warp占比”）。  
  更通俗的解释：**执行分支时，无需分化（所有线程走同一路径）的Warp占总Warp的比例**。  
- **数值**：74.05%（中等偏上）。  
- **意义**：  
  - 74.05%的Warp在执行分支时，内部线程行为完全一致（无需分化，效率100%）；  
  - 剩余25.95%的Warp存在线程分化（效率下降，如分化为两部分则效率减半）。  
  这个指标直接反映分支对性能的影响——效率越高，性能损失越小。


#### 4. Avg. Divergent Branches（平均发散分支数）  
- **定义**：**平均每个分支点的线程发散程度**（或“每个Warp中导致分化的分支指令数”）。  
  更直观的理解：**每个分支指令平均导致多少线程走不同路径**。  
- **数值**：0.37（极低）。  
- **意义**：  
  0.37意味着**大部分分支点只有极少数线程发散**（比如32线程的Warp中，仅1-2个线程走不同路径），几乎不会造成明显的效率损失。

## 2.1 control_divergence_reduce

```c
__global__ void FixDivergenceKernel(float* input, float* output) {
    unsigned int i = threadIdx.x; //threads start next to each other
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) { // furthest element is blockDim away
        if (threadIdx.x < stride) { // 
            input[i] += input[i + stride]; // each thread adds a distant element to its assigned position
        }
        __syncthreads();

    }
    if (threadIdx.x == 0) {
    *output = input[0];
    }
}
```

```ncu
  FixDivergenceKernel(float *, float *) (1, 1, 1)x(1024, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.64
    SM Frequency                    Ghz         1.53
    Elapsed Cycles                cycle         5185
    Memory Throughput                 %         3.12
    DRAM Throughput                   %         3.12
    Duration                         us         3.39
    L1/TEX Cache Throughput           %        13.70
    L2 Cache Throughput               %         1.45
    SM Active Cycles              cycle       173.75
    Compute (SM) Throughput           %         0.93
    ----------------------- ----------- ------------

    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Gbyte/s         3.92
    Mem Busy                               %         1.49
    Max Bandwidth                          %         1.81
    L1/TEX Hit Rate                        %        66.88
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Compression Input Sectors      sector            0
    L2 Hit Rate                            %        58.17
    Mem Pipes Busy                         %         0.68
    ---------------------------- ----------- ------------

    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.31
    Branch Instructions              inst         1126
    Branch Efficiency                   %        99.32
    Avg. Divergent Branches                       0.06
    ------------------------- ----------- ------------
```

## 2.2 shared_reduce

```c
__global__ void SharedMemoryReduction(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t  + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /=2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}
```

```ncu
  SharedMemoryReduction(float *, float *) (1, 1, 1)x(1024, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.72
    SM Frequency                    Ghz         1.53
    Elapsed Cycles                cycle         5841
    Memory Throughput                 %         1.88
    DRAM Throughput                   %         1.38
    Duration                         us         3.81
    L1/TEX Cache Throughput           %        52.50
    L2 Cache Throughput               %         0.91
    SM Active Cycles              cycle       207.60
    Compute (SM) Throughput           %         1.88
    ----------------------- ----------- ------------

    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Gbyte/s         3.27
    Mem Busy                               %         0.96
    Max Bandwidth                          %         1.87
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Compression Input Sectors      sector            0
    L2 Hit Rate                            %        40.81
    Mem Pipes Busy                         %         1.87
    ---------------------------- ----------- ------------
```

要理解**共享内存归约内核**中`L1/TEX Hit Rate`与`L1/TEX Cache Throughput`的表现，以及它们与`__shared__`的关系，需要从**存储层次架构**、**内核访问模式**和**指标定义**三个维度深入分析：


### **先回顾核心背景：GPU存储层次与共享内存**
GPU的存储层次从快到慢依次是：  
**寄存器 → 共享内存（__shared__） → L1/TEX缓存 → L2缓存 → DRAM显存**  

其中：
- **共享内存（__shared__）**：每个SM内部的**片上SRAM**，容量小（如A100的SM有48KB/112KB共享内存），但**访问延迟极低（~10ns）**，且**不经过L1/TEX缓存**（独立存储层次）；  
- **L1/TEX缓存**：每个SM的**片上缓存**，用于加速**全局内存**（`float*`）、纹理内存的访问，缓存行大小通常为128字节。  


### **指标解读：为什么L1/TEX Hit Rate=0%，Throughput=52.5%？**
结合**共享内存归约**的内核行为（如：线程从全局内存读数据→写入共享内存→共享内存归约），两个指标的表现可拆解为：


#### **L1/TEX Hit Rate=0%：几乎没有全局/纹理内存访问命中缓存**  
- **定义回顾**：`L1/TEX Hit Rate`是**全局/纹理内存访问中命中L1/TEX缓存的比例**。  
- **为何为0%？**  
  共享内存归约的核心是**大量访问共享内存**（而非全局内存）：  
  - 步骤1：每个线程从全局内存读取1个元素→写入共享内存（这部分是全局内存访问，但占比极低）；  
  - 步骤2：后续的归约操作（如`shared_mem[i] += shared_mem[i+stride]`）**全部访问共享内存**，不经过L1/TEX缓存。  
  因此，**几乎没有全局/纹理内存访问命中L1/TEX缓存**，导致Hit Rate=0%。  


#### **L1/TEX Cache Throughput=52.5%：仍有少量全局内存访问贡献吞吐量**  
- **定义回顾**：`L1/TEX Cache Throughput`是**L1/TEX缓存处理的数据总量占最大带宽的比例**。  
- **为何不为0？**  
  内核的**初始加载阶段**需要从全局内存读取数据到共享内存：  
  每个线程从全局内存读取1个元素→存入寄存器→写入共享内存。这部分**全局内存读取请求**会走L1/TEX缓存（即使最终数据要写入共享内存，读取时仍需经过L1缓存）。  
  因此，这部分全局内存访问贡献了`52.5%`的L1/TEX吞吐量——**缓存处理了“从全局内存加载数据到共享内存”的请求**。  


### 与`__shared__`的直接关系：共享内存“分流”了L1/TEX的访问**
`__shared__`的使用是导致两个指标表现的关键原因：  
1. **共享内存“接管”了主要访问**：  
   归约的核心操作（线程间数据交换）全部在共享内存中完成，**不经过L1/TEX缓存**，因此L1/TEX的命中率被“稀释”到0%；  
2. **仅初始加载依赖全局内存**：  
   共享内存的初始数据来自全局内存，这部分访问仍需经过L1/TEX缓存，因此Throughput不为0。  


### 进一步验证：共享内存的“高效性”如何体现？**
虽然L1/TEX指标看似“一般”，但共享内存的使用大幅提升了内核性能：  
- **低延迟**：共享内存访问延迟（~10ns）远低于L1缓存（~40ns）和DRAM（~100ns），归约的核心操作因此极快；  
- **高带宽**：共享内存的带宽是L1缓存的数倍（如A100的共享内存带宽达1.5TB/s，L1缓存约500GB/s），支持大量线程并行访问。  


### **总结：共享内存如何影响L1/TEX指标？**
| 指标                | 表现          | 原因                                                                 |
|---------------------|---------------|----------------------------------------------------------------------|
| L1/TEX Hit Rate     | 0%            | 共享内存归约的核心操作访问共享内存，不经过L1/TEX；全局内存访问占比极低。 |
| L1/TEX Cache Throughput | 52.5% | 初始加载阶段从全局内存读取数据到共享内存的请求，仍需经过L1/TEX缓存。   |

### **关键结论**
- **`__shared__`的作用**：将归约的核心操作从“慢速的全局内存”转移到“快速的共享内存”，避免了L1/TEX的瓶颈；  
- **指标的合理性**：L1/TEX Hit Rate=0%不是问题，反而说明共享内存的使用“分流”了大部分访问，是优化的结果；  
- **性能的核心**：共享内存的低延迟、高带宽让归约操作快速完成，即使L1/TEX指标一般，内核整体性能仍远优于全局内存归约。

**一句话总结**：  
共享内存归约的内核通过`__shared__`将核心操作移出L1/TEX缓存，因此L1/TEX Hit Rate为0%，但初始加载的全局内存访问仍贡献了Throughput——这是**共享内存优化成功的标志**！

## 2.3 segment_reduce

```c
__global__ void SharedMemoryReduction(float* input, float* output, int n) {
    __shared__ float input_s[BLOCK_DIM]; 
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // index within a block
    unsigned int t = threadIdx.x; // global index

    // Load elements into shared memory
    if (idx < n) {
        input_s[t] = input[idx];
    } else {
        input_s[t] = 0.0f;
    }
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride && idx + stride < n) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }

    // Reduction across blocks in global memory
    // needs to be atomic to avoid contention
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}
```

## 2.4 reduce_coarsening

```c
__global__ void CoarsenedReduction(float* input, float* output, int size) {
    __shared__ float input_s[BLOCK_DIM];

    unsigned int i = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = 0.0f;

    // Reduce within a thread
    for (unsigned int tile = 0; tile < COARSE_FACTOR; ++tile) {
        unsigned int index = i + tile * blockDim.x;
        if (index < size) {
            sum += input[index];
        }
    }

    input_s[t] = sum;
    __syncthreads();
    
    //Reduce within a block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }

    //Reduce over blocks
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}
```