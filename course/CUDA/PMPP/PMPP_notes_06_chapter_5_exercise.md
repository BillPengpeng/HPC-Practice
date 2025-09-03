本文主要整理PMPP Chapter 5 Exercise。

## Q1

1. Consider matrix addition. Can one use shared memory to reduce the
global memory bandwidth consumption? Hint: Analyze the elements that
are accessed by each thread and see whether there is any commonality
between threads.

- Answer:

### **问题核心：矩阵加法的内存访问特性**
给定两个输入矩阵 `A`、`B` 和输出矩阵 `C`，计算：
```math
C[i][j] = A[i][j] + B[i][j]
```
#### **线程访问模式分析**
1. **每个线程独立工作**：
   - 线程负责计算单个输出元素 `C[i][j]`
   - 需读取 **仅两个输入元素**：`A[i][j]` 和 `B[i][j]`
   - **无数据重叠**：不同线程访问完全独立的存储位置

2. **数据复用可能性**：
   - 输入元素 `A[i][j]` 和 `B[i][j]` **仅被当前线程使用一次**
   - 无跨线程数据共享需求（对比矩阵乘法需行列复用）

---

### **共享内存优化可行性评估**
| **优化方案**                | **效果分析**                                                                 | **结论**          |
|------------------------------|-----------------------------------------------------------------------------|------------------|
| **尝试分块加载到共享内存**   | - 线程需先将 `A[i][j]`/`B[i][j]` 从全局内存加载到共享内存<br>- 再从共享内存读取计算<br>**总访问量：2次读共享内存 + 2次读全局内存** | ❌ **劣化性能**（增加额外访问） |
| **线程间数据复用**           | - 每个输入元素仅被单线程使用<br>- **无跨线程复用机会**                                        | ❌ **不可行**      |

---

### **根本矛盾：计算强度（FLOP/B）**
- **原始计算强度**：
  ```math
  \text{FLOP/B} = \frac{1\ \text{次加法}}{8\ \text{字节（读A+B）} + 4\ \text{字节（写C）}} = \frac{1}{12} \approx 0.083
  ```
- **共享内存方案**：
  ```math
  \text{FLOP/B} = \frac{1}{8\text{(全局读)} + 8\text{(共享读)} + 4\text{(全局写)}} = \frac{1}{20} = 0.05
  ```
  → **强度进一步降低**，违反优化原则

---

### **正确优化方向**
1. **合并访问（Coalesced Access）**  
   - 确保线程连续访问内存（如 `threadIdx.x` 对应相邻列）  
   - 单次事务加载128字节（如A100支持32线程×4字节合并访问）  

2. **向量化加载**  
   - 使用 `float4` 类型一次读/写4个元素：  
     ```cpp
     float4 a = reinterpret_cast<float4*>(A)[index];
     float4 b = reinterpret_cast<float4*>(B)[index];
     reinterpret_cast<float4*>(C)[index] = make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
     ```
   - **访问量降为1/4**，计算强度↑至 0.33 FLOP/B  

3. **零拷贝内存（Pinned Memory）**  
   - 主机-设备数据传输优化，减少PCIe瓶颈  

---

### **结论**
- **共享内存无法优化矩阵加法**：因缺乏**数据复用**机会，强行使用会**增加访问次数**。  
- **有效方法**：聚焦**内存访问模式优化**（合并访问/向量化），而非数据复用策略。  

> **关键洞察**：共享内存的价值源于**数据被多次读取**（如矩阵乘中的输入行/列复用）。无复用场景下，其低延迟优势无法抵消额外访问开销。

## Q2

2. Draw the equivalent of Fig. 5.7 for a 8 * 8 matrix multiplication with 2 * 2
tiling and 4 * 4 tiling. Verify that the reduction in global memory bandwidth
is indeed proportional to the dimension size of the tiles.

- Answer:

### **1. 2×2分块优化（TILE_WIDTH=2）**
#### **分块结构**
```plaintext
M矩阵（8×8）                  N矩阵（8×8）                  P矩阵（8×8）
┌───┬───┬───┬───┐            ┌───┬───┬───┬───┐            ┌───┬───┬───┬───┐
│B00│B01│B02│B03│            │B00│B01│B02│B03│            │C00│C01│C02│C03│
├───┼───┼───┼───┤            ├───┼───┼───┼───┤            ├───┼───┼───┼───┤
│B10│B11│B12│B13│            │B10│B11│B12│B13│            │C10│C11│C12│C13│
├───┼───┼───┼───┤            ├───┼───┼───┼───┤            ├───┼───┼───┼───┤
│B20│B21│B22│B23│            │B20│B21│B22│B23│            │C20│C21│C22│C23│
├───┼───┼───┼───┤            ├───┼───┼───┼───┤            ├───┼───┼───┼───┤
│B30│B31│B32│B33│            │B30│B31│B32│B33│            │C30│C31│C32│C33│
└───┴───┴───┴───┘            └───┴───┴───┴───┘            └───┴───┴───┴───┘
```
- **分块数量**：4×4=16块（每块2×2元素）
- **计算阶段数**：8/2=4阶段

#### **全局内存访问分析**
| **访问类型**       | 计算公式                     | 值    |
|---------------------|------------------------------|-------|
| **原始访问量**      | 8×8×8 + 8×8×8 = 1024次       | 1024  |
| **分块加载次数**     | 16块×4阶段×(2×2 + 2×2) = 512次 | 512   |
| **访问量降幅**      | 1 - 512/1024 = **50%**       | 50%   |
| **理论降幅比例**    | 1/TILE_WIDTH = 1/2           | 50% ✅ |

> **关键验证**：实际降幅（50%）完全等于理论降幅比例（1/2）

---

### **2. 4×4分块优化（TILE_WIDTH=4）**
#### **分块结构**
```plaintext
M矩阵（8×8）                  N矩阵（8×8）                  P矩阵（8×8）
┌───────┬───────┐            ┌───────┬───────┐            ┌───────┬───────┐
│ Block00 │ Block01 │            │ Block00 │ Block01 │            │ Block00 │ Block01 │
├───────┼───────┤            ├───────┼───────┤            ├───────┼───────┤
│ Block10 │ Block11 │            │ Block10 │ Block11 │            │ Block10 │ Block11 │
└───────┴───────┘            └───────┴───────┘            └───────┴───────┘
```
- **分块数量**：2×2=4块（每块4×4元素）
- **计算阶段数**：8/4=2阶段

#### **全局内存访问分析**
| **访问类型**       | 计算公式                     | 值    |
|---------------------|------------------------------|-------|
| **原始访问量**      | 8×8×8 + 8×8×8 = 1024次       | 1024  |
| **分块加载次数**     | 4块×2阶段×(4×4 + 4×4) = 256次 | 256   |
| **访问量降幅**      | 1 - 256/1024 = **75%**       | 75%   |
| **理论降幅比例**    | 1/TILE_WIDTH = 1/4           | 75% ✅ |

> **关键验证**：实际降幅（75%）完全等于理论降幅比例（1/4）

---

### **降幅比例通用公式**
对于宽度为 **W** 的矩阵，采用 **T×T分块**：
```math
\text{全局内存访问降幅} = 1 - \frac{1}{T}
```
- **分子**：原始访问量 = $W^3$（M访问） + $W^3$（N访问） = $2W^3$
- **分母**：优化后访问量 = $\left(\frac{W}{T}\right)^2 \times \frac{W}{T} \times (T^2 + T^2) = \frac{2W^3}{T}$
- **比例**：$\frac{\frac{2W^3}{T}}{2W^3} = \frac{1}{T}$

---

### **结论验证**
1. **降幅与分块尺寸正相关**  
   - T=2 → 降幅50%  
   - T=4 → 降幅75%  
   - T=8 → 降幅87.5%（理论值）  

2. **根本原理**  
   分块尺寸 **T** 决定了 **数据复用次数**：  
   - 每个全局内存元素加载后，被 **T个线程** 复用  
   - 复用因子↑ → 全局访问量↓  

3. **硬件意义**  
   - T越大 → 共享内存需求↑（$2T^2$ 元素）  
   - 需平衡：T过大可能超出共享内存容量 → 降低占用率（Occupancy）

> **最终结论**：分块尺寸 **T** 与全局内存访问降幅成 **反比关系**，验证通过 ✅

## Q3

3. What type of incorrect execution behavior can happen if one forgot to use
one or both __syncthreads() in the kernel of Fig. 5.9?

- Answer:

在 CUDA 矩阵乘法内核（图 5.9）中省略 `__syncthreads()` 会导致**数据竞争和计算错误**，具体表现为以下两类严重问题：

---

### **1. 缺失第一个 `__syncthreads()`（加载后同步）**
**问题本质**：**写后读依赖未解决**（Read-After-Write Hazard）  
**错误场景**：  
- 线程 A 尚未完成共享内存写入（如 `Mds[ty][tx] = M[...]`）  
- 线程 B 已开始计算（执行 `Pvalue += Mds[ty][k] * ...`）  
**后果**：  
- **读取陈旧数据**：线程 B 使用共享内存中的**旧值或未初始化值**计算  
- **计算结果随机错误**：部分点积项基于错误输入 → 最终 P 元素值不可预测  
**示例**：  
- 若线程 B 读取时 `Mds` 尚未更新 → 使用前一分块残留数据 → 数值偏差  

---

### **2. 缺失第二个 `__syncthreads()`（计算后同步）**
**问题本质**：**读后写依赖未解决**（Write-After-Read Hazard）  
**错误场景**：  
- 线程 A 仍在计算当前分块的点积（访问 `Nds`）  
- 线程 B 已加载下一分块数据到共享内存（覆盖 `Nds`）  
**后果**：  
- **共享内存数据污染**：线程 A 在计算中途遭遇共享内存被覆盖  
- **部分计算项失效**：线程 A 的后半段计算使用**新分块数据** + **旧分块部分数据** → 结果混乱  
**示例**：  
$$  
\text{预期：} P = (M_1 \times N_1) + (M_2 \times N_2) \\  
\text{实际：} P = (M_1 \times N_1) + (M_2 \times \textcolor{red}{N_1}) \quad (\text{因} N_2 \text{覆盖了} N_1)  
$$

---

### **3. 同时缺失两个同步**
**综合后果**：  
- **数据完全失控**：共享内存读写顺序无保障  
- **错误叠加放大**：陈旧数据 + 提前覆盖 → 计算结果彻底失效  
- **非确定性错误**：每次运行结果不同（依赖线程执行速度随机性）  

---

### **根本原因：GPU 线程乱序执行**
- **硬件行为**：SM 以 warp 为单位调度线程 → **同一 block 内线程执行进度不一致**  
- **同步必要性**：`__syncthreads()` 强制所有线程到达同步点 → 保障共享内存操作时序  

---

### **调试特征**
| **错误类型**       | 典型现象                              | 调试难度 |
|--------------------|---------------------------------------|----------|
| 缺失加载同步        | 部分结果错误（如对角线元素正确，其余随机） | ★★★☆☆    |
| 缺失计算后同步      | 分块边界处数值异常                    | ★★★★☆    |
| 两者均缺失          | 完全随机结果 + 每次运行不一致         | ★★★★★    |

> **关键验证方法**：对比单线程 CPU 计算结果，定位偏差分块位置。

---

### **正确同步的意义**
```c
// 阶段循环内
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

// 1. 协作加载
Mds[ty][tx] = M[...];  // 写共享内存
Nds[ty][tx] = N[...];
__syncthreads();       // ✅ 确保所有加载完成

// 2. 计算
for (int k=0; k<TILE_WIDTH; k++)
    Pvalue += Mds[ty][k] * Nds[k][tx];
__syncthreads();       // ✅ 确保所有计算完成再加载下一分块
```
- **第一同步** → 建立 **生产者-消费者** 顺序（加载完才能用）  
- **第二同步** → 维护 **数据生命周期**（用完才能覆盖）  

**结论**：省略任一 `__syncthreads()` 将破坏共享内存的**数据一致性**，导致计算结果不可信。

## Q4

4. Assuming that capacity is not an issue for registers or shared memory, give
one important reason why it would be valuable to use shared memory
instead of registers to hold values fetched from global memory? Explain
your answer.

- Answer:

在寄存器容量充足的前提下，使用**共享内存（Shared Memory）**而非寄存器保存全局内存数据的**核心价值**在于实现**线程间数据共享**。以下是关键原因解析：

---

### **核心原因：跨线程数据复用**
#### **场景示例（矩阵乘法）**
- **需求**：线程块中所有线程需重复访问**同一全局内存数据**（如M矩阵的一行）。
- **寄存器方案**：
  - 每个线程独立加载所需数据到**私有寄存器** → 同一数据被加载**多次**。
  - 全局内存访问量 = 线程数 × 数据量（如16线程 × 1行 = **16次访问**）。
- **共享内存方案**：
  - 一个线程加载数据到**共享内存** → 所有线程**复用同一份数据**。
  - 全局内存访问量 = **1次访问**（降幅达 **94%**）。

#### **关键差异**
| **特性**         | 寄存器方案                     | 共享内存方案                  |
|-------------------|-------------------------------|-----------------------------|
| **数据可见性**    | 线程私有（其他线程不可见）     | 块内共享（所有线程可见）       |
| **协作成本**      | 无法共享 → 冗余加载            | 一次加载，全员复用            |
| **全局内存访问**  | 高（O(线程数)）                | 低（O(1)）                   |

---

### **本质矛盾：寄存器无法解决数据共享**
- **寄存器设计本质**：  
  → **线程私有**存储（每个线程拥有独立副本）  
  → 物理上**无法被其他线程访问**  
- **共享内存设计本质**：  
  → **线程块级共享**存储（同一块内线程访问相同物理位置）  
  → 通过硬件支持**低延迟块内通信**  

---

### **性能影响量化（以A100 GPU为例）**
| **方案**         | 全局内存访问量（加载一行M） | 带宽消耗（GB/s） | 计算强度（FLOP/B） |
|------------------|----------------------------|-----------------|-------------------|
| **寄存器方案**   | 16次 × 4B = 64 B           | 消耗1555 GB/s   | 0.25 → 未提升      |
| **共享内存方案** | 1次 × 4B = 4 B             | 节省97.5%带宽   | 0.25 → 4.0 (↑16×) |

> **关键结论**：共享内存通过**消除冗余加载**，将计算强度提升 **TILE_WIDTH倍**（分块尺寸），突破内存墙。

---

### **附加优势**
1. **同步控制**：  
   `__syncthreads()` 确保数据就绪性 → 避免**竞态条件**（寄存器无法实现协作同步）。
2. **动态数据交换**：  
   支持线程间**实时交换中间结果**（如归约运算），寄存器仅支持静态私有数据。

---

### **总结**
当算法存在**跨线程数据复用**时（如矩阵乘、卷积、排序），共享内存通过**一次加载多次复用**的协作模式，从根本上减少全局内存访问。寄存器因**物理隔离性**无法实现此机制，故即使容量充足，共享内存仍是优化全局内存带宽的**不可替代方案**。

## Q5

5. For our tiled matrix-matrix multiplication kernel, if we use a 32 3 32 tile,
what is the reduction of memory bandwidth usage for input matrices M
and N?

- Answer:

### **全局内存带宽减少原理**
#### **1. 数据复用率提升**
- **原始分块（16×16）**：  
  每个输入元素（M/N）被加载到共享内存后，被 **16个线程** 复用  
- **新分块（32×32）**：  
  每个输入元素被 **32个线程** 复用 → **复用率翻倍**

#### **2. 访问量计算公式**
全局内存访问总量降幅比例：  
$$\text{降幅比例} = 1 - \frac{1}{\text{TILE\_WIDTH}}$$  

| **分块尺寸** | 降幅比例       | 剩余访问比例 |
|--------------|---------------|------------|
| 16×16        | \(1 - \frac{1}{16} = 93.75\%\) | 6.25%      |
| **32×32**    | \(1 - \frac{1}{32} = 96.875\%\) | **3.125%** |

---

### **量化带宽减少**
假设矩阵宽度为 **W**：
| **指标**               | 原始版本（无分块） | 16×16分块   | **32×32分块** |
|------------------------|-------------------|------------|--------------|
| **M矩阵访问次数**      | \(W^2 \times W\)  | \(W^3/16\) | \(W^3/32\)   |
| **N矩阵访问次数**      | \(W^2 \times W\)  | \(W^3/16\) | \(W^3/32\)   |
| **总访问量**           | \(2W^3\)          | \(2W^3/16\)| \(2W^3/32\)  |
| **带宽使用比例**       | 100%              | 6.25%      | **3.125%**   |
| **较16×16的改进**      | -                 | -          | **↓50%**     |

## Q6

6. Assume that a CUDA kernel is launched with 1000 thread blocks, each of
which has 512 threads. If a variable is declared as a local variable in the
kernel, how many versions of the variable will be created through the
lifetime of the execution of the kernel?

- Answer:

在CUDA编程模型中，当一个变量在内核（kernel）中被声明为**局部变量（local variable）**时，该变量的存储和生命周期与每个线程（thread）绑定。

### **问题关键点**
- **局部变量的特性**：
  - 局部变量在CUDA内核中声明（例如：`int localVar;`）。
  - 它是**线程私有（thread-private）**的，即每个线程都有自己独立的副本。
  - 变量的生命周期仅限于其所属线程的执行期间（线程启动时创建，线程结束时销毁）。
- **执行配置**：
  - 线程块（blocks）数量：1000。
  - 每个线程块的线程（threads）数量：512。
  - 总线程数：\(1000 \times 512 = 512,000\)。

### **原因分析**
1. **线程私有性**：
   - CUDA的并行执行模型要求每个线程拥有独立的执行上下文（包括寄存器、本地内存等）。
   - 局部变量存储在**寄存器（registers）** 或 **本地内存（local memory）** 中，这些资源是线程私有的。
   - 例如，如果内核代码如下：
     ```cpp
     __global__ void myKernel() {
         int localVar = threadIdx.x; // 每个线程创建自己的localVar
         // ... 其他操作 ...
     }
     ```
     每个线程（共512,000个）都会初始化并维护自己的`localVar`副本，互不影响。

2. **执行过程**：
   - 当内核启动时，所有512,000个线程并行执行。
   - 每个线程在启动时创建其局部变量副本，并在执行结束时销毁。
   - 即使线程块被调度到不同的流多处理器（SMs）上执行，每个线程的局部变量仍是独立的（无共享）。

3. **与共享内存的区别**：
   - 如果变量声明为`__shared__`（共享内存），则整个线程块共享一个副本（此时版本数 = 线程块数 = 1000）。
   - 但问题中明确是**局部变量**（无修饰符），因此不适用共享内存规则。

### **附加说明**
- **资源限制的影响**：
  - 如果寄存器资源不足，局部变量可能溢出到全局内存（称为"register spilling"），但这**不改变版本数量**（仍为每个线程一个副本）。
  - 在您的配置中（1000块 × 512线程），总线程数512,000是合理的（现代GPU如NVIDIA A100支持高达2048线程/SM，总线程数取决于GPU型号）。
- **公式总结**：
  - 局部变量的版本数 = 线程块数 × 每个线程块的线程数 = \(1000 \times 512 = 512,000\).

## Q7

7. In the previous question, if a variable is declared as a shared memory
variable, how many versions of the variable will be created through the
lifetime of the execution of the kernel?

- Answer: 1000

## Q8

8. Consider performing a matrix multiplication of two input matrices with
dimensions N * N. How many times is each element in the input matrices
requested from global memory when:
- a. There is no tiling?
- b. Tiles of size T * T are used?

- Answer:
   - a. N 
   - b. N / T

## Q9

9. A kernel performs 36 floating-point operations and seven 32-bit global
memory accesses per thread. For each of the following device
properties, indicate whether this kernel is compute-bound or memory-
bound.
- a. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second  
- b. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second

- Answer:
   - a. memory-bound 
   - b. compute-bound​​

## Q10

```c
// 01-03：主机端配置与内核启动
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);       // 定义线程块尺寸（方形）
dim3 gridDim(A_width / blockDim.x, A_height / blockDim.y); // 计算网格尺寸
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height); // 启动内核

// 04-12：设备端内核实现
__global__ void BlockTranspose(float* A_elements, int A_width, int A_height) {
    // 07：声明共享内存（存储原始数据块）
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH]; 

    // 08-09：计算全局内存索引（行优先存储）
    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;  // 当前块内x偏移
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width; // 当前块内y偏移

    // 10：从全局内存加载数据 → 按转置坐标存入共享内存
    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx]; 

    // 关键同步：确保所有线程完成数据加载
    __syncthreads();  // 需手动添加！

    // 11：从共享内存读取转置数据 → 写回全局内存
    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y]; 
}
```

10. To manipulate tiles, a new CUDA programmer has written a device kernel
that will transpose each tile in a matrix. The tiles are of size
BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of
matrix A is known to be a multiple of BLOCK_WIDTH. The kernel
invocation and code are shown below. BLOCK_WIDTH is known at
compile time and could be set anywhere from 1 to 20.
- a. Out of the possible range of values for BLOCK_SIZE, for what values
of BLOCK_SIZE will this kernel function execute correctly on the
device?
- b. If the code does not execute correctly for all BLOCK_SIZE values, what
is the root cause of this incorrect execution behavior? Suggest a fix to the
code to make it work for all BLOCK_SIZE values.

- Answer:
   - a. BLOCK_SIZE >= BLOCK_WIDTH
   - b. lack one __syncthreads();

## Q11

```c
__global__ void foo_kernel(float* a, float* b) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x[4];
  __shared__ float y_s;
  __shared__ float b_s[128];
  for (unsigned int j = 0; j < 4; ++j) {
    x[j] = a[j * blockDim.x * gridDim.x + i];
  }
  if (threadIdx.x == 0) {
    y_s = 7.4f;
  }
  b_s[threadIdx.x] = b[i];
  __syncthreads();
  b[i] = 2.5f * x[0] + 3.7f * x[1] + 6.3f * x[2] + 8.5f * x[3]
         + y_s * b_s[threadIdx.x] + b_s[(threadIdx.x + 3) % 128];
}
```

11. Consider the following CUDA kernel and the corresponding host function
that calls it:

a. How many versions of the variable i are there?
b. How many versions of the array x[] are there?
c. How many versions of the variable y_s are there?
d. How many versions of the array b_s[] are there?
e. What is the amount of shared memory used per block (in bytes)?
f. What is the floating-point to global memory access ratio of the kernel (in OP/B)?

- Answer:
   - a. 8 * 128 = 1024
   - b. 8 * 128 = 1024
   - c. 8
   - d. 8
   - e. 32 + 128 * 32 = 4128
   - f. FLOP=10 (5乘+5加), 内存访问=24B (6次×4B) → 10/24 ≈ 0.4167

# Q12

12. Consider a GPU with the following hardware limits: 2048 threads/SM, 32
blocks/SM, 64K (65,536) registers/SM, and 96 KB of shared memory/SM.
For each of the following kernel characteristics, specify whether the kernel
can achieve full occupancy. If not, specify the limiting factor.
a. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared
memory/SM.
b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of
shared memory/SM.

- Answer:
   - a. block limit: min(2048 / 64, 32) * 64 = 2048
        registers limit: 65536 / 27 = 2427
        shared memory limit: 96 / 4 * 64 = 1536
        occupancy: 1536 / 2048 = 75%

   - b. block limit: min(2048 / 256, 32) * 64 = 2048
        registers limit: 65536 / 31 = 2114
        shared memory limit: 96 / 8 * 256 = 3072
        full occupancy
