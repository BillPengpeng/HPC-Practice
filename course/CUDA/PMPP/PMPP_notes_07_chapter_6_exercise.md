本文主要整理PMPP Chapter 6 Exercise。

## Q1

1. Write a matrix multiplication kernel function that corresponds to the design
illustrated in Fig. 6.4.

```c
#define TILE_SIZE 32

__global__ void matrixMulCornerTurning(float* A, float* B, float* C, int width) {
    // 声明共享内存分块
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 计算当前线程负责的C矩阵元素行索引
    int row = by * TILE_SIZE + ty;
    
    // 计算当前线程负责的C矩阵元素列索引（考虑粗化因子）
    int colStart = bx * TILE_SIZE;
    
    // 局部累加器
    float Cvalue = 0.0f;

    // 分块循环
    for (int ph = 0; ph < width / TILE_SIZE; ++ph) {
        // 加载A的分块（行优先，连续访问）
        As[ty][tx] = A[row * width + (ph * TILE_SIZE + tx)];
        
        // Corner Turning：加载B的分块（列优先存储优化）
        // 关键：交换tx和ty的角色，实现合并访问
        int b_col = colStart + tx;  // 当前线程负责的B矩阵列
        int b_row = ph * TILE_SIZE + ty; // 当前线程负责的B矩阵行
        Bs[tx][ty] = B[b_col * width + b_row];  // 列优先存储的访问方式
        
        __syncthreads();

        // 计算当前分块的贡献
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[tx][k];
        }
        __syncthreads();
    }

    // 将结果写入C矩阵（行优先）
    int col = colStart + tx;
    C[row * width + col] = Cvalue;
}
```

## Q2

2. For tiled matrix multiplication, of the possible range of values for
BLOCK_SIZE, for what values of BLOCK_SIZE will the kernel completely
avoid uncoalesced accesses to global memory? (You need to consider only
square blocks.)

- Answer: 

在分块矩阵乘法（Tiled Matrix Multiplication）中，**完全避免全局内存的非合并访问（Uncoalesced Access）** 的关键在于确保线程块（Block）的维度 `BLOCK_SIZE`（即代码中的 `TILE_WIDTH`）满足以下条件：

---

### **核心条件**
**`BLOCK_SIZE` 必须是 `warp大小（32）的整数倍`**  
即：  
$
\text{BLOCK\_SIZE} = 32 \times k \quad (k = 1, 2, 3, \dots)
$

---

### **原因分析**
#### **1. 合并访问（Coalesced Access）的硬件要求**
- GPU的全局内存访问优化依赖于 **线程束（Warp）内线程访问连续内存地址**。
- 当线程束中所有线程访问的全局内存地址连续时，硬件会将这32次访问合并为**1次DRAM突发传输（Burst）**，最大化带宽利用率。

#### **2. 分块加载时的访问模式**
在分块矩阵乘法中，每个线程块负责加载输入矩阵的一个分块（Tile）到共享内存：
- **加载矩阵 `M` 的分块**（行优先存储）：
  ```c
  Mds[ty][tx] = M[row * width + (ph * TILE_WIDTH + tx)];
  ```
  - `tx`（线程的x索引）连续 → 线程束内 `tx` 从0到31连续递增 → **访问地址连续** → **可合并**。
- **加载矩阵 `N` 的分块**（需考虑存储布局）：
  - 若 `N` 为行优先存储，访问需通过线程粗化（Coarsening）确保连续性（见代码中的 `COARSE_FACTOR` 循环）。

#### **3. `BLOCK_SIZE` 与合并访问的关系**
- **当 `BLOCK_SIZE` 是32的整数倍时**：  
  线程块宽度（`blockDim.x`）是32的整数倍 → **每个Warp的线程在x方向连续** → 加载分块时，Warp内线程访问的全局内存地址连续 → **完全避免非合并访问**。  
  **示例**：  
  - `BLOCK_SIZE = 32`：每个Warp刚好覆盖一行（32个元素），地址连续。  
  - `BLOCK_SIZE = 64`：每行分块由2个Warp覆盖（Warp0访问列0-31，Warp1访问列32-63），每个Warp内部地址仍连续。

- **当 `BLOCK_SIZE` 不是32的整数倍时**：  
  线程块宽度无法被32整除 → 部分Warp的线程可能跨越不连续的内存地址 → **访问不连续** → **无法合并**。  
  **反例**：  
  - `BLOCK_SIZE = 31`：线程块宽度31（非32倍数）→ 第一个Warp覆盖31个线程，第二个Warp只有1个线程（浪费31线程），且地址不连续 → **非合并访问**。

---

### **结论**
为完全避免全局内存的非合并访问，`BLOCK_SIZE` 必须满足：  
$
\boxed{\text{BLOCK\_SIZE} \mod 32 = 0}
$  
即：**`BLOCK_SIZE` 需为32的整数倍**（如32、64、96、128等）。  

#### **附加说明**
- **方形分块（Square Blocks）**：  
  由于问题限定只考虑方形分块（`blockDim.x = blockDim.y`），因此只需确保 `BLOCK_SIZE` 是32的整数倍。
- **实际应用建议**：  
  - 优先选择 `BLOCK_SIZE=32`（最小粒度，最大化并行度）。  
  - 若需更大分块（如减少分块循环次数），可选择 `BLOCK_SIZE=64`（需验证共享内存和寄存器占用率）。

## Q3

3. Consider the following CUDA kernel:

```c
// 优化版本示例
__global__ void optimized_foo_kernel(float* a, float* b, float* c, float* d, float* e) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float a_s[256];
    __shared__ float bc_s[4][256 + 1];  // 添加填充避免Bank冲突
    
    // 预计算减少冗余
    int c_base = i * 4;  // 预计算c的基地址

    a_s[threadIdx.x] = a[i];
    
    for (unsigned int j = 0; j < 4; ++j) {
        // 使用二维共享内存+填充
        bc_s[j][threadIdx.x] = b[j * blockDim.x * gridDim.x + i] + c[c_base + j];
    }
    __syncthreads();

    d[i + 8] = a_s[threadIdx.x];
    // 重构e的写入模式（需结合具体业务逻辑）
    e[i] = bc_s[threadIdx.x % 4][threadIdx.x / 4]; 
}
```

For each of the following memory accesses, specify whether they are
coalesced or uncoalesced or coalescing is not applicable:
- a. The access to array a of line 05
- b. The access to array a_s of line 05
- c. The access to array b of line 07
- d. The access to array c of line 07
- e. The access to array bc_s of line 07
- f. The access to array a_s of line 10
- g. The access to array d of line 10
- h. The access to array bc_s of line 11
- i. The access to array e of line 11

- Answer: 
   - a. coalesced
   - b. coalescing is not applicable 
   - c. coalesced
   - d. uncoalesced
   - e. coalescing is not applicable 
   - f. coalescing is not applicable 
   - g. coalesced
   - h. coalescing is not applicable 
   - i. uncoalesced

- 全局内存访问中，​​连续索引访问可合并​​（a,c,g），​​跨步访问不可合并​​（d,i）
- 共享内存访问​​不适用合并概念​​，但需关注Bank冲突（e,h存在冲突风险）

## Q4

4. What is the floating point to global memory access ratio (in OP/B) of each of
the following matrix-matrix multiplication kernels?
- a. The simple kernel described in Chapter 3, Multidimensional Grids and
Data, without any optimizations applied.
- b. The kernel described in Chapter 5, Memory Architecture and Data
Locality, with shared memory tiling applied using a tile size of 32 * 32.
- c. The kernel described in this chapter with shared memory tiling applied
using a tile size of 32 * 32 and thread coarsening applied using a
coarsening factor of 4.

- Answer: 
   - a. 2K / ((2K + 1) * 4)
   - b. 2K / ((2K / 32 + 1) * 4)
   - c. 8k / ((K / 32 + K / 8 + 4) * 4)

以下是三种矩阵乘法内核的**浮点运算与全局内存访问比率（OP/B）** 分析，基于CUDA优化技术原理推导：

---

### **核心公式**
$$
\text{OP/B} = \frac{\text{浮点运算次数 (FLOPs)}}{\text{全局内存访问字节数 (Bytes)}}
$$
- **单次矩阵乘法计算**：  
  $ P = M \times N $ 中每个输出元素需 $ 2 \times K $ 次浮点运算（乘加各1次）
- **全局内存访问**：以字节为单位（`float` = 4字节）

---

### **a. 基础内核（无优化）**
- **计算逻辑**：  
  每个线程计算1个输出元素 $$ P_{i,j} = \sum_{k=0}^{K-1} M_{i,k} \times N_{k,j} $$
- **浮点运算 (FLOPs)**：  
  $ 2K $ 次（K次乘法 + K次加法）
- **全局内存访问**：  
  - 读取 $ M_{i,k} $：$ K $ 次 × 4字节  
  - 读取 $ N_{k,j} $：$ K $ 次 × 4字节  
  - 写入 $ P_{i,j} $：1次 × 4字节  
  **总计**：$ (2K + 1) \times 4 $ 字节
- **OP/B 比率**：
  $$
  \text{OP/B} = \frac{2K}{(2K + 1) \times 4} \approx \frac{2K}{8K} = \frac{1}{4} \quad (\text{当 } K \gg 1)
  $$
  **结果**：≈ **0.25 OP/B**

---

### **b. 共享内存分块优化（Tile Size = 32×32）**
- **计算逻辑**：  
  线程块加载 $ M $ 和 $ N $的分块到共享内存，复用数据计算输出分块。
- **浮点运算 (FLOPs)**：  
  不变，仍为 $ 2K $ 次/元素
- **全局内存访问优化**：  
  - 输入矩阵 $ M $ 和 $ N $：每个分块被加载一次，供 $ \text{TILE\_WIDTH}^2 $ 个元素复用  
    **访问次数**：$ \frac{K}{\text{TILE\_WIDTH}} \times 2 $ 次/元素  
  - 输出矩阵 $ P $：1次写入/元素  
  **总计字节/元素**：
  $$
  \left( \frac{2K}{32} + 1 \right) \times 4 = \left( \frac{K}{16} + 1 \right) \times 4
  $$
- **OP/B 比率**：
  $$
  \text{OP/B} = \frac{2K}{\left( \frac{K}{16} + 1 \right) \times 4} \approx \frac{2K}{\frac{K}{4}} = 8 \quad (\text{当 } K \gg 16)
  $$
  **结果**：≈ **8 OP/B**（提升32倍）

---

### **c. 分块+线程粗化（Coarsening Factor = 4）**
- **计算逻辑**：  
  单线程处理4个输出元素（水平相邻），复用 $ M $ 的分块。
- **浮点运算 (FLOPs)**：$ 4 \times 2K = 8K $ 次/线程
- **全局内存访问优化**：  
  - $ M $：加载一次分块供4个元素复用 → 访问次数：$K / 32$ 次/线程  
  - $ N $：每个输出元素需独立加载 $ N $ 的分块 → 访问次数：$K / 8次/线程  
  - 写入 $ P $：4次/线程  
  **总计字节/线程**：
  $$
  \left( \frac{K}{32} + \frac{K}{8} + 4 \right) \times 4 = \left( \frac{5K}{32} + 4 \right) \times 4
  $$
- **OP/B 比率**：
  $$
  \text{OP/B} = \frac{8K}{\left( \frac{5K}{32} + 4 \right) \times 4} \approx \frac{8K}{\frac{5K}{8}} = 12.8 \quad (\text{当 } K \gg 32)
  $$
  **结果**：≈ **12.8 OP/B**（较b再提升60%）

---

### **总结对比**
| **内核类型**               | **OP/B 比率** | **优化效果**                     |
|----------------------------|--------------|----------------------------------|
| a. 基础内核（无优化）      | 0.25 OP/B    | 基准（内存带宽瓶颈）             |
| b. 共享内存分块 (32×32)    | 8 OP/B       | 提升32倍（数据复用）             |
| c. 分块+线程粗化 (因子=4)  | 12.8 OP/B    | 较b再提升60%（减少冗余加载）      |

> **关键结论**：  
> - 共享内存分块通过**数据复用**显著提升OP/B（突破内存墙）。  
> - 线程粗化进一步**减少分块加载次数**，逼近计算峰值。  
> - 实际比率需结合硬件带宽（如A100: 1.5TB/s）评估是否达到计算瓶颈。