本文主要整理PMPP Chapter 5 Memory architecture and data locality的要点。

## 5.4 A tiled matrix multiplication kernel

### **内容概况**
本节基于共享内存的**平铺技术（Tiling）**，实现高性能矩阵乘法内核。通过**线程协作加载数据**、**分阶段计算**与**双重同步机制**，显著减少全局内存访问，提升计算强度（FLOP/B）。

---

### **核心代码解析（图5.9）**

```c
#define TILE_WIDTH 16

__global__ void matrixMulKernel(float* M, float* N, float* P, int Width) {

    __shared__ float Md[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nd[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {

        // Collaborative loading of M and N tiles into shared memory
        Md[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        Nd[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Md[ty][k] * Nd[k][tx];
        }
        __syncthreads();
    }
    P[Row*Width + Col] = Pvalue;
}
```

#### **1. 共享内存声明与初始化**
```cpp
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];  // M矩阵分块 (Line 04)
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];  // N矩阵分块 (Line 05)
```
- **作用域**：块内所有线程共享（对应图示中的"per-block shared memory"）。
- **生命周期**：内核执行期间存在（分阶段复用）。

#### **2. 线程索引计算**
```cpp
int tx = threadIdx.x, ty = threadIdx.y;  // 线程ID (Line 07)
int bx = blockIdx.x, by = blockIdx.y;   // 块ID (Line 08)
int Row = by * TILE_WIDTH + ty;        // 目标P元素行索引 (Line 11)
int Col = bx * TILE_WIDTH + tx;        // 目标P元素列索引 (Line 12)
```
- **寄存器存储**：`tx, ty, bx, by`为自动变量 → 极速访问（对应图示"per-thread registers"）。
- **索引计算原理**：  
  - 每个块负责计算 `TILE_WIDTH × TILE_WIDTH` 的P子矩阵  
  - 线程`(tx,ty)`在块`(bx,by)`中计算`P[Row][Col]`

#### **3. 分阶段计算循环**
```cpp
for (int ph = 0; ph < Width/TILE_WIDTH; ph++) {  // 分阶段 (Line 16)
    // 协作加载分块数据到共享内存
    Mds[ty][tx] = M[Row*Width + (ph*TILE_WIDTH + tx)];  // 加载M (Line 19)
    Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];  // 加载N (Line 20)
    __syncthreads();  // 同步1：确保数据加载完成 (Line 21)

    // 基于共享内存计算部分点积
    for (int k = 0; k < TILE_WIDTH; k++)  // (Line 23)
        Pvalue += Mds[ty][k] * Nds[k][tx]; 
    __syncthreads();  // 同步2：确保数据使用完毕 (Line 26)
}
```
- **加载逻辑**：  
  - 每个线程加载**1个M元素 + 1个N元素**到共享内存  
  - `ph`控制分块偏移（`ph*TILE_WIDTH`为当前分块起始位置）
- **同步机制**：  
  | **同步点** | **依赖类型** | **作用** |  
  |------------|--------------|----------|  
  | `__syncthreads()` 1 | 写后读（True依赖） | 确保所有线程完成数据加载 |  
  | `__syncthreads()` 2 | 读后写（False依赖） | 防止下一阶段覆盖未使用数据 |  

#### **4. 结果写入全局内存**
```cpp
P[Row*Width+Col] = Pvalue;  // 写入结果 (Line 29)
```
- 每个线程将最终结果写入全局内存（对应图示"per-grid global memory"）。

---

### **关键优化技术**
#### **1. 条带挖掘（Strip-mining）**
- **本质**：将长循环`for(int k=0; k<Width; k++)`拆分为`Width/TILE_WIDTH`个阶段。
- **目的**：强制线程在每阶段聚焦**同一数据分块**，最大化共享内存复用。

#### **2. 计算强度提升**
- **原始FLOP/B**：0.25（每8字节访问对应2次浮点操作）
- **TILE_WIDTH=16优化后**：  
  $$ \text{FLOP/B} = 0.25 \times 16 = 4.0 $$
- **性能收益**：  
  - A100 GPU理论算力：1555 GB/s × 4 FLOP/B = **6220 GFLOPS**  
  - 较非平铺版本（389 GFLOPS）提升 **16倍**  

---

### **局限与改进方向**
| **局限** | **后果** | **解决方案** |
|----------|----------|--------------|
| **矩阵宽度需为TILE_WIDTH倍数** | 无法处理任意尺寸矩阵 | 增加边界检查（5.5节） |
| **仅支持方阵** | 无法处理矩形矩阵 | 扩展行列独立计算逻辑 |
| **计算强度仍不足** | 仅达硬件峰值32% | 使用cuBLAS/CUTLASS库进一步优化 |

---

### **与CPU优化的本质差异**
| **平台** | **数据复用机制** | **原因** |
|----------|------------------|----------|
| **CPU** | 依赖缓存隐式复用 | 单线程独占缓存，时间局部性易保障 |
| **GPU** | 显式共享内存管理 | 多线程竞争缓存，需主动控制数据局部性 |

---

### **总结**
平铺矩阵乘法内核通过：
1. **共享内存分块缓存** → 减少全局内存访问  
2. **双重同步机制** → 保障线程协作安全  
3. **条带挖掘** → 实现分阶段计算  
将计算强度提升 **16倍**（0.25 → 4.0 FLOP/B），是突破内存瓶颈的典范实现。后续需处理边界条件以支持通用场景。

## 5.5 Boundary checks

### **内容概况**
本节解决平铺矩阵乘法内核的**通用性问题**，通过引入边界检查机制，使其能处理**任意尺寸矩阵**（非TILE_WIDTH倍数、矩形矩阵）。核心挑战在于防止线程访问**越界内存**导致的未定义行为（数据污染/程序崩溃）。

---

### **边界检查解决方案**

```cpp
// Loop over the M and N tiles required to compute P element
float Pvalue = 0;
for (int ph = 0; ph < ceil(Width/(float)TILE_WIDTH); ++ph) {

    // Collaborative loading of M and N tiles into shared memory
    if ((Row < Width) && (ph*TILE_WIDTH+tx) < Width)
        Md[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
    else Md[ty][tx] = 0.0f;
    if ((ph*TILE_WIDTH+ty) < Width && Col < Width)
        Nd[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
    else Nd[ty][tx] = 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
        Pvalue += Md[ty][k] * Nd[k][tx];
    }
    __syncthreads();
}
if ((Row < Width) && (Col < Width))
    P[Row*Width + Col] = Pvalue;
```

#### **1. 输入加载阶段（M/N矩阵）**
```cpp
// 加载M元素时的边界检查（图5.13 Line 19扩展）
if (Row < Width && (ph * TILE_WIDTH + tx) < Width) 
    Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
else 
    Mds[ty][tx] = 0.0f;  // 越界填充0

// 加载N元素时的边界检查（图5.13 Line 20扩展）
if ((ph * TILE_WIDTH + ty) < Width && Col < Width) 
    Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
else 
    Nds[ty][tx] = 0.0f;  // 越界填充0
```
- **检查逻辑**：同时验证**行索引**和**列索引**是否小于矩阵宽度`Width`
- **填充0原理**：0值在点积计算中为中性元素（`a*0=0`），不影响结果正确性

#### **2. 结果写入阶段（P矩阵）**
```cpp
// 写入P元素前的边界检查（图5.13 Line 29扩展）
if (Row < Width && Col < Width) 
    P[Row*Width + Col] = Pvalue;
```
- **必要性**：部分线程可能负责无效P位置（如图5.12中Block₁,₁的线程(1,0)）
- **避免写入**：防止破坏其他有效内存区域

---

### **关键设计思想**
1. **按需检查**  
   - 每个内存访问点独立检查 → 因越界可能出现在**任意阶段**（非仅最后阶段）
   - 示例：图5.12中Block₁,₁在**Phase 0**即出现越界访问

2. **无效线程仍需协作**  
   - 即使线程不计算有效P（如Block₁,₁的thread₁,₀），仍需参与共享内存加载
   - 否则其他线程无法获取所需数据（如该线程负责加载的`M₂,₁`）

3. **零填充的数学合理性**  
   $$P_{row,col} = \sum_{k} M_{row,k} \times N_{k,col}$$  
   - 若$M_{row,k}$或$N_{k,col}$越界 → 等价于该项为0 → 和式结果不变

---

### **扩展至矩形矩阵**
1. **参数调整**  
   - 输入：替换`Width`为三个独立维度  
     - `M`：$j \times k$ → `dimM`  
     - `N`：$k \times l$ → `dimN`  
     - `P`：$j \times l$ → `dimP`  
   - 索引计算：  
     ```cpp
     int Row = by * TILE_WIDTH + ty;  // 范围 [0, j)
     int Col = bx * TILE_WIDTH + tx;  // 范围 [0, l)
     ```

2. **检查条件更新**  
   - 加载`M`：`Row < j && (ph*TILE_WIDTH + tx) < k`  
   - 加载`N`：`(ph*TILE_WIDTH + ty) < k && Col < l`  
   - 写入`P`：`Row < j && Col < l`  

---

### **总结**
通过三重边界检查：
1. **输入加载检查** → 防越界读（填充0）  
2. **结果写入检查** → 防越界写  
3. **独立维度参数** → 支持矩形矩阵  
平铺矩阵乘法内核最终成为**通用、鲁棒的高性能实现**。此模式可推广至其他需分块优化的算法（如卷积、矩阵分解）。

## 5.6 Impact of memory usage on occupancy

### **内容概况**
本节揭示**寄存器与共享内存用量**对SM线程占用率（Occupancy）的制约关系，提出**动态内存分配策略**以适配不同硬件资源，最大化并行效率。

---

### **关键概念与公式**
#### **1. 占用率瓶颈**
- **硬件资源上限**（以A100为例）：  
  - 共享内存总量：**164 KB/SM**  
  - 最大线程数：**2048 threads/SM**  
- **线程资源消耗公式**：  
  $$\text{资源/线程} = \frac{\text{共享内存总量}}{\text{块线程数}}$$
- **占用率计算公式**：  
  $$\text{占用率} = \frac{\text{实际线程数}}{\text{2048}} \times 100\%$$

#### **2. 平铺矩阵乘法案例**
- **共享内存消耗**：  
  - `Mds` + `Nds` = $2 \times \text{TILE\_WIDTH}^2 \times 4\text{B}$  
  - 线程均耗 = $\frac{8 \times \text{TILE\_WIDTH}^2}{\text{TILE\_WIDTH}^2} = 8\text{B/线程}$  
- **结论**：  
  $8\text{B/线程} < 82\text{B/线程}$ → **不构成瓶颈**（A100支持满占用率）

#### **3. 高消耗内核示例**
- **假设场景**：  
  - 块共享内存：32 KB  
  - 块线程数：256  
- **线程均耗**：$32\text{KB}/256 = 132\text{B/线程}$  
- **最大支持线程数**：$164\text{KB} / 132\text{B} \approx 1241$ 线程  
- **占用率上限**：$1241 / 2048 \approx 62\%$  

---

### **动态内存分配策略**
#### **1. 静态声明局限**
```cpp
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // 编译时固定尺寸
```
- **问题**：无法运行时调整尺寸，适配不同硬件需重新编译。

#### **2. 动态声明方案**
```cpp
// 内核声明（省略尺寸）
extern __shared__ float shared_data[]; 

// 主机端配置
size_t shared_size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
kernel<<<grid, block, shared_size>>>(...);
```
- **原理**：  
  - `extern __shared__` 声明未定尺寸共享数组  
  - 内核启动时通过**第三参数**动态分配字节数  

#### **3. 内核内部分区**
```cpp
__global__ void kernel(...) {
    float* Mds = &shared_data[0]; // Mds起始地址
    float* Nds = &shared_data[TILE_WIDTH * TILE_WIDTH]; // Nds起始地址
    // 线性索引访问：Mds[ty * TILE_WIDTH + tx]
}
```
- **手动分区**：依据分块尺寸计算`Mds`/`Nds`偏移量  
- **访问方式**：一维数组 + 手动计算二维索引  

---

### **设备查询与自适应**
```cpp
// 主机端查询设备属性
cudaDeviceProp devProp;
cudaGetDeviceProperties(&devProp, 0);
size_t max_shared_per_block = devProp.sharedMemPerBlock;

// 动态计算分块尺寸
int tile_width = sqrt(max_shared_per_block / (2 * sizeof(float)));
```
- **关键API**：`cudaGetDeviceProperties` → `sharedMemPerBlock`  
- **自适应逻辑**：根据硬件共享内存上限反推最大`TILE_WIDTH`  

---

### **设计意义总结**
| **策略**          | **优势**                          | **应用场景**               |
|--------------------|-----------------------------------|--------------------------|
| **静态分配**       | 代码简洁，访问直观                | 固定硬件/算法场景         |
| **动态分配**       | 跨硬件适配，资源利用率高          | 通用库开发（如cuBLAS）    |
| **设备查询**       | 避免超限，保证兼容性              | 异构计算平台部署          |

---

### **优化本质**
通过**共享内存用量控制**与**动态资源分配**：  
1. 提升SM内**并发线程数** → 增强延迟隐藏能力  
2. 避免资源竞争 → 逼近硬件峰值占用率  
3. 实现**“算得慢但跑得满”优于“算得快但跑得少”**  

此章为算法级优化（如平铺）与系统级调优（如占用率）的衔接点，是解锁GPU极限性能的关键一环。

## 5.7 Summary

### **核心问题：内存墙（Memory Wall）**
- **现象**：程序性能受限于**全局内存访问速度**（高延迟、有限带宽）。  
- **量化指标**：**计算强度（FLOP/B）** = 浮点操作数 / 全局内存字节访问量  
  - 强度低 → **内存受限（Memory-bound）**  
  - 示例：原始矩阵乘法仅 **0.25 FLOP/B**（A100算力仅389 GFLOPS → 峰值2%）  

---

### **解决方案：内存层次优化**
#### **1. 利用高速内存资源**
| **内存类型**   | **特性**                              | **优化作用**                          |
|----------------|---------------------------------------|---------------------------------------|
| **寄存器**     | 线程私有，纳秒级访问                  | 存储频繁访问的标量（如循环计数器）    |
| **共享内存**   | 块内共享，高带宽低延迟                | 缓存分块数据（Tiling技术核心）         |
| **常量内存**   | 全局只读，缓存加速广播访问            | 存储滤波器系数等不变数据              |

#### **2. 平铺技术（Tiling）**
- **本质**：数据分块 → 共享内存缓存 → 分阶段计算  
- **流程**：  
  1. **协作加载**：线程块将全局内存数据块加载到共享内存  
  2. **屏障同步**：`__syncthreads()` 确保数据就绪  
  3. **局部计算**：基于共享内存进行高效计算  
  4. **结果回写**：仅有效线程写入全局内存  
- **收益**：  
  - 计算强度↑ **TILE_WIDTH倍**（0.25 → 4.0 FLOP/B）  
  - 全局内存访问量↓ **至1/TILE_WIDTH**  

#### **3. 边界处理**
- **挑战**：非方阵/尺寸非TILE_WIDTH倍数时越界访问  
- **方案**：  
  - **加载检查**：`if (index < Width)` 否则填充0  
  - **写入检查**：仅有效位置写入结果  
- **数学合理性**：越界数据填充0不影响点积结果  

#### **4. 资源约束与占用率**
- **关键限制**：  
  - 共享内存容量（如A100：164 KB/SM）  
  - 寄存器数量（影响SM最大线程数）  
- **动态适配**：  
  ```cpp
  extern __shared__ float buffer[];  // 动态声明
  kernel<<<grid, block, shared_size>>>(...); // 运行时分配
  ```
- **占用率公式**：  
  $$\text{最大线程数} = \min\left(\frac{\text{共享内存总量}}{\text{每块用量}}, \frac{\text{寄存器总量}}{\text{每线程用量}}\right)$$  

---

### **跨平台通用性**
- **CPU同理**：平铺技术提升**缓存命中率**（时间/空间局部性）  
- **差异**：  
  - CPU：依赖硬件缓存隐式优化  
  - GPU：需显式管理共享内存（因多线程竞争缓存）  

---

### **未竟议题与进阶方向**
1. **寄存器优化**：Part II讨论分块算法中寄存器的使用  
2. **高级优化库**：cuBLAS/CUTLASS实现近峰值性能  
3. **非矩阵场景**：卷积（第7章）、排序等算法的平铺适配  

---

### **终极洞见**
- **优化本质**：通过**数据复用** ↑分子（FLOP）、**减少DRAM访问** ↓分母（B）→ 提升FLOP/B突破内存墙。  
- **哲学启示**：  
  > “在计算机科学中，所有问题都可以通过增加一层间接性解决，除了太多间接性导致的问题。”  
  > ——此处“间接层”即**共享内存缓存**，平衡了速度与容量矛盾。  

此章为GPU高性能编程的基石，掌握内存优化即解锁算力之钥。