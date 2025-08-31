本文主要整理PMPP Chapter 3 Multidimensional grids and data的要点。

## 3.0 前言

### 内容概括

本章是继第一章（介绍使用一维网格线程处理一维数组）之后的知识进阶。它将线程组织的概念从**一维扩展到多维**，重点讲解如何使用二维和三维的线程网格（Grid）和线程块（Block）来高效地处理同样也是多维的数据结构（如二维图像和矩阵）。

本章通过多个**实际应用案例**（彩色图像转灰度图、图像模糊滤波、矩阵乘法）来具体说明和巩固这些概念。这些例子的目的是帮助读者建立对数据并行性的直观理解和编程思维，为后续深入学习GPU硬件架构、内存模型和性能优化技术打下坚实的基础。

---

### 要点总结

1.  **核心主题：线程组织的多维化**
    *   从第一章的**一维**网格（`vector`）和**一维**数组处理，扩展到本章的**多维**（二维/三维）网格和块来映射和处理**多维**数据（如图像、矩阵）。
    *   这是CUDA编程中至关重要的一步，因为现实世界中的许多数据（图像、视频、科学计算数据）本质上是多维的。

2.  **关键概念：线程层次结构**
    *   **线程（Thread）**：最基本的执行单元。
    *   **线程块（Block）**：由一组线程组成，块内的线程可以通过**共享内存**进行协作和通信，并且可以**同步**。
    *   **网格（Grid）**：由多个线程块组成，是内核函数启动时在GPU上执行的最高层次线程集合。
    *   本章重点是如何利用 `blockIdx`、`threadIdx`、`blockDim`、`gridDim` 这些内置变量，在多维布局下正确计算每个线程的唯一标识符（ID），从而让每个线程都能找到自己所要处理的那部分数据。

3.  **应用场景（案例学习）**
    *   **彩色转灰度图像**：一个典型的“一对一”映射，即网格中的**每个线程**处理图像中的**一个像素**。这是一个简单的embarrassingly parallel问题，用于引入多维线程索引的计算。
    *   **图像模糊**：一个需要**邻域操作**的例子。每个线程处理一个输出像素，但需要读取其周围多个像素（一个窗口）的值进行计算。这引入了**数据复用**的概念，并为后续学习共享内存优化做了铺垫。
    *   **矩阵乘法**：一个更为复杂的例子，展示了如何将大规模计算问题（计算输出矩阵C的每个元素）分解到成千上万的线程上。它清晰地体现了**数据并行性**的威力——每个输出元素的计算相互独立，可以完全并行处理。这个例子也会自然引出对**全局内存访问模式**（合并访问）的初步思考。

4.  **本章的学习定位（承上启下）**
    *   本章的核心目标是帮助读者建立**数据并行化的思维模式**，即如何将一个计算任务分解，并由大量线程并行完成。
    *   它侧重于**算法和编程模型**的理解，是学习后续章节（GPU架构、内存组织、性能优化）的**必要前提**。只有在理解了“要做什么”之后，才能更好地学习“如何做得更快”。本章的案例（如图像模糊）在后续章节中通常会再次被用作例子，以展示如何通过优化内存访问等策略来提升性能。

总而言之，**第二章是从CUDA基础编程迈向高效实用编程的关键一步，它将线程组织与实际问题中的数据结构对齐，并通过实例教学为后续的深度优化打下概念基础。**

## 3.1 Multidimensional grid organization

### 内容概括

本节详细阐述了CUDA编程模型中线程组织的核心机制——**多维网格（Grid）和线程块（Block）**。它解释了如何通过执行配置参数（`<<<grid, block>>>`）在主机端定义网格和块的维度（使用`dim3`类型），以及如何在设备端的内核函数中通过内置变量（`gridDim`, `blockDim`, `blockIdx`, `threadIdx`）访问这些维度信息和线程的唯一坐标。核心在于理解如何利用这些坐标（索引）将线程映射到要处理的多维数据元素上。本节还明确了网格和块维度的取值范围以及每个块的最大线程数限制（1024）。

---

### 要点总结

1.  **线程组织层级与索引变量：**
    *   **网格 (Grid)**：最高层级，包含多个线程块。维度由执行配置的第一个参数 (`grid`) 指定。
    *   **线程块 (Block)**：网格的组成单元，包含多个线程。维度由执行配置的第二个参数 (`block`) 指定。块内的线程可以协作（共享内存、同步）。
    *   **线程 (Thread)**：最基本的执行单元。
    *   **内核内置变量：**
        *   `gridDim`： (内核内) 网格的维度（x, y, z 分量表示各维度包含的块数）。
        *   `blockDim`： (内核内) 块的维度（x, y, z 分量表示各维度包含的线程数）。
        *   `blockIdx`： (内核内) 当前线程所在块在网格中的索引坐标（x, y, z）。
        *   `threadIdx`： (内核内) 当前线程在其所属块内的索引坐标（x, y, z）。

2.  **执行配置 (`<<<grid, block>>>`)：**
    *   在主机端调用内核时指定网格和块的维度。
    *   使用 `dim3` 类型变量定义维度。`dim3` 是一个包含 x, y, z 三个整数字段的结构体。
    *   可以定义 **1D, 2D, 3D** 的网格和块。未使用的维度大小设为 1。
    *   **1D 快捷方式：** 可以直接用整数值代替 `dim3` 变量，编译器会将其解释为 x 维度大小，y 和 z 维度默认为 1 (例如 `<<<128, 256>>>` 表示 1D 网格有 128 个块，每个 1D 块有 256 个线程)。

3.  **维度范围限制：**
    *   **网格维度范围：**
        *   `gridDim.x`: 1 到 2³¹ - 1 (非常大)
        *   `gridDim.y`, `gridDim.z`: 1 到 2¹⁶ - 1 (65,535)
    *   **块索引范围：**
        *   `blockIdx.x`: 0 到 `gridDim.x - 1`
        *   `blockIdx.y`: 0 到 `gridDim.y - 1`
        *   `blockIdx.z`: 0 到 `gridDim.z - 1`
    *   **线程索引范围：**
        *   `threadIdx.x`: 0 到 `blockDim.x - 1`
        *   `threadIdx.y`: 0 到 `blockDim.y - 1`
        *   `threadIdx.z`: 0 到 `blockDim.z - 1`
    *   **块大小限制：** 每个块包含的总线程数 **不能超过 1024**。线程可以在 x, y, z 维度上任意分布，只要乘积 ≤ 1024 (例如 (512,1,1), (8,16,4), (32,16,2) 合法；(32,32,2) 非法)。

4.  **网格与块的维度独立性：**
    *   网格和其包含的块**不需要具有相同的维度数**。例如，可以有一个 2D 网格（`gridDim.y > 1, gridDim.z=1`）包含 3D 块（`blockDim.z > 1`），或者反之。

5.  **索引计算与数据映射：**
    *   线程的唯一全局标识（用于定位要处理的数据）通常需要结合 `blockIdx`, `blockDim`, `threadIdx` 进行计算。例如，在 2D 网格处理 2D 图像时，一个像素 (i, j) 可能由 `blockIdx.y * blockDim.y + threadIdx.y` 和 `blockIdx.x * blockDim.x + threadIdx.x` 对应的线程处理。
    *   图 3.1 的示例（`gridDim = (2, 2, 1)`, `blockDim = (4, 2, 2)`）清晰地展示了网格（4个块）、块（16个线程/块）和线程索引（x, y, z）的三维组织结构。

**核心目的：** 本节为理解和使用CUDA的多维线程组织来处理多维数据（如图像、矩阵）奠定了理论基础和实践方法，是后续学习图像处理、矩阵运算等应用的关键基础。

## 3.2 Mapping threads to multidimensional data

### **内容概括**
本节深入探讨了**如何将多维线程组织（Grid/Block）映射到多维数据（如图像、矩阵）** 的具体方法。通过**彩色图像转灰度图**的案例，详细说明：
1. **线程索引与数据坐标的换算关系**（行/列坐标计算）
2. **边界处理机制**（多余线程的屏蔽方法）
3. **多维数组的线性化存储原理**（行优先布局）
4. **动态多维数组的访问技巧**（手动计算1D偏移量）
5. **实际执行中的线程利用率分析**（图3.5的四种区块案例）

---

```c
__global__
void rgb_to_grayscale_kernel(unsigned char* output, unsigned char* input, int width, int height) {
    const int channels = 3;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int outputOffset = row * width + col;
        int inputOffset = (row * width + col) * channels;

        unsigned char r = input[inputOffset + 0];   // red
        unsigned char g = input[inputOffset + 1];   // green
        unsigned char b = input[inputOffset + 2];   // blue

        output[outputOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

dim3 threads_per_block(16, 16);     // using 256 threads per block
dim3 number_of_blocks(cdiv(width, threads_per_block.x),
                      cdiv(height, threads_per_block.y));
rgb_to_grayscale_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height
    );
```

### **要点总结**

#### **1. 线程索引 → 数据坐标映射**
- **核心公式**（以2D图像处理为例）：
  ```c
  row = blockIdx.y * blockDim.y + threadIdx.y  // 行坐标（垂直方向）
  col = blockIdx.x * blockDim.x + threadIdx.x  // 列坐标（水平方向）
  ```
- **示例**：  
  处理62×76像素图像时，若使用16×16线程块：
  - 需 `ceil(62/16)=4` 行块 + `ceil(76/16)=5` 列块 → **共20个块**
  - 线程(0,0)在块(1,0)中处理的像素坐标为：  
    `row = 1*16+0=16`, `col = 0*16+0=0` → 像素 `P[16][0]`

#### **2. 边界处理（关键机制）**
- **问题**：线程数可能**超过实际数据量**（如62×76图像需80×64线程）
- **解决方案**：在内核中添加**条件判断**，屏蔽无效线程
  ```c
  if (col < width && row < height) { 
      // 仅有效线程执行计算
  }
  ```
- **执行效率**（图3.5案例）：
  - **区域1**（内部块）：100%线程有效（16×16=256）
  - **区域2**（右边界）：192线程有效（16行×12列）
  - **区域3**（下边界）：224线程有效（14行×16列）
  - **区域4**（右下角）：168线程有效（14行×12列）

#### **3. 多维数组的线性化存储**
- **行优先布局（Row-Major）**：  
  内存中按行连续存储（C/C++/CUDA默认）  
  **偏移量公式**：`index = row * width + col`  
  !https://example.com/row-major.png
- **列优先布局（Column-Major）**：  
  FORTRAN使用，CUDA中需手动转置

#### **4. 动态多维数组访问**
- **挑战**：CUDA C不支持动态二维数组语法（如 `P_d[j][i]`）
- **解决方案**：手动计算1D偏移量
  ```c
  // 灰度图输出（单通道）
  int grayOffset = row * width + col; 
  Pout_d[grayOffset] = grayscaleValue;

  // 彩色图输入（RGB三通道）
  int rgbOffset = (row * width + col) * 3; 
  unsigned char r = Pin_d[rgbOffset];
  unsigned char g = Pin_d[rgbOffset + 1];
  unsigned char b = Pin_d[rgbOffset + 2];
  ```

#### **5. 三维数据扩展**
- **坐标计算**：  
  ```c
  depth = blockIdx.z * blockDim.z + threadIdx.z;
  row   = blockIdx.y * blockDim.y + threadIdx.y;
  col   = blockIdx.x * blockDim.x + threadIdx.x;
  ```
- **线性化索引**：  
  `index = depth * (height * width) + row * width + col`

---

### **关键结论**
- **线程组织维度需匹配数据结构**（2D图像→2D网格/块）
- **边界检查必不可少**：防止无效线程越界访问
- **行优先布局是核心**：掌握 `index = row*width + col` 的推导与应用
- **动态数组需手动线性化**：通过偏移量访问多维数据
- **实际线程利用率受数据尺寸影响**：设计时需权衡块大小与边界浪费

## 3.3 Image blur: a more complex kernel

### **内容概括**

本节通过一个**图像模糊（Image Blur）** 的实际案例，展示了比前两个例子（向量加法和灰度转换）更为复杂的CUDA内核。其复杂性体现在：
1.  **计算复杂性**：每个输出像素的值不再是单个输入像素的简单转换，而是需要由其周围一个区域（N x N 像素块）的像素值**共同计算**得出。
2.  **线程协作需求**：虽然线程间没有直接的通信，但每个线程都需要读取**大量其他线程负责的数据**（邻域像素），这引入了**数据复用**的概念。
3.  **边界条件处理**：当处理图像边缘的像素时，其所需的邻域会超出图像边界，内核必须包含复杂的逻辑来**处理这些特殊情况**，防止非法内存访问。

本节详细阐述了实现均值模糊的数学原理、内核代码的结构，并重点分析了边界处理的机制。

---

### **要点总结**

```c
#include <stdio.h>

// 常量定义：模糊半径（控制模糊程度）
#define BLUR_SIZE 1  // 3x3滤波核
// #define BLUR_SIZE 2 // 5x5滤波核

__global__ void blurKernel(unsigned char* in, unsigned char* out, int width, int height) {
    // 计算当前线程负责的输出像素坐标
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 边界检查：跳过超出图像边界的线程
    if (col >= width || row >= height) 
        return;
    
    // 初始化累加器和像素计数器
    int pixVal = 0;
    int pixels = 0;
    
    // 遍历滤波核区域 (N x N 邻域)
    for (int dy = -BLUR_SIZE; dy <= BLUR_SIZE; dy++) {
        for (int dx = -BLUR_SIZE; dx <= BLUR_SIZE; dx++) {
            // 计算邻域像素坐标
            int curRow = row + dy;
            int curCol = col + dx;
            
            // 检查坐标是否在图像范围内
            if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                // 计算线性索引（行优先）
                int idx = curRow * width + curCol;
                // 累加像素值
                pixVal += in[idx];
                // 增加有效像素计数
                pixels++;
            }
        }
    }
    
    // 计算输出图像线性索引
    int outIdx = row * width + col;
    // 计算平均值并写入输出（避免除以零）
    out[outIdx] = (unsigned char)(pixels > 0 ? pixVal / pixels : 0);
}

int main() {
    // 图像参数（示例值）
    int width = 76;   // 图像宽度
    int height = 62;  // 图像高度
    
    // 创建图像缓冲区
    size_t imgSize = width * height * sizeof(unsigned char);
    unsigned char* d_in, *d_out;
    
    // GPU内存分配
    cudaMalloc(&d_in, imgSize);
    cudaMalloc(&d_out, imgSize);
    
    // 线程块配置（16x16线程块）
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    // 启动内核
    blurKernel<<<gridSize, blockSize>>>(d_in, d_out, width, height);
    
    // 清理
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}
```

#### **1. 算法原理：均值模糊**
*   **核心思想**：输出图像中的每个像素值，是其输入图像中**对应像素及其周围邻域像素值的平均值**。
*   **邻域块（Patch）**：以一个目标像素为中心，取一个 `(2*BLUR_SIZE+1) x (2*BLUR_SIZE+1)` 的方形区域。例如，`BLUR_SIZE = 1` 对应 3x3 区域。
*   **数学表示**：`输出像素 = (邻域内所有有效像素值之和) / (邻域内有效像素的个数)`

#### **2. 线程到数据的映射**
*   **映射策略保持不变**：与灰度转换例子相同，采用 **“一个线程计算一个输出像素”** 的映射策略。
*   线程使用 `blockIdx`, `blockDim`, `threadIdx` 计算出其负责的输出像素坐标 `(row, col)`。

#### **3. 内核实现的关键机制**
*   **嵌套循环遍历邻域**：每个线程使用两层循环，遍历以其 `(row, col)` 为中心的整个邻域块的所有像素。
    *   外循环：遍历行 (`curRow` 从 `row-BLUR_SIZE` 到 `row+BLUR_SIZE`)
    *   内循环：遍历列 (`curCol` 从 `col-BLUR_SIZE` 到 `col+BLUR_SIZE`)
*   **累加求和**：在内循环中，将每个邻域像素的值累加到一个变量 `pixVal` 中，并用 `pixels` 变量记录实际累加的有效像素个数。
*   **计算平均值**：循环结束后，用 `pixVal / pixels` 得到平均值，并写入输出图像。

#### **4. 边界处理（核心与难点）**
*   **问题**：处理图像边缘像素时，邻域会超出图像边界（如图3.9所示），访问到无效内存地址。
*   **解决方案**：在内循环中，在累加之前增加一个 **`if` 条件判断**：
    ```c
    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
        // 只有坐标合法的像素才会被累加
        pixVal += input_image[curRow * width + curCol];
        pixels++;
    }
    ```
*   **边界处理的后果**：
    *   **内部像素**（Case 4）：邻域完全在图像内，`pixels = 9`（对于3x3），计算正常的平均值。
    *   **边缘像素**（Case 2/3）：邻域部分超出，`pixels = 6`，平均值只由有效的6个像素决定。
    *   **角落像素**（Case 1）：邻域大部分超出，`pixels = 4`，平均值只由有效的4个像素决定。
    *   这样确保了算法的正确性和鲁棒性，但导致了**不同位置的线程计算量略有不同**。

#### **5. 与简单内核的对比**
| 特征 | 简单内核（VecAdd, Grayscale） | 复杂内核（Image Blur） |
| :--- | :--- | :--- |
| **数据访问** | 每个线程只访问一个输入位置 | 每个线程访问一个输入区域（N*N个位置） |
| **计算** | 简单、独立 | 复杂（循环、累加、除法）、但仍独立 |
| **协作** | 无 | **隐式协作**（读取其他线程负责的数据） |
| **边界处理** | 简单if判断，跳过整个线程 | 复杂if判断，在循环内部跳过无效数据点 |

**总结**：图像模糊内核是理解** stencil/convolution （卷积）模式**的入门案例。它展示了如何让线程处理更复杂的计算模式，以及如何安全高效地处理边界条件，为后续学习更复杂的并行模式（如共享内存优化）打下了基础。

## 3.4 Matrix multiplication

### **内容概括**
本节详细阐述了**CUDA实现矩阵乘法**的核心方法：
1. **数学基础**：矩阵乘法定义为行-列点积运算（\( P_{row,col} = \sum_{k=0}^{Width-1} M_{row,k} \times N_{k,col} \)）
2. **线程映射策略**：采用输出矩阵驱动的并行化模式，每个线程负责计算一个输出元素 \( P_{row,col} \)
3. **数据访问模式**：利用行优先存储规则推导全局内存访问公式
4. **边界处理**：通过条件判断确保线程仅在有效范围内计算
5. **性能限制与扩展**：讨论大规模矩阵的拆分解法

---

### **要点总结**

#### **1. 数学原理（图3.10图示）**
- **输出元素计算**：  
  \( P_{row,col} \) = 第`row`行(M) 与第`col`列(N) 的**点积**  
  \( P_{row,col} = \sum_{k=0}^{Width-1} M_{row,k} \times N_{k,col} \)
- **简化假设**：仅处理方阵（所有维度 = `Width`）

#### **2. 线程组织与数据映射**
| **组件** | **计算公式** | **说明** |
|----------|--------------|----------|
| **线程坐标** | `row = blockIdx.y*blockDim.y + threadIdx.y` <br> `col = blockIdx.x*blockDim.x + threadIdx.x` | 线程ID直接对应P的元素坐标 |
| **输出矩阵分割** | - | P被划分为区块（图3.12示例）|
| **执行配置** | `dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH)` <br> `dim3 gridSize(Width/BLOCK_WIDTH, Width/BLOCK_WIDTH)` | 需保证`Width % BLOCK_WIDTH == 0` |

#### **3. 内存访问关键公式**
| **矩阵** | **元素访问公式** | **原理** |
|----------|------------------|----------|
| **输入矩阵 M** | `M[row * Width + k]` | 行优先存储：访问第`row`行第`k`列 |
| **输入矩阵 N** | `N[k * Width + col]` | 行优先存储：访问第`k`行第`col`列 |
| **输出矩阵 P** | `P[row * Width + col]` | 行优先存储：写入计算结果的坐标 |

#### **4. 核心计算内核（图3.11代码）**
```c
__global__ void matrixMulKernel(float* M, float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < Width && col < Width) {
        float Pvalue = 0;
        for (int k = 0; k < Width; k++) {
            // 关键计算：点积累加
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue; // 写入结果
    }
}
```
**执行流程**（以图3.12/3.13为例）：
1. 线程(0,0)在块(0,0)计算 \( P_{0,0} \)
2. 遍历 `k=0` 到 `3`：
   - `k=0`：取 `M[0*4+0]` 和 `N[0*4+0]` → \( M_{0,0} \times N_{0,0} \)
   - `k=1`：取 `M[0*4+1]` 和 `N[1*4+0]` → \( M_{0,1} \times N_{1,0} \)
   - ...（累加4次乘积）

#### **5. 限制与扩展方案**
| **限制** | **解决方案** |
|----------|--------------|
| **网格尺寸上限** | 分割大矩阵为子矩阵，多次启动内核 |
| **线程利用率** | 单线程计算多个P元素（提升计算/内存比） |
| **非方阵支持** | 扩展内核：独立设置高度/宽度参数 |

---

### **关键结论**
1. **一对一映射**：线程坐标与输出矩阵元素坐标直接对应
2. **内存访问模式**：
   - M：**行连续访问**（`row`固定，`k`变化）
   - N：**列跳跃访问**（`k`变化导致跨行访问 → 潜在性能瓶颈）
3. **可扩展性**：通过子矩阵分解支持超大矩阵计算
4. **优化方向**：后续章节将解决跨行访问问题（共享内存/分块算法）

> 此实现为**基础版本**，实际应用需优化内存访问（见第5章优化技术）

## 3.5 Summary

### 内容概括

本节是对CUDA中**多维线程组织**的核心概念和其重要性的一个总结。它强调了使用多维的网格（Grid）和块（Block）来高效处理多维数据（如图像、矩阵）是CUDA编程的基本范式。程序员的责任是利用内置的坐标变量（`blockIdx`, `threadIdx`）将线程唯一地映射到数据上，并掌握将多维数据**线性化**到一维内存空间的技巧。这些技能是理解更复杂的并行模式和进行性能优化的基础。

---

### 要点总结

1.  **核心目的：数据映射**
    *   多维网格和块的设计**主要是为了更方便地组织和映射线程，以处理同样是多维的数据结构**（例如2D图像、3D体数据、矩阵）。

2.  **执行配置 (`<<<grid, block>>>`)**
    *   内核启动参数**定义了网格和块的维度**，从而决定了线程的组织层次结构。

3.  **线程标识与数据域**
    *   **`blockIdx`** 和 **`threadIdx`** 是内置变量，让每个线程都能在网格和块中获取自己**唯一的、多维的坐标**。
    *   程序员必须在内核代码中**主动使用这些坐标变量**，来计算每个线程应该处理哪一部分数据。这是程序员的核心职责。

4.  **线性化 (Linearization)**
    *   由于C语言中动态分配的多维数组在内存中实际上是以**行优先（row-major）** 的方式存储在一维连续空间中的，因此程序员经常需要**将多维索引转换（线性化）为一维偏移量**来访问数组元素。这是一个关键技巧。

5.  **学习路径与重要性**
    *   通过**由浅入深（From Simple to Complex）** 的示例（如向量加法→图像处理→矩阵乘法）来让读者熟悉这些机制。
    *   这里学到的基本技能是**基础性的（Foundational）**，为后续学习更高级的**并行模式（Parallel Patterns）** 和**优化技术（Optimization Techniques）** 至关重要。

### 核心思想图示

| **概念** | **作用** | **关键变量/操作** |
| :--- | :--- | :--- |
| **多维网格/块** | 组织线程结构，匹配数据维度 | `dim3 gridDim`, `dim3 blockDim` |
| **线程唯一标识** | 区分线程，定位数据段 | `blockIdx`, `threadIdx` |
| **线性化** | 访问扁平内存中的多维数据 | `index = row * width + col` |

**总而言之**：这段文字强调了**“通过多维线程坐标映射到多维数据，并掌握线性化方法”** 是CUDA编程的基本功，是所有后续学习的基础。