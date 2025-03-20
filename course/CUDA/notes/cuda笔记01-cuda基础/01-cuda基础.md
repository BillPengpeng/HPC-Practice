## 1. GPU架构与线程层次映射关系

在CUDA编程中，理解GPU架构与线程层次（如block、grid、warp）的映射关系对优化程序性能至关重要。

### 1.1 CUDA线程模型的核心抽象
#### **(1) Thread Hierarchy**
- **Thread（线程）**：最小的执行单元，每个线程独立执行相同的代码（SIMT模型）。
- **Block（线程块）**：一组线程的集合，共享同一块共享内存（Shared Memory），可通过同步（`__syncthreads()`）协作。
- **Grid（网格）**：由多个Block组成的集合，每个Grid对应一个Kernel函数的全局执行范围。

#### **(2) 编程模型示意图**
```text
Grid → [Block0, Block1, ..., BlockN]
每个Block → [Thread0, Thread1, ..., ThreadM]
```

### 1.2. GPU硬件架构的物理映射
GPU由多个 **SM（Streaming Multiprocessor）** 组成，每个SM包含多个 **CUDA Core**（或类似的计算单元）。以下为关键映射关系：

#### **(1) SM（流多处理器）**
- **角色**：SM是GPU的核心计算单元，负责执行线程块（Block）。
- **资源限制**：
  - 每个SM可同时驻留多个Block（例如NVIDIA Ampere架构的SM最多16个Block）。
  - 资源（如寄存器、共享内存）总量限制SM内同时驻留的Block数量。

#### **(2) Block到SM的分配**
- GPU调度器将Grid中的Block分配到空闲的SM上。
- **一个SM可同时执行多个Block**，具体数量由以下因素决定：
  - **寄存器数量**：每个线程的寄存器使用量。
  - **共享内存大小**：每个Block分配的共享内存。
  - **线程数上限**：每个SM支持的线程总数（如Ampere架构每个SM最多2048线程）。

#### **(3) Warp（线程束）**
- **定义**：32个线程的集合（硬件调度基本单位）。
- **SIMT执行**：一个Warp内的所有线程执行相同的指令（但可能因分支导致部分线程停顿）。
- **Warp调度**：SM以Warp为单位调度线程，通过隐藏延迟（如内存访问）提高吞吐量。

#### **(4) Architecture of a modern GPU**

A typical CUDA-capable GPU is organized into an array of highly threaded **streaming multiprocessors (SMs)**. Each SM has several processing units called **streaming processors or CUDA cores**. **Multiple blocks** are likely to be simultaneously assigned to the same SM. However, blocks need to reserve hardware resources to execute, so only a limited number of blocks can be simultaneously assigned to a given SM.  

In most implementations to date, once a block has been assigned to an SM, it is further divided into **32-thread units called warps**. The size of warps is implementation specific and can vary in future generations of GPUs. (threadIdx.x, threadIdx.y, threadIdx.z) 组织warps时， **优先级 x > y > z， 参考cuda mode Lecture 4**。 When threads in the same warp follow different execution paths, we say that these threads exhibit **control/warp divergence**, that is, they diverge in their execution. If all threads in a warp must complete a phase of their execution before any of them can move on, one must use a **barrier synchronization mechanism such as __syncwarp()** to ensure correctness.

### 1.3. 关键映射关系
#### **(1) Block与SM的映射**
- **动态分配**：Grid中的Block会被动态分配到多个SM上。
- **资源竞争**：若Block消耗过多资源（如共享内存），SM内同时执行的Block数量会减少。

#### **(2) Thread与Warp的映射**
- **自动分组**：每个Block内的线程按连续32个为一组，形成Warp。
  - 例如，Block大小为128线程 → 4个Warp。
- **Warp调度**：SM内的Warp调度器轮流执行就绪的Warp，最大化计算单元利用率。

#### **(3) 物理执行流程**
1. **Kernel启动**：CPU调用Kernel，生成Grid。
2. **Block分配**：GPU将Block分配到空闲SM。
3. **Warp划分**：SM将Block内的线程划分为Warp。
4. **指令执行**：SM的CUDA Core执行Warp中的指令，通过流水线隐藏延迟。

### 1.4. 编程优化要点
#### **(1) Block大小的选择**
- **目标**：最大化SM的资源利用率。
- **推荐规则**：
  - Block线程数应为32的倍数（如128、256、512）。
  - 根据资源限制调整Block数量（例如使用`cudaOccupancyMaxPotentialBlockSize`工具）。

#### **(2) 避免Warp分歧（Divergence）**
- **分支语句**：若Warp内线程执行不同分支（如`if-else`），会导致串行化。
- **优化策略**：
  - 尽量让同一Warp内的线程走相同分支。
  - 使用`__syncwarp()`同步部分线程。

#### **(3) 内存访问优化**
- **合并访问**：同一Warp内的线程应访问连续内存地址（利用全局内存的合并访问特性）。
- **共享内存**：利用共享内存减少全局内存访问延迟。

#### **(4) Getting good occupancy – balance resources**

Have 82 SM → **many blocks = good** (for comparison Jetson Xavier has 8 Volta SM).  
Can schedule up to 1536 threads per SM → power of two **block size < 512** desirable (some other GPUs 2048).  
**Avoid divergence** to execute an entire warp (32 threads) at each cycle.  
**Avoid FP64/INT64** if you can on Gx102 (GeForce / Workstation GPUs).  
Shared Memory and Register File → **limits number of scheduled on SM**. (use __launch_bounds__ / C10_LAUNCH_BOUNDS to advise compiler of # of threads for register allocation, but register spill makes things slow).  
Use torch.cuda.get_device_properties(<gpu_num>) to get properties (e.g. max_threads_per_multi_processor)

### 1.5. 示例：Ampere架构的典型参数
以NVIDIA Ampere架构（如A100 GPU）为例：
- **每个SM的配置**：
  - 最大线程数：2048
  - 最大Block数：16
  - 每个Block最大线程数：1024
- **资源限制公式**：
  ```
  同时驻留Block数 = min(
      SM支持的最大Block数,
      SM寄存器总量 / (每个Block寄存器需求),
      SM共享内存总量 / (每个Block共享内存需求)
  )
  ```

### 1.6. 总结
- **逻辑层**（编程模型）：通过Grid、Block、Thread组织并行任务。
- **物理层**（硬件架构）：SM执行Block，Warp是调度单位。
- **优化核心**：合理分配Block和Warp，避免资源竞争与分支分歧。


## 2. cuda基本函数

### 2.1 限定词

| Qualifier keyword  | Callable From | Executed on| Executed by | 
| :--:   | :--:    | :--:      | :--:    | 
| \_\_host\_\_ | Host | Host | Caller host thread | 
| \_\_global\_\_ | Host | Device | New grid of device threads | 
| \_\_device\_\_ | Device | Device | Caller device thread | 

### 2.2 设备管理
#### **(1) 设备初始化与信息查询**
- **`cudaGetDeviceCount(int* count)`**  
  获取可用GPU设备的数量。  
  ```c
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  ```

- **`cudaSetDevice(int deviceId)`**  
  选择要使用的GPU设备（默认使用设备0）。  
  ```c
  cudaSetDevice(0); // 使用第一个GPU
  ```

- **`cudaGetDeviceProperties(cudaDeviceProp* prop, int deviceId)`**  
  获取GPU设备的属性（如计算能力、核心数、内存大小等）。  
  ```c
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device Name: %s\n", prop.name);
  ```

### 2.3 内存管理
#### **(1) 设备内存分配与释放**
- **`cudaMalloc(void** devPtr, size_t size)`**  
  在GPU上分配全局内存。  
  ```c
  float *d_data;
  cudaMalloc(&d_data, 1024 * sizeof(float)); // 分配1024个float
  ```

- **`cudaFree(void* devPtr)`**  
  释放设备内存。  
  ```c
  cudaFree(d_data);
  ```

#### **(2) 主机与设备内存拷贝**
- **`cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)`**  
  在主机（CPU）与设备（GPU）之间拷贝数据。  
  ```c
  float h_data[1024];
  cudaMemcpy(d_data, h_data, 1024 * sizeof(float), cudaMemcpyHostToDevice); // CPU→GPU
  cudaMemcpy(h_data, d_data, 1024 * sizeof(float), cudaMemcpyDeviceToHost); // GPU→CPU
  ```

#### **(3) 其他内存操作**
- **`cudaMallocHost(void** ptr, size_t size)`**  
  分配页锁定主机内存（Pinned Memory），加速数据传输。  
  ```c
  float *h_pinned;
  cudaMallocHost(&h_pinned, 1024 * sizeof(float));
  ```

- **`cudaMallocManaged(void** ptr, size_t size)`**  
  分配统一内存（Unified Memory），CPU和GPU共享访问。  
  ```c
  float *u_data;
  cudaMallocManaged(&u_data, 1024 * sizeof(float));
  ```

### **2.3 核函数（Kernel）启动**
- **核函数调用语法**  
  使用三重尖括号 `<<<grid, block>>>` 指定线程组织方式。  
  ```c
  // 定义核函数
  __global__ void addKernel(float *a, float *b, float *c) {
      int i = threadIdx.x;
      c[i] = a[i] + b[i];
  }

  // 启动核函数
  addKernel<<<1, 1024>>>(d_a, d_b, d_c); // 1个Block，每个Block1024个线程
  ```

### **2.4 线程同步**
#### **(1) 设备级同步**
- **`cudaDeviceSynchronize()`**  
  等待所有设备操作完成（常用于调试或计时）。  
  ```c
  addKernel<<<1, 1024>>>(d_a, d_b, d_c);
  cudaDeviceSynchronize(); // 等待核函数执行完毕
  ```

#### **(2) 块内同步**
- **`__syncthreads()`**  
  同步同一Block内的所有线程（仅限核函数内部使用）。  
  ```c
  __global__ void kernel() {
      // ...
      __syncthreads(); // 确保所有线程执行到此点
  }
  ```

#### **(3) __syncthreads / __syncwarp对比**

| **特性**               | **`__syncthreads()`**          | **`__syncwarp()`**            |
|------------------------|--------------------------------|--------------------------------|
| **同步范围**           | 整个线程块（Block）            | 单个Warp（32线程）             |
| **粒度**               | 粗粒度（块级）                 | 细粒度（Warp级）               |
| **硬件依赖**           | 所有CUDA架构                   | Volta及更新架构（需计算能力≥7.0）|
| **性能开销**           | 较高（需等待所有线程）         | 较低（仅同步32线程）           |
| **适用场景**           | 块内全局共享内存操作           | Warp内协作（如Shuffle指令、Warp矩阵操作）|
| **线程掩码控制**       | 不支持                         | 支持（可指定部分线程同步）      |

```c
  __global__ void kernel() {
      __shared__ int s_data[1024];
      int tid = threadIdx.x;
      
      s_data[tid] = ...;   // 写入共享内存
      __syncthreads();     // 同步：确保所有线程完成写入
      ... = s_data[...];   // 安全读取其他线程写入的数据
  }
  __global__ void kernel() {
      int tid = threadIdx.x;
      int warp_id = tid / 32;
      int lane_id = tid % 32;

      // 仅同步当前Warp内的线程
      if (lane_id < 16) {
          ... // 某些操作
      }
      __syncwarp(); // 同步当前Warp的所有线程（无论是否参与分支）
  }
```


### **2.5 流管理（异步操作）**
- **`cudaStreamCreate(cudaStream_t* stream)`**  
  创建CUDA流，用于异步操作。  
  ```c
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  ```

- **`cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)`**  
  异步内存拷贝（需指定流）。  
  ```c
  cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
  ```

- **`cudaStreamSynchronize(cudaStream_t stream)`**  
  等待指定流中的操作完成。  
  ```c
  cudaStreamSynchronize(stream);
  ```

  ```
  cudaStream_t stream;       // CUDA streams are of type `cudaStream_t`.
  cudaStreamCreate(&stream); // Note that a pointer must be passed to `cudaCreateStream`.
  someKernel<<<number_of_blocks, threads_per_block, 0, stream>>>(); // `stream` is passed as 4th EC argument.
  cudaStreamDestroy(stream); // Note that a value, not a pointer, is passed to `cudaDestroyStream`.
  ```

### **2.6 数学函数（Device端）**
- **内置函数**：GPU核函数中可直接使用优化的数学函数，例如：  
  - `__sinf(x)`, `__cosf(x)`（快速单精度三角函数）
  - `__expf(x)`（指数函数）
  - `__logf(x)`（对数函数）
  - `atomicAdd(int* address, int val)`（原子加法，避免竞争条件）

### **2.7 错误处理**
- **`cudaGetLastError()`**  
  获取最近一次CUDA API调用的错误代码。  
  ```c
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
  }
  ```

- **`cudaGetErrorString(cudaError_t error)`**  
  将错误代码转换为可读字符串。


### **2.8 常用函数总结表**
| **类别**       | **函数**                     | **用途**                           |
|----------------|-----------------------------|-----------------------------------|
| 设备管理       | `cudaSetDevice`             | 选择GPU设备                       |
| 内存管理       | `cudaMalloc` / `cudaFree`   | 分配/释放设备内存                 |
| 数据拷贝       | `cudaMemcpy`                | 主机与设备间数据拷贝              |
| 核函数启动     | `<<<grid, block>>>`         | 启动核函数并配置线程结构          |
| 同步           | `cudaDeviceSynchronize`     | 等待设备完成所有任务              |
| 流管理         | `cudaStreamCreate`          | 创建异步操作流                    |
| 错误处理       | `cudaGetLastError`          | 检查CUDA API调用是否成功          |


### **2.9 示例：完整CUDA程序流程**
```c
#include <stdio.h>

// 核函数定义
__global__ void addKernel(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    const int N = 1024;
    float h_a[N], h_b[N], h_c[N];
    float *d_a, *d_b, *d_c;

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 分配设备内存
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // 数据拷贝到设备
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数
    addKernel<<<1, N>>>(d_a, d_b, d_c);

    // 拷贝结果回主机
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 验证结果
    printf("Result[0] = %f\n", h_c[0]); // 应输出3.0
    return 0;
}
```

### **2.10 注意事项**
1. **错误检查**：所有CUDA API调用后应检查返回值，避免静默失败。
2. **资源释放**：动态分配的设备内存和流必须显式释放。
3. **线程配置**：合理选择Block和Grid的大小以最大化GPU利用率（例如使用`cudaOccupancyCalculator`工具）。


## 3. Compiler

The host code is straight ANSI C code, which is compiled with the host’s standard C/C++ compilers and is run as a traditional CPU process.  
The device code, which is marked with CUDA keywords that designate CUDA kernels and their associated helper functions and data structures, is compiled by NVCC into virtual binary files called **PTX files**. Graphics driver translates PTX into executable binary code (**SASS**).

`nvcc`（NVIDIA CUDA Compiler）是CUDA编程的官方编译器，基于LLVM构建，支持将CUDA代码（`.cu`文件）编译为可在GPU和CPU上执行的程序。

### **3.1 基本编译流程**
#### **(1) 单文件编译**
将CUDA源码（`.cu`文件）直接编译为可执行文件：
```bash
nvcc example.cu -o example
```

#### **(2) 分步编译**
生成中间文件（如PTX、Cubin）：
```bash
nvcc -c example.cu        # 生成目标文件 example.o
nvcc example.o -o example # 链接生成可执行文件
```

### **3.2 常用编译选项**
#### **(1) 指定计算架构**
通过 `-arch` 或 `-gencode` 指定目标GPU的计算能力（Compute Capability）：
```bash
nvcc -arch=sm_75 example.cu -o example  # 针对Turing架构（如RTX 2080）
```
- **常见架构代号**：
  - `sm_35`（Kepler）、`sm_60`（Pascal）、`sm_70`（Volta）、`sm_80`（Ampere）、`sm_90`（Hopper）。
- **兼容性规则**：`-arch=sm_XX` 指定最低支持的架构，`-code=compute_XX` 生成PTX中间代码（兼容未来架构）。

#### **(2) 多架构兼容**
生成多版本代码以支持不同GPU：
```bash
nvcc -gencode arch=compute_60,code=sm_60 \
     -gencode arch=compute_80,code=sm_80 \
     example.cu -o example
```

#### **(3) 调试与优化**
- **调试信息**：添加 `-G` 生成调试符号：
  ```bash
  nvcc -G example.cu -o example  # 支持cuda-gdb调试
  ```
- **优化级别**：使用 `-O` 控制优化（如 `-O0` 禁用优化，`-O3` 最大优化）。

#### **(4) 头文件与库路径**
- **包含路径**：`-I<path>` 指定头文件目录。
- **库路径**：`-L<path>` 指定库目录，`-l<lib>` 链接库（如 `-lcudart`）。
```bash
nvcc -I/usr/local/cuda/include \
     -L/usr/local/cuda/lib64 \
     example.cu -o example -lcudart
```

#### **(5) 生成PTX/Cubin文件**
- **PTX（虚拟汇编）**：生成中间表示：
  ```bash
  nvcc --ptx example.cu -o example.ptx
  ```
- **Cubin（二进制）**：生成设备代码：
  ```bash
  nvcc --cubin -arch=sm_80 example.cu -o example.cubin
  ```

### **3.3 多文件编译**
#### **(1) 编译多个CUDA文件**
```bash
nvcc -c kernel.cu          # 生成kernel.o
nvcc -c main.cpp           # 生成main.o（C++文件）
nvcc kernel.o main.o -o program
```

#### **(2) 分离主机与设备代码**
- **仅编译设备代码**：
  ```bash
  nvcc -dc kernel.cu -o kernel.o  # 生成可链接的设备对象文件
  ```
- **链接所有对象**：
  ```bash
  nvcc -dlink kernel.o main.o -o program
  ```


### **3.4 常用环境变量**
- **CUDA路径**：`CUDA_PATH`（默认 `/usr/local/cuda`）。
- **GPU架构覆盖**：`CUDAARCHS`（覆盖默认架构，如 `export CUDAARCHS="80;90"`）。


### **3.5 示例：完整编译命令**
```bash
# 编译支持Ampere和Volta架构的调试版本
nvcc -arch=sm_80 -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_70,code=sm_70 \
     -I./include -L/usr/local/cuda/lib64 \
     -O3 -G \
     main.cu kernel.cu -o program -lcudart
```

### **3.6 常用命令速查表**
| **选项**                | **用途**                                      |
|-------------------------|----------------------------------------------|
| `-o <file>`             | 指定输出文件名                               |
| `-c`                    | 仅编译不链接（生成`.o`文件）                 |
| `-arch=sm_XX`           | 指定目标GPU架构（如 `sm_80`）                |
| `-G`                    | 生成调试信息                                 |
| `-O0` / `-O3`           | 优化级别（0禁用，3最大优化）                 |
| `--ptx`                 | 生成PTX中间代码                              |
| `-I<path>`              | 添加头文件搜索路径                           |
| `-L<path>`              | 添加库文件搜索路径                           |
| `-l<lib>`               | 链接库（如 `-lcudart` 链接CUDA运行时库）     |
| `-Xcompiler "<flag>"`   | 传递选项给主机编译器（如 `-Xcompiler "-fopenmp"`）|
| `--default-stream per-thread` | 启用每线程默认流（避免隐式同步）       |


### **3.7 工具相关**
安装NsightSystems: https://developer.nvidia.com/nsight-systems/get-started#platforms  
jupter配置nsys：https://pypi.org/project/jupyterlab-nvidia-nsight/

```
!nvcc -o 01-vector-add 01-vector-add.cu -run
!nsys nvprof ./01-vector-add
!rm report*
```
torch.autograd.profiler.profile  
torch.profiler.profile，采用prof.export_chrome_trace可导出[网页可视化](chrome://tracing/)json格式  

## 4. load_inline

在 PyTorch 中，`torch.utils.cpp_extension.load_inline` 是一个用于**动态内联编译 C++/CUDA 代码**的工具，允许用户直接在 Python 脚本中编写 C++ 或 CUDA 代码并即时编译成可调用的 PyTorch 扩展模块。

### **4.1 核心功能**
- **无需单独文件**：直接在 Python 中嵌入 C++/CUDA 代码。
- **自动编译**：调用时自动触发编译（通过 Ninja 或系统编译器）。
- **集成张量操作**：支持 `torch::Tensor` 类型，与 PyTorch 张量无缝交互。

### **4.2 基本用法**
**(1) 导入模块**
```python
import torch
from torch.utils.cpp_extension import load_inline
```

**(2) 编写内联代码**
定义 C++ 或 CUDA 代码为字符串，注意必须包含必要的头文件（如 `torch/extension.h`）。

```cpp
cpp_source = '''
#include <torch/extension.h>

// 定义一个C++函数
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

// 绑定到Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors", &add_tensors, "Add two tensors");
}
'''
```

**(3) 加载内联代码**
调用 `load_inline`，指定代码字符串和导出的函数名：
```python
# 加载C++扩展
cpp_extension = load_inline(
    name='cpp_extension',  # 扩展模块名称（任意）
    cpp_sources=cpp_source,
    functions=['add_tensors']  # 导出的函数名
)
```

**(4) 调用函数**
```python
a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
c = cpp_extension.add_tensors(a, b)
print(c)  # 输出 tensor([4., 6.])
```

### **4.3 内联 CUDA 代码示例**
**(1) 编写 CUDA 核函数**
```cpp
cuda_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA核函数：逐元素加法
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 包装函数，处理张量内存
torch::Tensor add_tensors_cuda(torch::Tensor a, torch::Tensor b) {
    // 检查输入合法性
    assert(a.device().is_cuda() && b.device().is_cuda());
    assert(a.sizes() == b.sizes());

    // 创建输出张量
    torch::Tensor c = torch::empty_like(a);

    // 获取张量数据指针
    float* a_ptr = a.data_ptr<float>();
    float* b_ptr = b.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();
    int n = a.numel();

    // 配置CUDA核函数参数
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // 启动核函数
    add_kernel<<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, n);

    return c;
}

// 绑定到Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors_cuda", &add_tensors_cuda, "Add two CUDA tensors");
}
'''
```

**(2) 加载 CUDA 扩展**
```python
cuda_extension = load_inline(
    name='cuda_extension',
    cpp_sources=[],  # 如果没有C++代码，留空列表
    cuda_sources=cuda_source,
    functions=['add_tensors_cuda'],
    with_cuda=True,  # 启用CUDA支持
    extra_cuda_cflags=['-O2']  # 附加编译选项
)
```

**(3) 调用 CUDA 函数**
```python
a = torch.tensor([1.0, 2.0], device='cuda')
b = torch.tensor([3.0, 4.0], device='cuda')
c = cuda_extension.add_tensors_cuda(a, b)
print(c.cpu())  # 输出 tensor([4., 6.])
```

### **4.4 关键参数说明**
| **参数**            | **用途**                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `name`              | 扩展模块名称（需唯一，避免重复加载冲突）                                |
| `cpp_sources`       | C++ 代码字符串或文件路径列表                                            |
| `cuda_sources`      | CUDA 代码字符串或文件路径列表                                           |
| `functions`         | 导出的函数名列表（需与代码中的 `m.def` 一致）                           |
| `with_cuda`         | 是否启用 CUDA 支持（默认自动检测）                                      |
| `extra_include_paths` | 附加头文件搜索路径（如自定义头文件目录）                                |
| `extra_cflags`      | 附加 C++ 编译选项（如 `-O3`、`-march=native`）                          |
| `extra_cuda_cflags` | 附加 CUDA 编译选项（如 `-Xcompiler -fopenmp`）                          |


### **4.5 常见问题**
**(1) 编译错误**
- **错误信息**：若编译失败，会抛出 `RuntimeError`，显示具体编译日志。
- **调试方法**：检查代码语法、头文件路径、CUDA 版本是否与 PyTorch 匹配。

**(2) 张量设备一致性**
- CUDA 函数要求输入张量位于 GPU 上，需提前调用 `.cuda()`：
  ```python
  a = a.cuda()
  ```

**(3) 性能优化**
- **合并内存访问**：CUDA 核函数中确保线程访问连续内存。
- **使用共享内存**：优化数据复用，减少全局内存访问。


### **4.6 完整示例：向量加法**
**C++ 版本**
```python
import torch
from torch.utils.cpp_extension import load_inline

cpp_code = '''
#include <torch/extension.h>

torch::Tensor add_vectors(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_vectors", &add_vectors, "Add two vectors");
}
'''

extension = load_inline(
    name='vector_add',
    cpp_sources=cpp_code,
    functions=['add_vectors']
)

a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
result = extension.add_vectors(a, b)
print(result)  # 输出 tensor([4., 6.])
```

**CUDA 版本**
```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_code = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor add_vectors_cuda(torch::Tensor a, torch::Tensor b) {
    torch::Tensor c = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_vectors_cuda", &add_vectors_cuda, "Add two vectors on GPU");
}
'''

extension = load_inline(
    name='vector_add_cuda',
    cuda_sources=cuda_code,
    functions=['add_vectors_cuda'],
    with_cuda=True
)

a = torch.tensor([1.0, 2.0], device='cuda')
b = torch.tensor([3.0, 4.0], device='cuda')
result = extension.add_vectors_cuda(a, b)
print(result.cpu())  # 输出 tensor([4., 6.])
```

### **4.7 总结**
- **适用场景**：快速验证自定义算子、小型项目原型开发、交互式环境（如 Jupyter）。
- **优势**：无需维护单独文件，简化编译流程。
- **局限性**：代码复杂时维护困难，建议大型项目使用 `setuptools` 编译独立扩展。
通过 `load_inline`，可以高效地将高性能 C++/CUDA 代码嵌入到 PyTorch 工作流中，灵活应对定制化计算需求。

## 5. triton

### 5.1 torch.compile

通过TORCH_LOGS="output_code"及torch.compile自动生成triton kernel

|        | CUDA | TRITION | 
| :--:   | :--: | :--:    |
| Memory Coalescing | Muaual | Automatic | 
| Shared Memory Management | Muaual | Automatic | 
| Scheduling (Within SMs) | Muaual | Automatic | 
| Scheduling (Across SMs) | Muaual | Muaual | 

### 5.2 Triton Kernel 基本语法
Triton内核是用Python编写的，通过`@triton.jit`装饰器标记，并利用Triton提供的张量操作接口（如`triton.language`模块）实现高效GPU计算。

#### **(1) 内核函数定义**
```python
import triton
import triton.language as tl

@triton.jit
def kernel_name(
    input_ptr,      # 输入张量指针
    output_ptr,     # 输出张量指针
    n_elements,     # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 编译时常量（如块大小）
    **其他参数      # 可选参数
):
    # 获取当前块的索引
    pid = tl.program_id(axis=0)
    # 计算当前块处理的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 掩码处理边界条件
    mask = offsets < n_elements
    # 从全局内存加载数据
    x = tl.load(input_ptr + offsets, mask=mask)
    # 执行计算
    result = x * 2
    # 将结果存回全局内存
    tl.store(output_ptr + offsets, result, mask=mask)
```

#### **(2) 关键组件**
- **`tl.program_id(axis)`**  
  获取当前块在多维网格中的索引（如`axis=0`表示第一维）。
  
- **`tl.arange(start, end)`**  
  生成连续的索引序列，用于计算线程处理的数据位置。

- **`tl.load` / `tl.store`**  
  安全地读写全局内存，通过`mask`参数处理边界条件。

- **`tl.constexpr`**  
  标记编译时常量（如`BLOCK_SIZE`），在编译时确定以优化性能。

### **5.3 Python 中调用 Triton Kernel**

#### **(1) 输入输出准备**
确保输入输出张量位于GPU（如使用PyTorch）：
```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
y = torch.empty_like(x)
```

#### **(2) 配置网格（Grid）和块（Block）**
- **网格函数**：动态计算所需的块数量。
- **块大小（BLOCK_SIZE）**：通常选择2的幂（如128、256、512）。

```python
def launch_kernel(x):
    n_elements = x.numel()
    # 定义网格大小（块数量）
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # 启动内核
    kernel_name[grid](x, y, n_elements, BLOCK_SIZE=128)
    return y
```

#### **(3) 完整示例：向量加法**
```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=256)
    return output

# 测试
x = torch.tensor([1.0, 2.0], device="cuda")
y = torch.tensor([3.0, 4.0], device="cuda")
result = add(x, y)
print(result)  # 输出 tensor([4., 6.], device='cuda:0')
```

### **5.4 高级功能**

#### **(1) 多维网格与块**
支持多维线程组织（如2D网格）：
```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # 2D块处理矩阵乘法
    ...

# 配置2D网格
grid = lambda meta: (M // meta["BLOCK_M"], N // meta["BLOCK_N"])
```

#### **(2) 共享内存与优化**
使用`tl.static`分配共享内存，优化数据复用：
```python
@triton.jit
def kernel_with_shared_memory(
    input_ptr, output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # 分配共享内存
    shmem = tl.static((BLOCK_SIZE,), dtype=tl.float32)
    pid = tl.program_id(axis=0)
    offsets = ...
    # 从全局内存加载到共享内存
    data = tl.load(input_ptr + offsets)
    shmem[offsets % BLOCK_SIZE] = data
    tl.barrier()  # 同步块内线程
    # 使用共享内存计算
    ...
```

#### **(3) 自动调优**
使用`triton.autotune`自动选择最优配置：
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def tuned_kernel(...):
    ...
```

### **5.5 关键注意事项**
1. **设备要求**：输入输出张量必须位于GPU。
2. **数据类型匹配**：Triton支持`tl.float16`、`tl.float32`、`tl.int32`等，需与输入张量类型一致。
3. **边界处理**：通过`mask`参数避免越界访问。
4. **性能调优**：选择合适的`BLOCK_SIZE`和`num_warps`（每个块的线程束数量）。

### **5.6 总结**
- **Triton优势**：用Python语法编写高性能GPU内核，无需深入CUDA细节。
- **适用场景**：深度学习算子优化、自定义数学运算、矩阵操作等。
- **调试工具**：使用`TORCH_COMPILE_DEBUG=1`环境变量查看编译日志。
通过结合Triton的简洁语法和PyTorch的生态，可以快速实现高效GPU计算任务。

## 6. numba

```
@cuda.jit
def square_matrix_kernel(matrix, result):
    # Calculate the row and column index for each thread
    row, col = cuda.grid(2)

    # Check if the thread's indices are within the bounds of the matrix
    if row < matrix.shape[0] and col < matrix.shape[1]:
        # Perform the square operation
        result[row, col] = matrix[row, col] ** 2
```

```
!python3 pytorch_square.py
!python3 hello_load_inline.py
!TORCH_LOGS="output_code" python3 pytorch_square_compiler.py
!python3 numba_square.py
```