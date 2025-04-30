## 1. CUDA编程内存类型

在 CUDA 编程中，合理使用不同类型的内存是优化性能的关键。以下是各类内存的具体使用方法、代码示例及适用场景：

### **1.1 全局内存（Global Memory）**
#### **使用方式**：
- **分配与释放**：
  ```c
  float *d_data;
  cudaMalloc(&d_data, size);        // 分配
  cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice); // 数据拷贝
  // 使用完毕后
  cudaFree(d_data);                 // 释放
  ```
- **核函数访问**：
  ```c
  __global__ void kernel(float *data) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      data[idx] = ...;  // 直接读写全局内存
  }
  ```
#### **适用场景**：
  - 存储输入/输出数据（如大型数组）。
  - 需要跨线程块共享的数据。
#### **优化技巧**：
  - **合并访问**：确保连续线程访问连续内存地址。
  - **使用内存预取**：异步拷贝（`cudaMemcpyAsync`）提升吞吐量。

### **1.2 共享内存（Shared Memory）**
#### **使用方式**：
- **静态声明**（在核函数内）：
  ```c
  __global__ void kernel() {
      __shared__ float s_data[1024];  // 静态共享内存
      s_data[threadIdx.x] = ...;
      __syncthreads();                // 同步块内线程
  }
  ```
- **动态声明**（核函数启动时指定大小）：
  ```c
  kernel<<<grid, block, shared_mem_size>>>(...);
  // 核函数内：
  extern __shared__ float s_data[];  // 动态共享内存
  ```
#### **适用场景**：
  - 缓存频繁访问的数据块（如矩阵乘法中的分块）。
  - 线程块内数据共享（如归约操作）。
#### **注意事项**：
  - 必须使用 `__syncthreads()` 确保数据同步。
  - 避免共享内存 bank 冲突（如访问同一 bank 的不同地址）。

### **1.3 常量内存（Constant Memory）**
#### **使用方式**：
- **声明与初始化**：
  ```c
  __constant__ float c_filter[64];  // 声明常量内存
  // 主机端拷贝数据到常量内存
  cudaMemcpyToSymbol(c_filter, h_filter, sizeof(float)*64);
  ```
- **核函数访问**：
  ```c
  __global__ void kernel() {
      float val = c_filter[threadIdx.x];  // 只读访问
  }
  ```
#### **适用场景**：
  - 存储只读常量（如滤波器系数、配置参数）。
  - 所有线程同时访问相同常量数据时效率最高。
#### **优化技巧**：
  - 常量内存通过缓存加速，适合小规模高频读取。

### **1.4 纹理内存（Texture Memory）**
#### **使用方式**：
- **声明与绑定**：
  ```c
  texture<float, 2> tex;  // 声明 2D 纹理
  cudaArray *d_array;
  cudaMallocArray(&d_array, &channelDesc, width, height);
  cudaMemcpyToArray(d_array, 0, 0, h_data, size, cudaMemcpyHostToDevice);
  cudaBindTextureToArray(tex, d_array);  // 绑定纹理
  ```
- **核函数访问**：
  ```c
  __global__ void kernel() {
      float val = tex2D(tex, x, y);  // 使用纹理读取
  }
  ```
- **解绑与释放**：
  ```c
  cudaUnbindTexture(tex);
  cudaFreeArray(d_array);
  ```
#### **适用场景**：
  - 图像处理（支持硬件插值）。
  - 非对齐访问或具有空间局部性的数据。
#### **优势**：
  - 自动缓存优化，减少全局内存访问延迟。

### **1.5 寄存器（Register）**
#### **使用方式**：
- **自动分配**：
  ```c
  __global__ void kernel() {
      float local_var = ...;  // 局部变量自动使用寄存器
  }
  ```
#### **适用场景**：
  - 存储线程私有临时变量。
#### **注意事项**：
  - 避免过多寄存器使用（通过编译选项 `-maxrregcount=N` 限制）。
  - 复杂操作（如循环、大数组）可能导致寄存器溢出到本地内存。


### **1.6 固定内存（Pinned Memory）**
#### **使用方式**：
- **分配与释放**：
  ```c
  float *h_pinned;
  cudaHostAlloc(&h_pinned, size, cudaHostAllocDefault);  // 分配
  // 使用完毕后
  cudaFreeHost(h_pinned);  // 释放
  ```
- **异步传输**：
  ```c
  cudaMemcpyAsync(d_data, h_pinned, size, cudaMemcpyHostToDevice, stream);
  ```
#### **适用场景**：
  - 加速主机与设备间数据传输（如流水线处理）。
  - 需要与设备异步交互的主机端数据。


### **1.7 统一内存（Unified Memory）**
#### **使用方式**：
- **分配与访问**：
  ```c
  float *u_data;
  cudaMallocManaged(&u_data, size);  // 分配统一内存
  // 主机和设备均可直接访问
  kernel<<<grid, block>>>(u_data);
  host_function(u_data);
  ```
#### **适用场景**：
  - 简化内存管理（无需显式拷贝）。
  - 数据在主机和设备间频繁交互但访问模式不固定的场景。
#### **注意事项**：
  - 统一内存的自动迁移可能引入性能开销，需谨慎用于高性能计算。


### **1.8 内存使用原则总结**

| 内存类型       | 作用域          | 速度     | 容量       | 使用场景                     |
|----------------|-----------------|----------|------------|------------------------------|
| **全局内存**   | 全局            | 慢       | 大         | 主数据存储                   |
| **共享内存**   | 线程块          | 快       | 小         | 块内缓存和通信               |
| **常量内存**   | 全局（只读）    | 较快     | 极小       | 频繁读取的常量               |
| **纹理内存**   | 全局（只读）    | 较快     | 小         | 空间局部性数据（如图像）     |
| **寄存器**     | 线程私有        | 最快     | 极小       | 局部变量                     |
| **本地内存**   | 线程私有        | 慢       | 大         | 寄存器溢出时的备用           |
| **固定内存**   | 主机端          | 传输快   | 大         | 加速主机-设备数据传输        |
| **统一内存**   | 主机+设备       | 中等     | 大         | 简化内存管理                 |


### **1.9 示例：矩阵乘法优化（共享内存）**
```c
__global__ void matmul(float *A, float *B, float *C, int N) {
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE + ty;
    int col = bx * TILE + tx;
    float sum = 0;

    for (int i = 0; i < N/TILE; ++i) {
        // 从全局内存加载分块到共享内存
        s_A[ty][tx] = A[row * N + i * TILE + tx];
        s_B[ty][tx] = B[(i * TILE + ty) * N + col];
        __syncthreads();

        // 计算分块乘积
        for (int k = 0; k < TILE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```
**优化点**：
- 使用共享内存缓存分块数据，减少全局内存访问次数。
- 通过 `TILE` 大小平衡共享内存使用和线程块配置。

通过合理选择内存类型和优化访问模式，可显著提升 CUDA 程序的性能。建议结合 NVIDIA Nsight 工具分析内存访问模式，针对性优化瓶颈。







