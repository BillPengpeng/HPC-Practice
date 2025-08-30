本文主要整理PMPP Chapter 2 Exercise。


## Q1

1. If we want to use each thread in a grid to calculate one output element of a
 vector addition, what would be the expression for mapping the thread/block
 indices to the data index (i)?
 - (A) i=threadIdx.x + threadIdx.y;
 - (B) i=blockIdx.x + threadIdx.x;
 - (C) i=blockIdx.x * blockDim.x + threadIdx.x;
 - (D) i=blockIdx.x * threadIdx.x;
 - Answer: C

## Q2

2. Assume that we want to use each thread to calculate two adjacent elements of
 a vector addition. What would be the expression for mapping the thread/block
 indices to the data index (i) of the first element to be processed by a thread?
 - (A) i=blockIdx.x * blockDim.x + threadIdx.x + 2;
 - (B) i=blockIdx.x * threadIdx.x2;
 - (C) i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;
 - (D) i=blockIdx.x * blockDim.x * 2 + threadIdx.x;
 - Answer: C

## Q3 

3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes 2*blockDim.x consecutive elements that form
 two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for
 mapping the thread/block indices to data index of the first element?
 - (A) i=blockIdx.x * blockDim.x + threadIdx.x + 2;
 - (B) i=blockIdx.x * threadIdx.x * 2;
 - (C) i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;
 - (D) i=blockIdx.x * blockDim.x * 2 + threadIdx.x;
 - Answer: D

 根据问题描述，每个线程需要计算向量加法中的两个元素，且线程块的处理模式为：**所有线程先共同处理第一个连续数据段（每个线程处理一个元素），再共同处理第二个连续数据段**。

### **核心逻辑分析**
1. **数据分段**  
   每个线程块处理 `2 * blockDim.x` 个连续元素，分为两个等长的数据段：  
   - **第一段**：索引范围 `[base, base + blockDim.x - 1]`  
   - **第二段**：索引范围 `[base + blockDim.x, base + 2*blockDim.x - 1]`  
   （`base` 是当前线程块的起始索引）

2. **线程分工**  
   - 所有线程**先处理第一段**：线程 `threadIdx.x` 处理元素 `base + threadIdx.x`  
   - 所有线程**再处理第二段**：线程 `threadIdx.x` 处理元素 `base + blockDim.x + threadIdx.x`

3. **起始索引 `i` 的定义**  
   `i` 是线程处理的**第一个元素**的索引（即第一段中的位置）。

---

### **索引表达式推导**
- **步骤1：计算线程块的起始索引 `base`**  
  每个线程块负责 `2 * blockDim.x` 个元素，因此：  
  ```  
  base = blockIdx.x * (2 * blockDim.x)  
  ```  
  （`blockIdx.x` 是线程块在网格中的索引）

- **步骤2：计算线程在第一段中的位置**  
  线程 `threadIdx.x` 在第一段中处理的元素索引为：  
  ```  
  i = base + threadIdx.x  
  ```  

- **最终表达式**  
  代入 `base` 的表达式：  
  ```  
  i = blockIdx.x * (2 * blockDim.x) + threadIdx.x  
  ```  

---

### **完整示例验证**
假设：  
- `blockDim.x = 256`（每块256线程）  
- `blockIdx.x = 0`（第一个线程块）  
- `threadIdx.x = 10`（第10号线程）  

**计算结果**：  
```
i = 0 * (2 * 256) + 10 = 10
```
- **第一段元素**：索引 `10`（正确）  
- **第二段元素**：索引 `10 + 256 = 266`（符合分段逻辑）  

---

### **结论**
线程的第一个元素索引 `i` 的表达式为：  
```c
i = blockIdx.x * (2 * blockDim.x) + threadIdx.x
```  
此公式确保：  
1. 每个线程块处理连续的两段数据（每段 `blockDim.x` 个元素）  
2. 所有线程先并行处理第一段，再并行处理第二段  
3. 线程间无冲突访问，且覆盖所有数据

## Q4

4. For a vector addition, assume that the vector length is 8000, each thread
 calculates one output element, and the thread block size is 1024 threads. The
 programmer configures the kernel call to have a minimum number of thread
 blocks to cover all output elements. How many threads will be in the grid?
 - (A) 8000
 - (B) 8196
 - (C) 8192
 - (D) 8200
 - Answer: C

## Q5

5. If we want to allocate an array of v integer elements in the CUDA device
 global memory, what would be an appropriate expression for the second
 argument of the cudaMalloc call?
 - (A) n
 - (B) v
 - (C) n * sizeof(int)
 - (D) v * sizeof(int)
 - Answer: D

## Q6

6. If we want to allocate an array of n floating-point elements and have a
 floating-point pointer variable A_d to point to the allocated memory, what
 would be an appropriate expression for the first argument of the cudaMalloc
 () call?
 - (A) n
 - (B) (void *) A_d
 - (C) *A_d
 - (D) (void **)&A_d
 - Answer: D

## Q7

7. If we want to copy 3000 bytes of data from host array A_h (A_h is a pointer
 to element 0 of the source array) to device array A_d (A_d is a pointer to
 element 0 of the destination array), what would be an appropriate API call
 for this data copy in CUDA?
 - (A) cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);
 - (B) cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceTHost);
 - (C) cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);
 - (D) cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);
 - Answer: C

## Q8

8. How would one declare a variable err that can appropriately receive the
returned value of a CUDA API call?
 - (A) int err;
 - (B) cudaError err;
 - (C) cudaError_t err;
 - (D) cudaSuccess_t err;
 - Answer: C

`cudaError_t` 是 CUDA 编程中用于**错误处理的核心枚举类型**，所有 CUDA Runtime API 函数均返回此类型值。其基本使用流程如下：

---

### **核心使用步骤**
#### 1. **检查 API 调用返回值**
```c
cudaError_t err = cudaMalloc(&d_A, size);  // 示例：设备内存分配
if (err != cudaSuccess) {                  // 判断是否成功
    // 错误处理
}
```

#### 2. **获取错误描述信息**
```c
printf("Error: %s\n", cudaGetErrorString(err)); 
// 输出示例：Error: invalid argument
```

#### 3. **内核启动错误需同步检测**
```c
myKernel<<<blocks, threads>>>(...);  // 异步启动内核
cudaError_t err = cudaGetLastError(); // 捕获启动错误（如配置错误）
if (err != cudaSuccess) { ... }

err = cudaDeviceSynchronize();       // 同步等待内核完成
if (err != cudaSuccess) {             // 捕获内核执行错误（如越界）
    printf("Kernel failed: %s\n", cudaGetErrorString(err));
}
```

---

### **关键错误类型**
| 错误码 | 宏定义 | 含义 |
|--------|--------|------|
| 0 | `cudaSuccess` | 操作成功 |
| 1 | `cudaErrorInvalidValue` | 非法参数值 |
| 2 | `cudaErrorMemoryAllocation` | 设备内存分配失败 |
| 7 | `cudaErrorLaunchTimeout` | 内核执行超时 |
| 30 | `cudaErrorUnknown` | 未知错误 |

> 完整列表见 https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html

---

### **最佳实践示例**
```c
// 1. 设备内存分配检查
float* d_data;
cudaError_t err = cudaMalloc(&d_data, sizeof(float)*N);
if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

// 2. 内核启动与执行检查
vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
err = cudaGetLastError(); // 检查启动错误
if (err != cudaSuccess) {
    fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
}

err = cudaDeviceSynchronize(); // 检查内核运行时错误
if (err != cudaSuccess) {
    fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
}
```

---

### **重要说明**
1. **异步性**：内核启动是异步的，需 `cudaDeviceSynchronize()` 后检查运行时错误
2. **错误传播**：一个 API 调用失败后，后续 API 可能返回 `cudaErrorUnknown`
3. **调试工具**：结合 `cuda-gdb` 或 `Nsight Systems` 定位错误上下文
4. **返回值忽略风险**：**必须检查**每次 API 调用，否则可能导致静默失败

> 建议封装检查函数：
> ```c
> #define CHECK_CUDA_ERROR(call) { \
>     cudaError_t err = call; \
>     if (err != cudaSuccess) { \
>         fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
>         exit(EXIT_FAILURE); \
>     } \
> }
> 
> // 使用示例
> CHECK_CUDA_ERROR( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
> ```

## Q9

9. Consider the following CUDA kernel and the corresponding host function
 that calls it:
 01 __global__ void foo_kernel(float a, float b, unsigned int N){
 02    unsigned int i=blockIdx.xblockDim.x + threadIdx.x;
 03    if(i < N) {
 04        b[i]=2.7fa[i]- 4.3f;
 05    }
 06 }
 07 void foo(float a_d, float b_d) {
 08    unsigned int N=200000;
 09    foo_kernel <<<(N + 128 - 1)/128, 128>>>(a_d, b_d, N);
 10 }
 - a. What is the number of threads per block?  
 - b. What is the number of threads in the grid?
 - c. What is the number of blocks in the grid?
 - d. What is the number of threads that execute the code on line 02?
 - e. What is the number of threads that execute the code on line 04?

 - Answer: a. 128; b. 200192; c. 1564; d. 200192; e. 200000

## Q10

10. A new summer intern was frustrated with CUDA. He has been complaining
 that CUDA is very tedious. He had to declare many functions that he plans
 to execute on both the host and the device twice, once as a host function and
 once as a device function. What is your response?

CUDA 实际上提供了一种更高效的方式来处理这种情况，而不需要您为每个函数声明两次。您可以使用 `__host__ __device__` 关键字组合来声明函数，这样编译器会自动为主机（CPU）和设备（GPU）生成两个版本的函数代码。这意味着您只需要编写一次函数定义，CUDA 工具链会处理剩下的工作。

### 示例：
 instead of doing this:
```cpp
// 繁琐的方式：单独声明两次
__host__ float myFunction(float x) {
    return x * x;
}

__device__ float myFunction_device(float x) {
    return x * x;
}
```

您可以直接这样写：
```cpp
// 高效的方式：使用 __host__ __device__ 组合
__host__ __device__ float myFunction(float x) {
    return x * x;
}
```
编译器会为主机和设备分别编译这个函数，因此您可以在主机代码和设备代码中调用相同的 `myFunction`，而不需要维护两个版本。

### 为什么这样做？
- **减少代码重复**：您只需要编写一次函数逻辑，避免了重复代码，从而降低了错误和维护成本。
- **保持一致性**：主机和设备版本的行为完全相同，因为它们是来自同一源代码的编译结果。
- **简化开发**：这对于数学函数、工具函数或任何需要在主机和设备上共享的逻辑特别有用。

### 注意事项：
- 这种方式适用于大多数情况，但如果函数涉及主机或设备特定的特性（如主机端的文件 I/O 或设备端的共享内存访问），则需要单独处理。但对于纯计算函数，这通常工作得很好。
- 确保您的 CUDA 开发环境支持此功能（所有现代 CUDA 版本都支持）。

