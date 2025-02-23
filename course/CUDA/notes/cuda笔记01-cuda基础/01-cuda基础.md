## cuda基本函数

### 限定词

| Qualifier keyword  | Callable From | Executed on| Executed by | 
| :--:   | :--:    | :--:      | :--:    | 
| \_\_host\_\_ | Host | Host | Caller host thread | 
| \_\_global\_\_ | Host | Device | New grid of device threads | 
| \_\_device\_\_ | Device | Device | Caller device thread | 

### 内存相关

设备分配内存：cudaMalloc  
CPU分配内存：cudaMallocHost  
分配统一内存，CPU/GPU均可访问 ：cudaMallocManaged  
CPU/GPU内存拷贝（复制方向包括：cudaMemcpyHostToDevice/cudaMemcpyDeviceToHost）：cudaMemcpy  
数据异步预取  cudaMemPrefetchAsync

```
void foo(cudaStream_t s) {
  char *data;
  cudaMallocManaged(&data, N);
  init_data(data, N);                                   // execute on CPU
  cudaMemPrefetchAsync(data, N, myGpuId, s);            // prefetch to GPU
  mykernel<<<..., s>>>(data, N, 1, compare);            // execute on GPU
  cudaMemPrefetchAsync(data, N, cudaCpuDeviceId, s);    // prefetch to CPU
  cudaStreamSynchronize(s);
  use_data(data, N);
  cudaFree(data);
}
```

### CUDA stream

```
cudaStream_t stream;       // CUDA streams are of type `cudaStream_t`.
cudaStreamCreate(&stream); // Note that a pointer must be passed to `cudaCreateStream`.
someKernel<<<number_of_blocks, threads_per_block, 0, stream>>>(); // `stream` is passed as 4th EC argument.
cudaStreamDestroy(stream); // Note that a value, not a pointer, is passed to `cudaDestroyStream`.
```

### Compiler

The host code is straight ANSI C code, which is compiled with the host’s standard C/C++ compilers and is run as a traditional CPU process.  
The device code, which is marked with CUDA keywords that designate CUDA kernels and their associated helper functions and data structures, is compiled by NVCC into virtual binary files called **PTX files**. Graphics driver translates PTX into executable binary code (**SASS**).

### Architecture of a modern GPU

A typical CUDA-capable GPU is organized into an array of highly threaded **streaming multiprocessors (SMs)**. Each SM has several processing units called **streaming processors or CUDA cores**. **Multiple blocks** are likely to be simultaneously assigned to the same SM. However, blocks need to reserve hardware resources to execute, so only a limited number of blocks can be simultaneously assigned to a given SM.  

In most implementations to date, once a block has been assigned to an SM, it is further divided into **32-thread units called warps**. The size of warps is implementation specific and can vary in future generations of GPUs. (threadIdx.x, threadIdx.y, threadIdx.z) 组织warps时， **优先级 x > y > z， 参考cuda mode Lecture 4**。 When threads in the same warp follow different execution paths, we say that these threads exhibit **control/warp divergence**, that is, they diverge in their execution. If all threads in a warp must complete a phase of their execution before any of them can move on, one must use a **barrier synchronization mechanism such as __syncwarp()** to ensure correctness.

### Getting good occupancy – balance resources

Have 82 SM → **many blocks = good** (for comparison Jetson Xavier has 8 Volta SM).  
Can schedule up to 1536 threads per SM → power of two **block size < 512** desirable (some other GPUs 2048).  
**Avoid divergence** to execute an entire warp (32 threads) at each cycle.  
**Avoid FP64/INT64** if you can on Gx102 (GeForce / Workstation GPUs).  
Shared Memory and Register File → **limits number of scheduled on SM**. (use __launch_bounds__ / C10_LAUNCH_BOUNDS to advise compiler of # of threads for register allocation, but register spill makes things slow).  
Use torch.cuda.get_device_properties(<gpu_num>) to get properties (e.g. max_threads_per_multi_processor)

### 工具相关
安装NsightSystems: https://developer.nvidia.com/nsight-systems/get-started#platforms  
jupter配置nsys：https://pypi.org/project/jupyterlab-nvidia-nsight/


```
!nvcc -o 01-vector-add 01-vector-add.cu -run
!nsys nvprof ./01-vector-add
!rm report*
```

## python集成cuda基础

### profiler

torch.autograd.profiler.profile  
torch.profiler.profile，采用prof.export_chrome_trace可导出[网页可视化](chrome://tracing/)json格式  

### Custom cpp extensions

#### 方式一：load_inline

```
cpp_source = """
std::string hello_world() {
  return "Hello World!";
}
"""

my_module = load_inline(
    name='my_module',
    cpp_sources=[cpp_source],
    functions=['hello_world'],
    verbose=True,
    build_directory='./tmp'
)
```

#### 方式二：Integrate a triton kernel

通过TORCH_LOGS="output_code"及torch.compile自动生成triton kernel

|        | CUDA | TRITION | 
| :--:   | :--: | :--:    |
| Memory Coalescing | Muaual | Automatic | 
| Shared Memory Management | Muaual | Automatic | 
| Scheduling (Within SMs) | Muaual | Automatic | 
| Scheduling (Across SMs) | Muaual | Muaual | 

#### 方式三：numba

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