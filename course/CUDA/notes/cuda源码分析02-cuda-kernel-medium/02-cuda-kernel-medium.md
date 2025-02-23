## block all reduce

实现基于Warp Shuffle Functions，进行一些warp内数据交换。参考博文【CUDA编程】束内洗牌函数（Warp Shuffle Functions）[link](https://zhuanlan.zhihu.com/p/669957986)。  
__shfl_sync()：从索引通道直接复制。  
__shfl_up_sync()：从相对于调用者 ID 较低的通道复制。  
__shfl_down_sync()：从相对于调用者 ID 较高的通道复制。  
__shfl_xor_sync()：基于自身通道 ID 的按位异或（XOR）从通道复制。  

### block_all_reduce_sum_f32_f32_kernel

block_all_reduce_sum_f32_f32_kernel输入输出均为f32，累计两次warp reduce累加、一次block累加，首先，执行warps内32线程reduce，累加和统计至共享内存;然后，执行warps内前NUM_WARPS线程reduce，累加和汇总至每个block线程0；最后，所有block线程0结果累加至sum。  
block_all_reduce_sum_f16_f16_kernel输入为f16，输出为f32，第一次warp reduce累加采用f16，第二次warp reduce累加采用f32。  
block_all_reduce_sum_f16_f32_kernel输入为f16，输出为f32，两次warp reduce累加均采用f32。  

```
// -------------------------------------- FP32 -------------------------------------- 
// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Block All Reduce Sum
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // keep the data in register is enougth for warp operaion.
  float sum = (idx < N) ? a[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = sum;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}
```

### block_all_reduce_sum_bf16_bf16_kernel

__nv_bfloat16：数值范围大，与float32相似，可以避免在训练大模型时出现的梯度溢出或下溢问题；精度较低，但足够处理神经网络中的权重更新、激活值等不太敏感的计算。  
half：数值范围较小，可能不适合处理极大或极小的数值；精度较高，适用于需要更高精度但数值范围受控的场景。  

block_all_reduce_sum_bf16_bf16_kernel输入__nv_bfloat16，输出f32，两次warp reduce累加均采用__nv_bfloat16。  
block_all_reduce_sum_bf16_f32_kernel输入__nv_bfloat16，输出f32，两次warp reduce累加均采用f32。  

```
// -------------------------------------- BF16 -------------------------------------- 
// Warp Reduce Sum: Half
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_sum_bf16_bf16(
  __nv_bfloat16 val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Block All Reduce Sum: BF16
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_bf16_kernel(
  __nv_bfloat16* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];

  // keep the data in register is enougth for warp operaion.
  __nv_bfloat16 sum_bf16 = (idx < N) ? a[idx] : __float2bfloat16(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_bf16;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  __nv_bfloat16 sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2bfloat16(0.0f);
  if (warp == 0) sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, __bfloat162float(sum));
}
```

### block_all_reduce_sum_fp8_e4m3_f16_kernel

__nv_fp8_storage_t 是一个与 NVIDIA 的 FP8 格式相关的数据类型。FP8 通常采用E4M3（4位指数，3位尾数）或E5M2（5位指数，2位尾数）的表示方式，具备与FP16相当的动态范围，但在精度上有所降低。

block_all_reduce_sum_fp8_e4m3_f16_kernel输入__nv_fp8_storage_t，输出f32，第一次warp reduce采用warp_reduce_sum_fp8_e4m3_f16，结果按half精度保存；第二次warp reduce采用warp_reduce_sum_f16_f16。  
block_all_reduce_sum_fp8_e5m2_f16_kernel区别是E4M3调整为E5M2。  

```
// -------------------------------------- FP8 -------------------------------------- 
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp8_e4m3_f16(
  __nv_fp8_storage_t val) {
  // typedef unsigned char __nv_fp8_storage_t;
  // __half &operator=(const __half_raw &hr);
  half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E4M3);
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
  }
  return val_f16;
}
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e4m3_f16_kernel(
  __nv_fp8_storage_t* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ half reduce_smem[NUM_WARPS];

  // keep the data in register is enougth for warp operaion.
  __nv_fp8_storage_t sum_f8 = (idx < N) ? a[idx] : __nv_cvt_float_to_fp8(
    0.0f, __NV_SATFINITE, __NV_E4M3);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  half sum_f16 = warp_reduce_sum_fp8_e4m3_f16<WARP_SIZE>(sum_f8);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp16 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_f16;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  half sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
  if (warp == 0) sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, __half2float(sum));
}
```

### block_all_reduce_sum_i8_i32_kernel

block_all_reduce_sum_i8_i32_kernel输入int8_t，输出int32_t，第一次warp reduce采用warp_reduce_sum_i8_i32，结果按int32_t精度保存；第二次warp reduce采用warp_reduce_sum_i32_i32。   

```
// -------------------------------------- INT8 -------------------------------------- 
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i8_i32(int8_t val) {
  int32_t val_i32 = static_cast<int32_t>(val);
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_i32 += __shfl_xor_sync(0xffffffff, val_i32, mask);
  }
  return val_i32;
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_i8_i32_kernel(
  int8_t* a, int32_t* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ int32_t reduce_smem[NUM_WARPS];

  // keep the data in register is enougth for warp operaion.
  int8_t sum_i8 = (idx < N) ? a[idx] : 0;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  int32_t sum_i32 = warp_reduce_sum_i8_i32<WARP_SIZE>(sum_i8);
  if (lane == 0) reduce_smem[warp] = sum_i32;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  int32_t sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0;
  if (warp == 0) sum = warp_reduce_sum_i32_i32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}
```

## softmax

### 常规softmax

**__threadfence是一个内存同步指令**，主要用于确保线程对共享内存或全局内存的写操作已经完成，并且这些操作的结果对特定的线程集合是可见的。__syncthreads是一个线程同步指令，用于确保线程块中的所有线程都达到同一个同步点，然后才能继续执行。

```
// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// grid 1D block 1D, grid(N/256), block(256)
template<const int NUM_THREADS=256>
__device__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  float value = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  value = warp_reduce_sum_f32<NUM_WARPS>(value);  
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

// Softmax x: N, y: N
// grid(N/256), block(K=256)
template<const int NUM_THREADS = 256>
__global__ void softmax_f32_kernel(float* x, float* y, float* total, int N) {
  
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, exp_sum);
  __threadfence(); // grid level memory fence
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  // printf("N: %d, idx: %d, bid: %d, tid: %d, exp_val: %f, exp_sum: %f, total: %f\n", 
  //         N,     idx, blockIdx.x,  tid,     exp_val,     exp_sum,     *total);
  if (idx < N) y[idx] = exp_val / (*total); 
}
```

### softmax per-token

softmax per-token特点是**one token per thread block**，不需要做block间内存同步。

```
// NOTE: softmax per-token
// Softmax x: (S,h), y: (S,h)
// grid(S*h/h), block(h), assume h<=1024
// one token per thread block, only support 64<=h<=1024 and 2^n
// HEAD_SIZE/KV_LEN=NUM_THREADS
template<const int NUM_THREADS = 256>
__global__ void softmax_f32_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  // printf("N: %d, idx: %d, tid: %d, exp_val: %f, exp_sum: %f\n", 
  //         N, idx, tid, exp_val, exp_sum);
  if (idx < N) y[idx] = exp_val / exp_sum;
}
```

### safe_softmax per token

假设输入向量是 $\mathbf{z} = [z_1, z_2, \ldots, z_n]$，`safe_softmax` 的计算公式如下：

1. **找到输入向量的最大值**：$M = \max(z_1, z_2, \ldots, z_n)$

2. **从每个元素中减去最大值**：$\mathbf{z}_{\text{shifted}} = [z_1 - M, z_2 - M, \ldots, z_n - M]$

3. **计算改进的 softmax**：
   $$
   \text{safe\_softmax}(\mathbf{z})_i = \frac{e^{z_{\text{shifted}, i}}}{\sum_{j=1}^{n} e^{z_{\text{shifted}, j}}} \quad \text{for} \quad i = 1, 2, \ldots, n
   $$

```
// safe_softmax per token
template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float val = (idx < N) ? x[idx] : -FLT_MAX;
  float max_val = block_reduce_max_f32<NUM_THREADS>(val); // block max
  float exp_val = (idx < N) ? expf(x[idx] - max_val) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) y[idx] = exp_val / exp_sum; 
}
```

### safe_softmax_f16_f32

safe_softmax_f16_f32输入输出采用f16精度，中间过程计算采用f32精度。

__float2half_rd Converts float number to half precision in round-down mode，向下取整；__float2half_rn Converts float number to half precision in round-to-nearest-even mode，最近邻取整。[link](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html).

```
template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f16_f32_per_token_kernel(half* x, half* y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float val = (idx < N) ? __half2float(x[idx]) : -FLT_MAX;
  float max_val = block_reduce_max_f32<NUM_THREADS>(val); // block max
  float exp_val = (idx < N) ? expf(val - max_val) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) y[idx] = __float2half_rn(exp_val / exp_sum); 
}
```

### online_safe_softmax_f32

Safe Online Softmax 是一种逐步计算 Softmax 的方法，适用于流式数据或无法一次性获取全部输入的场景。其核心是通过动态维护最大值和归一化项，确保数值稳定性，避免指数运算的溢出问题。公式推导如下：

设输入序列为 $( x_1, x_2, \dots, x_N)$，逐步处理每个元素。维护两个状态变量：
- $m_k$: 前k个元素中的最大值。
- $s_k$: 前k个元素的指数和（基于当前最大值$m_k$进行归一化）。

**初始化**：  
$$m_0 = -\infty, s_0 = 0$$

**迭代更新**（对于第 \( k \) 个元素 \( x_k \)）：  
1. **更新最大值**：  
   $$m_k = \max(m_{k-1}, x_k)$$

2. **更新指数和**：  
   若 \( x_k > m_{k-1} \):  
   $$
   s_k = e^{m_{k-1} - m_k} \cdot s_{k-1} + e^{x_k - m_k}
   $$  
   否则：  
   $$
   s_k = s_{k-1} + e^{x_k - m_k}
   $$

**最终 Softmax**（对任意 \( x_i \)）：  
$$
\text{Softmax}(x_i) = \frac{e^{x_i - m_N}}{s_N}
$$

```
// -------------------------------------- FP32 -------------------------------------- 
// DS required for Online Softmax
struct __align__(8) MD { float m; float d; }; 
// Warp Reduce for Online Softmax
template<const int kWarpSize = WARP_SIZE >
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  unsigned int mask = 0xffffffff;
  #pragma unroll
  for(int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
    MD other;
    other.m = __shfl_xor_sync(mask, value.m, stride);
    other.d = __shfl_xor_sync(mask, value.d, stride);

    bool value_bigger = (value.m > other.m);
    MD bigger_m = value_bigger ? value : other;
    MD smaller_m = value_bigger ? other : value;
    
    value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    value.m = bigger_m.m;
  }
  return value;
}

template<const int NUM_THREADS = 256 >
__global__ void online_safe_softmax_f32_per_token_kernel(const float* x, float* y, int N) {
  // reference: https://arxiv.org/pdf/1805.02867 (Online normalizer calculation for softmax)
  int local_tid = threadIdx.x;
  int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
  const int WAPR_NUM = NUM_THREADS / WARP_SIZE;
  int warp_id = local_tid / WARP_SIZE;
  int lane_id = local_tid % WARP_SIZE;
  MD val;
  val.m = global_tid < N ? x[global_tid] : -FLT_MAX;
  val.d = global_tid < N ? 1.0f : 0.0f;

  __shared__ MD shared[WAPR_NUM]; 
  MD res = warp_reduce_md_op<WARP_SIZE>(val);

  if (lane_id == 0) shared[warp_id] = res; 
  __syncthreads();

  if (local_tid < WARP_SIZE) {
    MD block_res = shared[local_tid];
    block_res = warp_reduce_md_op<WAPR_NUM>(block_res); 
    if (local_tid == 0) {
      shared[0] = block_res; 
    }
  }
  __syncthreads();

  MD final_res = shared[0];
  float d_total_inverse = __fdividef(1.0f, final_res.d);
  if (global_tid < N) {
    y[global_tid] = __expf(x[global_tid] - final_res.m) * d_total_inverse;
  }
}
```

## Dot Product

### dot_prod_f32_f32_kernel

累计两次warp reduce累加、一次block累加，首先，执行warps内32线程reduce，累加和统计至共享内存;然后，执行warps内前NUM_WARPS线程reduce，累加和汇总至每个block线程0；最后，所有block线程0结果累加至y。

```
// -------------------------------------- FP32 -------------------------------------- 
// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Dot Product
// grid(N/256), block(256)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template<const int NUM_THREADS = 256>
__global__ void dot_prod_f32_f32_kernel(float* a, float* b, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  // keep the data in register is enougth for warp operaion.
  float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = prod;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0) atomicAdd(y, prod);
}
```

## Layer Norm

### layer_norm_f32_kernel

累计两次warp reduce累加，分别用于均值、标准差的计算。

```
// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/256), block(256)
template<const int NUM_THREADS=256>
__device__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  val = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum_f32<NUM_WARPS>(val);
  return val;
}

// Layer Norm: x: NxK(K=256<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template<const int NUM_THREADS=256>
__global__ void layer_norm_f32_kernel(float* x, float* y, float g, float b, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_mean; // shared within block
  __shared__ float s_variance; // shared within block
  float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float) K;
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  float variance = (value - s_mean) * (value - s_mean);
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / ((float) K + epsilon));
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  if (idx < N * K) y[idx] = ((value - s_mean) * s_variance) * g + b;
}
```

## RMS Norm

### rms_norm_f32_kernel

RMS Norm采用一次reduce sum计算标准差，再通过标准差计算归一化结果。

```
// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/256), block(256)
template<const int NUM_THREADS=256>
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  val = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum_f32<NUM_WARPS>(val);
  return val;
}

// RMS Norm: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template<const int NUM_THREADS=256>
__global__ void rms_norm_f32_kernel(float* x, float* y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_variance; // shared within block
  float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
  float variance = value * value;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads(); 
  if (idx < N * K) y[idx] = (value * s_variance) * g;
}
```

## NMS

非极大值抑制（Non-Maximum Suppression, NMS）是目标检测中用于去除冗余检测框的关键后处理步骤。其核心是通过计算交并比（IoU）筛选出置信度最高且不重叠的候选框。在CUDA加速中，主要优化思路是将计算密集型任务（如**IoU计算**和并行排序）迁移到GPU，利用其高并行性显著提升处理速度。

```
__global__ void nms_kernel(const float *boxes, const float *scores, int *keep, int num_boxes, float iou_threshold) {
  const int threadsPerBlock = blockDim.x;
  const int threadId = threadIdx.x;
  const int blockId = blockIdx.x;
  const int idx = blockId * threadsPerBlock + threadId;

  if (idx >= num_boxes)
    return;

  float x1 = boxes[idx * 4 + 0];
  float y1 = boxes[idx * 4 + 1];
  float x2 = boxes[idx * 4 + 2];
  float y2 = boxes[idx * 4 + 3];
  int suppressed = 0;

  for (int i = 0; i < idx; ++i) {
    if (keep[i] == 0)
      continue;

    float x1_i = boxes[i * 4 + 0];
    float y1_i = boxes[i * 4 + 1];
    float x2_i = boxes[i * 4 + 2];
    float y2_i = boxes[i * 4 + 3];

    float inter_x1 = max(x1, x1_i);
    float inter_y1 = max(y1, y1_i);
    float inter_x2 = min(x2, x2_i);
    float inter_y2 = min(y2, y2_i);
    float inter_w = max(0.0f, inter_x2 - inter_x1);
    float inter_h = max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area = (x2 - x1) * (y2 - y1);
    float area_i = (x2_i - x1_i) * (y2_i - y1_i);
    float iou = inter_area / (area + area_i - inter_area);

    if (iou > iou_threshold) {
      keep[idx] = 0;
      return;
    }
  }
  keep[idx] = 1;
  return;
}
```