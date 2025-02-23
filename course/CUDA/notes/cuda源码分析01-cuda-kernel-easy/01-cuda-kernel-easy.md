## sigmoid/relu/elu/gelu

### sigmoid cuda kernal实现

当x的值非常大时，expf(x)会溢出。对于float类型的输入，这个溢出点通常发生在x大约等于88.7；当x的值非常小时，expf(x)会趋近于零，下溢点通常发生在x小于大约-88.7时。  
对于公式中常量，采用__float2half由f32精度转换至half精度。

```
// -------------------------------------- FP32 -------------------------------------- 
// Sigmoid x: N, y: N y=1/(1+exp(-x))
// grid(N/256), block(K=256) 
__global__ void sigmoid_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = x[idx];
    // fminf/fmaxf/expf 出自cmath/math.h
    v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32); 
    y[idx] = 1.0f / (1.0f + expf(-v));
  }
}

// -------------------------------------- FP16 -------------------------------------- 
__global__ void sigmoid_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const half f = __float2half(1.0f);
  if (idx < N) {
    half v = x[idx];
    // __hmin/__hmax/hexp 出自Half Precision Intrinsics
    v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
    y[idx] = f / (f + hexp(-v));
  }
}
```

### relu cuda kernal实现

```
// -------------------------------------- FP32 -------------------------------------- 
// Relu x: N, y: N y=max(0,x)
// grid(N/256), block(K=256) 
__global__ void relu_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = fmaxf(0.0f, x[idx]);
}

// -------------------------------------- FP16 -------------------------------------- 
__global__ void relu_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = __hmax(__float2half(0.0f), x[idx]);
}
```

### elu cuda kernal实现

$$
\text{ELU}(x) = 
\begin{cases} 
x & \text{if } x > 0 \\
\alpha (\exp(x) - 1) & \text{if } x \leq 0 
\end{cases}
$$

```
// ELU 计算函数
// inline 是标准C/C++的一部分,__forceinline__ 是MSVC特有的指令
// -------------------------------------- FP32 --------------------------------------
__device__ __forceinline__ float elu(float x) {
  return x > 0.f ? x : ALPHA * (expf(x) - 1.f);
}

// -------------------------------------- FP16 --------------------------------------
__device__ __forceinline__ half elu_half(half x) {
  return __hgt(x, __float2half(0.f)) ? x : __hmul(__float2half(ALPHA), __hsub(hexp(x), __float2half(1.f)));
}

// -------------------------------------- FP32 --------------------------------------
__global__ void elu_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = elu(x[idx]);
}

// -------------------------------------- FP16 --------------------------------------
__global__ void elu_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = elu_half(x[idx]);
}
```

### gelu cuda kernal实现

$$
\text{GELU}(x) = 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)
$$

```
__inline__ __device__ float gelu_tanh_approximate(float x){
  return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x)));
}

__inline__ __device__ half gelu_tanh_approximate(half x){
  half x_cube = x * x * x;
  // compute mid value : inner = 0.7978845608 * (x + 0.044715 * x * x * x)
  half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * x_cube);
  // compute tanh
  return HALF_DIV2 * x * (HALF_1 + ((hexp(inner * HALF_2) - HALF_1) / (hexp(inner * HALF_2) + HALF_1))); 
}

// -------------------------------------- FP32 -------------------------------------- 
// GELU tanh approximate: x, y:x 0.5 * x * (1.0 + tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x)))
// grid(N/256), block(K=256) 
__global__ void gelu_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = GELU_OPS(v);
  }
}

__global__ void gelu_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    half v = x[idx];
    v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
    
    y[idx] = HALF_GELU_OPS(v);
  }
}
```

```
!TORCH_CUDA_ARCH_LIST=Ampere python3 sigmoid.py
!TORCH_CUDA_ARCH_LIST=Ampere python3 relu.py
!TORCH_CUDA_ARCH_LIST=Ampere python3 elu.py
!TORCH_CUDA_ARCH_LIST=Ampere python3 gelu.py
```

## swish/hardswish/hardshrink

### swish

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

```
// -------------------------------------- FP32 --------------------------------------
// Swish x: N, y: N y=x*sigmoid(x)
__device__ __forceinline__ float swish(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void swish_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = swish(x[idx]);
}

// -------------------------------------- FP16 --------------------------------------
__device__ __forceinline__ half swish_half(half x) {
  return __hmul(x, __hdiv(
    __float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(x)))));
}

__global__ void swish_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = swish_half(x[idx]);
}
```

### hardswish

$$
\text{HardSwish}(x) = 
\begin{cases} 
0, & \text{if } x \leq -3 \\
x \left( \frac{x}{6} + \frac{1}{2} \right), & \text{if } -3 < x < 3 \\
x, & \text{if } x \geq 3 
\end{cases}
$$

```
// -------------------------------------- FP32 --------------------------------------
__device__ __forceinline__ float hardswish(float x) {
  if (x >= THRESHOLD_A) {
    return x;
  } else if (x <= THRESHOLD_B) {
    return 0;
  } else {
    return x * (x + 3) / 6;
  }
}
__global__ void hardswish_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = hardswish(x[idx]);
}

// -------------------------------------- FP16 --------------------------------------
__device__ __forceinline__ half hardswish_half(half x) {
  if (x > __float2half(THRESHOLD_A)) {
    return x;
  } else if (x < __float2half(THRESHOLD_B)) {
    return __float2half(0.f);
  } else {
    return x * (x + __float2half(3.f)) / __float2half(6.f);
  }
}
__global__ void hardswish_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = hardswish_half(x[idx]);
}

```

### hardshrink

$$
\text{HardShrink}(x) = 
\begin{cases} 
x, & \text{if } x > \lambda \\
x, & \text{if } x < -\lambda \\
0, & \text{otherwise} 
\end{cases}
$$

```
// -------------------------------------- FP32 --------------------------------------
__device__ __forceinline__ float hardshrink(float x) {
  if (x > LAMBD || x < -LAMBD) {
    return x;
  } else {
    return 0;
  }
}
__global__ void hardshrink_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = hardshrink(x[idx]);
}

// -------------------------------------- FP16 --------------------------------------
__device__ __forceinline__ half hardshrink_half(half x) {
  if(x > __float2half(LAMBD) || x < __float2half(-LAMBD)) {
    return x;
  } else {
    return __float2half(0.f);
  }
}
__global__ void hardshrink_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = hardshrink_half(x[idx]);
}
```

## 其他

### embedding

对于输入中的每个ID，在embedding_weight中查找对应的行，并将该行的embedding向量作为该ID的embedding。

```
// idx    shape:N     
// weight shape:M*k
// output shape:N*k
__global__ void embedding_f32_kernel(const int *idx, float *weight, float *output, int n, int emb_size)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  // int tid = bx * blockDim.x + tx;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
}
```

### histogram

histogram对输入做直方图统计。  
atomicAdd函数是CUDA编程中的一个原子操作函数。  

```
// Histogram
// grid(N/256), block(256)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32_kernel(int* a, int* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) atomicAdd(&(y[a[idx]]), 1);
}
```