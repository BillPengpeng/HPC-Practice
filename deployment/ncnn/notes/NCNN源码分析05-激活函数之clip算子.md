cip的基本公式为：f(x) = max(min(a, x), b)，a、b为预设参数，在设置为0/6时，clip算子相当于ReLU6。

## clip基础实现

```
// src/layer/clip.cpp

int Clip::load_param(const ParamDict& pd)
{
    // 配置参数
    min = pd.get(0, -FLT_MAX);
    max = pd.get(1, FLT_MAX);

    return 0;
}

int Clip::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < min)
                ptr[i] = min;
            if (ptr[i] > max)
                ptr[i] = max;
        }
    }

    return 0;
}

```

## x86平台加速版本

```
// src/layer/x86/clip_x86.cpp

int Clip_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _min_avx512 = _mm512_set1_ps(min);
        __m512 _max_avx512 = _mm512_set1_ps(max);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            // max/min操作
            _p = _mm512_max_ps(_p, _min_avx512);
            _p = _mm512_min_ps(_p, _max_avx512);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        __m256 _min_avx = _mm256_set1_ps(min);
        __m256 _max_avx = _mm256_set1_ps(max);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = _mm256_max_ps(_p, _min_avx);
            _p = _mm256_min_ps(_p, _max_avx);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        __m128 _min = _mm_set1_ps(min);
        __m128 _max = _mm_set1_ps(max);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            _p = _mm_max_ps(_p, _min);
            _p = _mm_min_ps(_p, _max);
            _mm_store_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        // ... 
    }
    return 0;
}
```

## ARM平台加速版本

```
// src/layer/arm/clip_arm.cpp

int Clip_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

    // ...

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ARM_NEON
        float32x4_t _min = vdupq_n_f32(min);
        float32x4_t _max = vdupq_n_f32(max);
        for (; i + 15 < size; i += 16)
        {
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            // ...
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #512]      \n"
                "vldm       %0, {d0-d7}     \n"
                "vmax.f32   q0, q0, %q2     \n"
                "vmax.f32   q1, q1, %q2     \n"
                "vmax.f32   q2, q2, %q2     \n"
                "vmax.f32   q3, q3, %q2     \n"
                "vmin.f32   q0, q0, %q3     \n"
                "vmin.f32   q1, q1, %q3     \n"
                "vmin.f32   q2, q2, %q3     \n"
                "vmin.f32   q3, q3, %q3     \n"
                "vstm       %0!, {d0-d7}    \n"
                : "=r"(ptr) // %0
                : "0"(ptr),
                "w"(_min), // %2
                "w"(_max)  // %3
                : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            float32x4_t _p2 = vld1q_f32(ptr + 8);
            float32x4_t _p3 = vld1q_f32(ptr + 12);
            _p0 = vmaxq_f32(_p0, _min);
            _p1 = vmaxq_f32(_p1, _min);
            _p2 = vmaxq_f32(_p2, _min);
            _p3 = vmaxq_f32(_p3, _min);
            _p0 = vminq_f32(_p0, _max);
            _p1 = vminq_f32(_p1, _max);
            _p2 = vminq_f32(_p2, _max);
            _p3 = vminq_f32(_p3, _max);
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            vst1q_f32(ptr + 8, _p2);
            vst1q_f32(ptr + 12, _p3);
            ptr += 16;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            _p0 = vmaxq_f32(_p0, _min);
            _p1 = vmaxq_f32(_p1, _min);
            _p0 = vminq_f32(_p0, _max);
            _p1 = vminq_f32(_p1, _max);
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vmaxq_f32(_p, _min);
            _p = vminq_f32(_p, _max);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            if (*ptr < min)
                *ptr = min;

            if (*ptr > max)
                *ptr = max;

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
// F16推理情况
int Clip_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    // ...

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ARM_NEON
        float32x4_t _min = vdupq_n_f32(min);
        float32x4_t _max = vdupq_n_f32(max);
        for (; i + 15 < size; i += 16)
        {
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            // ...
#else  // __aarch64__
            // 左移16位转换为F32，结果右移16位转换回F16
            asm volatile(
                "pld        [%0, #256]      \n"
                "vld1.u16   {d4-d7}, [%0]   \n"
                "vshll.u16  q0, d4, #16     \n"
                "vshll.u16  q1, d5, #16     \n"
                "vshll.u16  q2, d6, #16     \n"
                "vshll.u16  q3, d7, #16     \n"
                "vmax.f32   q0, q0, %q2     \n"
                "vmax.f32   q1, q1, %q2     \n"
                "vmax.f32   q2, q2, %q2     \n"
                "vmax.f32   q3, q3, %q2     \n"
                "vmin.f32   q0, q0, %q3     \n"
                "vmin.f32   q1, q1, %q3     \n"
                "vmin.f32   q2, q2, %q3     \n"
                "vmin.f32   q3, q3, %q3     \n"
                "vshrn.u32  d0, q0, #16     \n"
                "vshrn.u32  d1, q1, #16     \n"
                "vshrn.u32  d2, q2, #16     \n"
                "vshrn.u32  d3, q3, #16     \n"
                "vst1.u16   {d0-d3}, [%0]!  \n"
                : "=r"(ptr) // %0
                : "0"(ptr),
                "w"(_min), // %2
                "w"(_max)  // %3
                : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            uint16x8_t _p = vld1q_u16(ptr);
            uint16x8_t _q = vld1q_u16(ptr + 8);
            // 转换为F32参与计算，计算结果转换为F16
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
            float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
            _p0 = vmaxq_f32(_p0, _min);
            _p1 = vmaxq_f32(_p1, _min);
            _p2 = vmaxq_f32(_p2, _min);
            _p3 = vmaxq_f32(_p3, _min);
            _p0 = vminq_f32(_p0, _max);
            _p1 = vminq_f32(_p1, _max);
            _p2 = vminq_f32(_p2, _max);
            _p3 = vminq_f32(_p3, _max);
            _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            _q = vcombine_u16(float2bfloat(_p2), float2bfloat(_p3));
            vst1q_u16(ptr, _p);
            vst1q_u16(ptr + 8, _q);
            ptr += 16;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            _p0 = vmaxq_f32(_p0, _min);
            _p1 = vmaxq_f32(_p1, _min);
            _p0 = vminq_f32(_p0, _max);
            _p1 = vminq_f32(_p1, _max);
            _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            vst1q_u16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _p = vmaxq_f32(_p, _min);
            _p = vminq_f32(_p, _max);
            vst1_u16(ptr, float2bfloat(_p));
            ptr += 4;
        }
#endif // __ARM_NEON
        // ...
    }

    return 0;
}
#endif // NCNN_BF16

```
