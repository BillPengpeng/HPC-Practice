ReLU的基本公式为：f(x) = max(0, x)。NCNN实现的ReLU算子实质是Leaky ReLU，当系数slope等于0时，Leaky ReLU退化为基础的ReLU。此处暂提供基础/x86/arm平台源码解析，其他平台及内嵌汇编源码解析待补充。

## ReLU基础实现

```
// src/layer/relu.cpp

ReLU::ReLU()
{
    // 支持单输入
    one_blob_only = true;

    // 支持inplace
    support_inplace = true;
}

int ReLU::load_param(const ParamDict& pd)
{
    // slope系数，非0时为Leaky ReLU
    slope = pd.get(0, 0.f);

    return 0;
}

int ReLU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            // ReLU算子实现
            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            // Leaky ReLU算子实现
            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }

    return 0;
}

```

## x86平台加速版本

```
// src/layer/x86/relu_x86.cpp

ReLU_x86::ReLU_x86()
{
#if __SSE2__
    // 支持packing
    support_packing = true;
#endif // __SSE2__
}

int ReLU_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

    // INT8推理情况
    if (elembits == 8)
        return forward_inplace_int8(bottom_top_blob, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _zero_avx512 = _mm512_setzero_ps();
            for (; i + 15 < size; i += 16)
            {
                // __AVX512F__情况一次性处理16个32字节元素，共512字节长度
                __m512 _p = _mm512_loadu_ps(ptr);
                _mm512_storeu_ps(ptr, _mm512_max_ps(_zero_avx512, _p));
                ptr += 16;
            }
#endif // __AVX512F__
            __m256 _zero_avx = _mm256_setzero_ps();
            for (; i + 7 < size; i += 8)
            {
                // __AVX__情况一次性处理8个32字节元素，共256字节长度
                __m256 _p = _mm256_loadu_ps(ptr);
                _mm256_storeu_ps(ptr, _mm256_max_ps(_zero_avx, _p));
                ptr += 8;
            }
#endif // __AVX__
            __m128 _zero = _mm_setzero_ps();
            for (; i + 3 < size; i += 4)
            {
                // SSE情况一次性处理4个32字节元素，共128字节长度
                __m128 _p = _mm_load_ps(ptr);
                _mm_store_ps(ptr, _mm_max_ps(_zero, _p));
                ptr += 4;
            }
#endif // __SSE2__
            // 剩余元素按基础实现
            for (; i < size; i++)
            {
                *ptr = std::max(*ptr, 0.f);
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _zero_avx512 = _mm512_setzero_ps();
            __m512 _slope_avx512 = _mm512_set1_ps(slope);
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                // 计算小于等于0元素对应mask
                __mmask16 _is_negative = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_LT_OQ);
                // _is_negative对应元素执行乘法运算
                _p = _mm512_mask_mul_ps(_p, _is_negative, _p, _slope_avx512);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
            }
#endif // __AVX512F__
            __m256 _zero_avx = _mm256_setzero_ps();
            __m256 _slope_avx = _mm256_set1_ps(slope);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _pos = _mm256_max_ps(_zero_avx, _p);
                __m256 _neg = _mm256_min_ps(_zero_avx, _p);
                // 针对小于0元素执行乘法运算
                _p = _mm256_add_ps(_pos, _mm256_mul_ps(_slope_avx, _neg));
                _mm256_storeu_ps(ptr, _p);
                ptr += 8;
            }
#endif // __AVX__
            __m128 _zero = _mm_setzero_ps();
            __m128 _slope = _mm_set1_ps(slope);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                __m128 _pos = _mm_max_ps(_zero, _p);
                __m128 _neg = _mm_min_ps(_zero, _p);
                _p = _mm_add_ps(_pos, _mm_mul_ps(_slope, _neg));
                _mm_store_ps(ptr, _p);
                ptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
        }
    }

    return 0;
}

```

## ARM平台加速版本

```
// src/layer/arm/relu_arm.cpp

ReLU_arm::ReLU_arm()
{
#if __ARM_NEON
    // 支持pack
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    // 支持BF16
    support_bf16_storage = true;
#endif
}

int ReLU_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

    // INT8情况
    if (elembits == 8)
        return forward_inplace_int8(bottom_top_blob, opt);

#if NCNN_ARM82
    // F16情况
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

#if NCNN_BF16
    // F16情况
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __ARM_NEON
            float32x4_t _zero = vdupq_n_f32(0.f);
            // 每行一次性处理16个像素
            for (; i + 15 < size; i += 16)
            {
#if NCNN_GNU_INLINE_ASM
             // ...
#else  // NCNN_GNU_INLINE_ASM
                // vld1q_f32加载128个字节元素
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                float32x4_t _p2 = vld1q_f32(ptr + 8);
                float32x4_t _p3 = vld1q_f32(ptr + 12);
                _p0 = vmaxq_f32(_p0, _zero);
                _p1 = vmaxq_f32(_p1, _zero);
                _p2 = vmaxq_f32(_p2, _zero);
                _p3 = vmaxq_f32(_p3, _zero);
                vst1q_f32(ptr, _p0);
                vst1q_f32(ptr + 4, _p1);
                vst1q_f32(ptr + 8, _p2);
                vst1q_f32(ptr + 12, _p3);
                ptr += 16;
#endif // NCNN_GNU_INLINE_ASM
            }
            // 剩余元素按一次性处理8个像素
            for (; i + 7 < size; i += 8)
            {
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                _p0 = vmaxq_f32(_p0, _zero);
                _p1 = vmaxq_f32(_p1, _zero);
                vst1q_f32(ptr, _p0);
                vst1q_f32(ptr + 4, _p1);
                ptr += 8;
            }
            // 剩余元素按一次性处理4个像素
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _ptr = vld1q_f32(ptr);
                _ptr = vmaxq_f32(_ptr, _zero);
                vst1q_f32(ptr, _ptr);
                ptr += 4;
            }
#endif // __ARM_NEON
            // 剩余元素，基础实现
            for (; i < size; i++)
            {
                *ptr = std::max(*ptr, 0.f);
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __ARM_NEON
            float32x4_t _zero = vdupq_n_f32(0.f);
            float32x4_t _slope = vdupq_n_f32(slope);
            for (; i + 15 < size; i += 16)
            {
#if NCNN_GNU_INLINE_ASM
                // ... 
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                float32x4_t _p2 = vld1q_f32(ptr + 8);
                float32x4_t _p3 = vld1q_f32(ptr + 12);
                // 统计小于等于0元素mask
                uint32x4_t _lemask0 = vcleq_f32(_p0, _zero);
                uint32x4_t _lemask1 = vcleq_f32(_p1, _zero);
                uint32x4_t _lemask2 = vcleq_f32(_p2, _zero);
                uint32x4_t _lemask3 = vcleq_f32(_p3, _zero);
                float32x4_t _ps0 = vmulq_f32(_p0, _slope);
                float32x4_t _ps1 = vmulq_f32(_p1, _slope);
                float32x4_t _ps2 = vmulq_f32(_p2, _slope);
                float32x4_t _ps3 = vmulq_f32(_p3, _slope);
                // 依据mask挑选元素
                _p0 = vbslq_f32(_lemask0, _ps0, _p0);
                _p1 = vbslq_f32(_lemask1, _ps1, _p1);
                _p2 = vbslq_f32(_lemask2, _ps2, _p2);
                _p3 = vbslq_f32(_lemask3, _ps3, _p3);
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
                uint32x4_t _lemask0 = vcleq_f32(_p0, _zero);
                uint32x4_t _lemask1 = vcleq_f32(_p1, _zero);
                float32x4_t _ps0 = vmulq_f32(_p0, _slope);
                float32x4_t _ps1 = vmulq_f32(_p1, _slope);
                _p0 = vbslq_f32(_lemask0, _ps0, _p0);
                _p1 = vbslq_f32(_lemask1, _ps1, _p1);
                vst1q_f32(ptr, _p0);
                vst1q_f32(ptr + 4, _p1);
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1q_f32(ptr, _p);
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
        }
    }

    return 0;
}

// BF16推理情况
int ReLU_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    // ...

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            // BF16情况，采用unsigned short*
            unsigned short* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __ARM_NEON
            float32x4_t _zero = vdupq_n_f32(0.f);
            for (; i + 15 < size; i += 16)
            {
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
                // ...
#else  // __aarch64__
                // ...
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                uint16x8_t _p = vld1q_u16(ptr);
                uint16x8_t _q = vld1q_u16(ptr + 8);
                // 左移16位
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                _p0 = vmaxq_f32(_p0, _zero);
                _p1 = vmaxq_f32(_p1, _zero);
                _p2 = vmaxq_f32(_p2, _zero);
                _p3 = vmaxq_f32(_p3, _zero);
                // 右移16位
                _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
                _q = vcombine_u16(float2bfloat(_p2), float2bfloat(_p3));
                vst1q_u16(ptr, _p);
                vst1q_u16(ptr + 8, _q);
                ptr += 16;
#endif // NCNN_GNU_INLINE_ASM
            }
            // ...
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(ptr[0]);
                if (v < 0.f)
                    ptr[0] = float32_to_bfloat16(0.f);
                ptr += 1;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __ARM_NEON
            float32x4_t _zero = vdupq_n_f32(0.f);
            float32x4_t _slope = vdupq_n_f32(slope);
            for (; i + 15 < size; i += 16)
            {
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
                // ...
#else  // __aarch64__
                // ...
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                uint16x8_t _p = vld1q_u16(ptr);
                uint16x8_t _q = vld1q_u16(ptr + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                uint32x4_t _lemask0 = vcleq_f32(_p0, _zero);
                uint32x4_t _lemask1 = vcleq_f32(_p1, _zero);
                uint32x4_t _lemask2 = vcleq_f32(_p2, _zero);
                uint32x4_t _lemask3 = vcleq_f32(_p3, _zero);
                float32x4_t _ps0 = vmulq_f32(_p0, _slope);
                float32x4_t _ps1 = vmulq_f32(_p1, _slope);
                float32x4_t _ps2 = vmulq_f32(_p2, _slope);
                float32x4_t _ps3 = vmulq_f32(_p3, _slope);
                _p0 = vbslq_f32(_lemask0, _ps0, _p0);
                _p1 = vbslq_f32(_lemask1, _ps1, _p1);
                _p2 = vbslq_f32(_lemask2, _ps2, _p2);
                _p3 = vbslq_f32(_lemask3, _ps3, _p3);
                _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
                _q = vcombine_u16(float2bfloat(_p2), float2bfloat(_p3));
                vst1q_u16(ptr, _p);
                vst1q_u16(ptr + 8, _q);
                ptr += 16;
#endif // NCNN_GNU_INLINE_ASM
            }
            // ...
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(ptr[0]);
                if (v < 0.f)
                    ptr[0] = float32_to_bfloat16(v * slope);
                ptr += 1;
            }
        }
    }

    return 0;
}

```
