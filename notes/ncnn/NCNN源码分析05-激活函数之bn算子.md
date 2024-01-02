BN的基本公式为：f(x) = (x - mean(x)) / sqrt(var(x) + ε) * γ + β。NCNN实现的BN算子将4个参数缩减为2个参数，支持1D、2D、3D输入特征图。

## BN基础实现

```
// src/layer/batchnorm.cpp

int BatchNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // 具体a b计算公式
    // a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a

    int dims = bottom_top_blob.dims;

    // 1D输入情况，a/b维度等于W
    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        float* ptr = bottom_top_blob;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < w; i++)
        {
            ptr[i] = b_data[i] * ptr[i] + a_data[i];
        }
    }

    // 2D输入情况，a/b维度等于H
    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float a = a_data[i];
            float b = b_data[i];

            for (int j = 0; j < w; j++)
            {
                ptr[j] = b * ptr[j] + a;
            }
        }
    }

    // 3D 4D输入情况，a/b维度等于C
    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];
            for (int i = 0; i < size; i++)
            {
                ptr[i] = b * ptr[i] + a;
            }
        }
    }
    return 0;
}
```

## x86平台加速版本

```
// src/layer/x86/batchnorm_x86.cpp

src/layer/x86/batchnorm_x86.cpp
int BatchNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // ...
    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];

#if __SSE2__
            // 3/4维pack 
            // c/4-d-h-w-4  c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
            __m128 _a128 = (elempack == 4) ? _mm_loadu_ps((const float*)a_data + q * 4) : _mm_set1_ps(a);
            __m128 _b128 = (elempack == 4) ? _mm_loadu_ps((const float*)b_data + q * 4) : _mm_set1_ps(b);
#if __AVX__
            // c/8-d-h-w-8  c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
            __m256 _a256 = (elempack == 8) ? _mm256_loadu_ps((const float*)a_data + q * 8) : _mm256_insertf128_ps(_mm256_castps128_ps256(_a128), _a128, 1);
            __m256 _b256 = (elempack == 8) ? _mm256_loadu_ps((const float*)b_data + q * 8) : _mm256_insertf128_ps(_mm256_castps128_ps256(_b128), _b128, 1);
#if __AVX512F__
            __m512 _a512 = (elempack == 16) ? _mm512_loadu_ps((const float*)a_data + q * 16) : _mm512_insertf32x8(_mm512_castps256_ps512(_a256), _a256, 1);
            __m512 _b512 = (elempack == 16) ? _mm512_loadu_ps((const float*)b_data + q * 16) : _mm512_insertf32x8(_mm512_castps256_ps512(_b256), _b256, 1);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p512 = _mm512_loadu_ps(ptr);
                _p512 = _mm512_fmadd_ps(_p512, _b512, _a512);
                _mm512_storeu_ps(ptr, _p512);
                ptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p256 = _mm256_loadu_ps(ptr);
                _p256 = _mm256_comp_fmadd_ps(_p256, _b256, _a256);
                _mm256_storeu_ps(ptr, _p256);
                ptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p128 = _mm_loadu_ps(ptr);
                _p128 = _mm_comp_fmadd_ps(_p128, _b128, _a128);
                _mm_storeu_ps(ptr, _p128);
                ptr += 4;
            }
#endif // __SSE__
            for (; i < size; i++)
            {
                *ptr = b * *ptr + a;
                ptr++;
            }
        }
    }
    return 0;
}

```

## ARM平台加速版本

```
// src/layer/arm/batchnorm_arm.cpp

int BatchNorm_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // ...

#if __ARM_NEON
    if (elempack == 4)
    {
        // ...
        // 3/4维pack 
        // c/4-d-h-w-4  c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
        if (dims == 3 || dims == 4)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int d = bottom_top_blob.d;
            int c = bottom_top_blob.c;
            int size = w * h * d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < c; q++)
            {
                float32x4_t _a = vld1q_f32((const float*)a_data + q * 4);
                float32x4_t _b = vld1q_f32((const float*)b_data + q * 4);

                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    _p = vmlaq_f32(_a, _p, _b);
                    vst1q_f32(ptr, _p);

                    ptr += 4;
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    // 常规非pack情况
    // ...
    return 0;
}

// F16推理情况
int BatchNorm_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    // ...

#if __ARM_NEON
    // BF16按unsigned short*处理
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 4;

                float32x4_t _a = vld1q_f32((const float*)a_data + i * 4);
                float32x4_t _b = vld1q_f32((const float*)b_data + i * 4);

                // 左移16位转为float32x4_t
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                _p = vmlaq_f32(_a, _p, _b);
                // 右移16位转为uint16_t
                vst1_u16(ptr, float2bfloat(_p));
            }
        }

        // ...

        return 0;
    }
#endif // __ARM_NEON

    // ... 

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            float a = a_data[q];
            float b = b_data[q];

            int j = 0;
#if __ARM_NEON
            float32x4_t _a = vdupq_n_f32(a);
            float32x4_t _b = vdupq_n_f32(b);

            for (; j + 3 < size; j += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                _p = vmlaq_f32(_a, _p, _b);
                vst1_u16(ptr, float2bfloat(_p));

                ptr += 4;
            }
#endif // __ARM_NEON
            for (; j < size; j++)
            {
                *ptr = float32_to_bfloat16(b * bfloat16_to_float32(*ptr) + a);

                ptr++;
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16
```
