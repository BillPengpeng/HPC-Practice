Bnll算子计算公式见基础实现，具体实现时正负情况有所不同。

## Bnll基础实现

```
// src/layer/bnll.cpp

BNLL::BNLL()
{
    // 单输入blob
    one_blob_only = true;

    // 支持inplace
    support_inplace = true;
}

int BNLL::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            // > 0时需要附加ptr[i]
            if (ptr[i] > 0)
                ptr[i] = ptr[i] + logf(1.f + expf(-ptr[i]));
            else
                ptr[i] = logf(1.f + expf(ptr[i]));
        }
    }

    return 0;
}
```

## x86平台加速版本

```
// src/layer/x86/bnll.cpp

BNLL_x86::BNLL_x86()
{
#if __SSE2__
    // 支持pack
    support_packing = true;
#endif // __SSE2__
}

int BNLL_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        __m512 _one_avx512 = _mm512_set1_ps(1.f);
        __m512 _zero_avx512 = _mm512_setzero_ps();
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            // 挑选大于0元素
            __mmask16 mask = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_GT_OQ);
            // __m512转换为__m512i，计算结果转换回__m512，负数情况会取反
            __m512 _abs_p = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(_p), _mm512_set1_epi32(0x7fffffff)));
            // 对应logf(1.f + expf(-ptr[i]))
            __m512 _tmp = log512_ps(_mm512_add_ps(_one_avx512, exp512_ps(_mm512_sub_ps(_zero_avx512, _abs_p))));
            // 附加正数
            _p = _mm512_mask_add_ps(_tmp, mask, _tmp, _p);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        __m256 _one_avx = _mm256_set1_ps(1.f);
        __m256 _zero_avx = _mm256_setzero_ps();
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 mask = _mm256_cmp_ps(_p, _mm256_setzero_ps(), _CMP_GT_OQ);
            __m256 _abs_p = _mm256_and_ps(_p, *(__m256*)_ps256_inv_sign_mask);
            __m256 _tmp = log256_ps(_mm256_add_ps(_one_avx, exp256_ps(_mm256_sub_ps(_zero_avx, _abs_p))));
            __m256 _x = _mm256_and_ps(_p, mask);
            _p = _mm256_add_ps(_x, _tmp);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        __m128 _one = _mm_set1_ps(1.f);
        __m128 _zero = _mm_setzero_ps();
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            __m128 mask = _mm_cmpgt_ps(_p, _zero);
            __m128 _abs_p = _mm_and_ps(_p, *(__m128*)_ps_inv_sign_mask);
            __m128 _tmp = log_ps(_mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _abs_p))));
            __m128 _x = _mm_and_ps(_p, mask);
            _p = _mm_add_ps(_x, _tmp);
            _mm_store_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            if (*ptr > 0)
                *ptr = *ptr + logf(1.f + expf(-(*ptr)));
            else
                *ptr = logf(1.f + expf(*ptr));
            ptr++;
        }
    }
    return 0;
}
```
