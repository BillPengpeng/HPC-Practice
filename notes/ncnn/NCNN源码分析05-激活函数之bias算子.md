Bias算子用于对输入特征图增加一个偏移量，对于1*C*D*H*W尺寸的输入，偏移量bias_data的尺寸一般等同于通道数量C。

## Absval基础实现

```
// src/layer/bias.cpp

Bias::Bias()
{
    // 单输入
    one_blob_only = true;
    
    // 支持inplace
    support_inplace = true;
}
int Bias::load_param(const ParamDict& pd)
{
    // bias数据尺寸
    bias_data_size = pd.get(0, 0);
    return 0;
}
int Bias::load_model(const ModelBin& mb)
{
    // 加载bias数据
    bias_data = mb.load(bias_data_size, 1);
    if (bias_data.empty())
        return -100;
    return 0;
}
int Bias::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        float bias = bias_data[q];
       
        // bias shape为C，逐通道处理所有元素
        for (int i = 0; i < size; i++)
        {
            ptr[i] += bias;
        }
    }
    return 0;
}

## x86平台加速版本

```
// src/layer/x86/bias_x86.cpp

int Bias_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
// ...

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float bias = bias_ptr[q];
        int i = 0;
#if __SSE2__
#if __AVX__
        {
            __m256 _bias256 = _mm256_set1_ps(bias);
            for (; i + 7 < size; i += 8)
            {
                // 两__m256元素相加
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _outp = _mm256_add_ps(_p, _bias256);
                _mm256_storeu_ps(ptr, _outp);
                ptr += 8;
            }
        }
#endif // __AVX__
        {
            __m128 _bias = _mm_set1_ps(bias);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _outp = _mm_add_ps(_p, _bias);
                _mm_storeu_ps(ptr, _outp);
                ptr += 4;
            }
        }
#endif // __SSE2__

        // ...
    }
    return 0;
}
```

## ARM平台加速版本

```
// src/layer/arm/bias_arm.cpp
// 遗留问题：Bias_arm不支持packing？

int Bias_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
// ...

    const float* bias_ptr = bias_data;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float bias = bias_ptr[q];

#if __ARM_NEON
        // 一次性处理4个元素
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        float32x4_t _bias = vdupq_n_f32(bias);
        for (; nn > 0; nn--)
        {
            // 两float32x4_t元素相加
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _outp = vaddq_f32(_p, _bias);
            vst1q_f32(ptr, _outp);

            ptr += 4;
        }
#endif // __ARM_NEON

        // ...
    }
    return 0;
}
```
