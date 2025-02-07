Absval算子用于对输入特征图取绝对值。

## Absval基础实现

```
// src/layer/absval.cpp

AbsVal::AbsVal()
{
    // 单输入blob
    one_blob_only = true;

    // 支持inplace
    support_inplace = true;
}
int AbsVal::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
            // 处理小于0元素
            if (ptr[i] < 0)
                ptr[i] = -ptr[i];
        }
    }
    return 0;
}
```

## ARM平台加速版本

```
// src/layer/arm/absval_arm.cpp
AbsVal_arm::AbsVal_arm()
{
#if __ARM_NEON
    // 支持packing
    support_packing = true;
#endif // __ARM_NEON
}

int AbsVal_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // ...
    // 注意此处元素数量考虑elempack
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ARM_NEON
        for (; i + 15 < size; i += 16)
        {
#if __aarch64__
            // ...
#else  // __aarch64__
       // pld 是一个预加载指令，用于告诉处理器预先加载数据到缓存中，以便后续的指令可以更快地访问这些数据
       // %0!，基地址寄存器%0通常会被更新为存储操作后下一个可用的内存地址。
            asm volatile(
                "pld        [%0, #512]      \n"
                "vldm       %0, {d0-d7}     \n"
                "vabs.f32   q0, q0          \n"
                "vabs.f32   q1, q1          \n"
                "vabs.f32   q2, q2          \n"
                "vabs.f32   q3, q3          \n"
                "vstm       %0!, {d0-d7}    \n" #
                : "=r"(ptr) // %0
                : "0"(ptr)
                : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
        }
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            // vabsq_f32计算float32x4_t对应的绝对值
            _p0 = vabsq_f32(_p0);
            _p1 = vabsq_f32(_p1);
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vabsq_f32(_p);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        // ...
    }
    return 0;
}
```
