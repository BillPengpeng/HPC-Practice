Kanna_rotate原理如下图所示，不同type表示由该type转换至1的图像变换，即1-8分别对应无变换、flip horizontal、rotate 180、flip vertical、transpose、rotate 90、transverse、rotate 270。

```
// src/mat.h

// type is the from type, 6 means rotating from 6 to 1
//
//     1        2       3      4         5            6           7          8
//
//   888888  888888      88  88      8888888888  88                  88  8888888888
//   88          88      88  88      88  88      88  88          88  88      88  88
//   8888      8888    8888  8888    88          8888888888  8888888888          88
//   88          88      88  88
//   88          88  888888  888888
//
// ref http://sylvana.net/jpegcrop/exif_orientation.html
// image pixel kanna rotate
NCNN_EXPORT void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
NCNN_EXPORT void kanna_rotate_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
NCNN_EXPORT void kanna_rotate_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
NCNN_EXPORT void kanna_rotate_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
// image pixel kanna rotate with stride(bytes-per-row) parameter
NCNN_EXPORT void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
NCNN_EXPORT void kanna_rotate_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
NCNN_EXPORT void kanna_rotate_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
NCNN_EXPORT void kanna_rotate_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);

```

这里的kanna_rotate_c1~kanna_rotate_c4分别应用于单通道~四通道输入图像，以kanna_rotate_c1为例，type1~8切换至不同的分支。

```
// src/mat_pixel_rotate.cpp

void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    // assert srcw == w && srch == h for type 1234
    // assert srcw == h && srch == w for type 5678

    switch (type)
    {
    case 1:
        kanna_rotate_1_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 2:
        kanna_rotate_2_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 3:
        kanna_rotate_3_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 4:
        kanna_rotate_4_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 5:
        kanna_rotate_5_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 6:
        kanna_rotate_6_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 7:
        kanna_rotate_7_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 8:
        kanna_rotate_8_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    default:
        // unsupported rotate type
        break;
    }
}
```

## kanna_rotate_1_c1/无变换

kanna_rotate_1_c1源码如下，相当于在考虑srcstride情况下的直接拷贝，一般情况下srcstride等同于srcw，这里逐行进行数据的直接拷贝。

```
// src/mat_pixel_rotate.cpp

static void kanna_rotate_1_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    // 每行冗余像素数量
    const int srcwgap = srcstride - srcw;
    const int wgap = stride - w;

    // 同时处理两行数据
    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dst;
    unsigned char* dst1 = dst + stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
#if __ARM_NEON
        // 每行一次性处理32个像素
        int nn = srcw >> 5;
        int remain = srcw - (nn << 5);
#if !NCNN_GNU_INLINE_ASM || __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "pld        [%2, #256]          \n"
                "vld1.u8    {d4-d7}, [%2]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%3]!      \n"
                "vst1.u8    {d4-d7}, [%4]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1)
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 += wgap + stride;
        dst1 += wgap + stride;
    }
    
    // 剩余行数按每次处理一行接着处理
    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = srcw >> 5;
        int remain = srcw - (nn << 5);
#if !NCNN_GNU_INLINE_ASM || __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1");
        }
#endif // __aarch64__
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}
```

## kanna_rotate_2_c1/flip horizontal

相较于kanna_rotate_1_c1，kanna_rotate_2_c1不同点有：1）逐行拷贝的内存增长方向相反；2）外层for循环每次处理的行数发生变换，有2行缩减为1行；3）内层for循环每行处理的像素数量发生变换，由32缩减为16。

```
// src/mat_pixel_rotate.cpp

static void kanna_rotate_2_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw;
    const int wgap = stride + w;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst + w - 1;

    int y = 0;
    // 每次处理一行
    for (; y < srch; y++)
    {
#if __ARM_NEON
        // 修改每行的起始点
        dst0 -= 15;

        // 每行处理16个元素
        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if !NCNN_GNU_INLINE_ASM || __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8_t _src = vld1_u8(src0);
            uint8x8_t _src2 = vld1_u8(src0 + 8);
            
            // 调换8个像素数值顺序
            _src = vrev64_u8(_src);
            _src2 = vrev64_u8(_src2);

            vst1_u8(dst0, _src2);
            vst1_u8(dst0 + 8, _src);

            // 内存增长方向相反
            src0 += 16;
            dst0 -= 16;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "mov        r4, #-16            \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.u8    {d0-d1}, [%1]!      \n"
                "vrev64.u8  d3, d0              \n"
                "vrev64.u8  d2, d1              \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d2-d3}, [%2], r4   \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1", "r4");
        }
#endif // __aarch64__

        dst0 += 15;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 -= 1;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}
```

## kanna_rotate_3_c1/rotate 180

相较于kanna_rotate_1_c1，kanna_rotate_3_c1不同点有：1）起始点发生变换，rotate 180等同于中心的完全对称，起始点由左上角调整为右下角；2）外层for循环每次处理的行数发生变换，有2行缩减为1行；3）内层for循环每行处理的像素数量发生变换，由32缩减为16。

```
// src/mat_pixel_rotate.cpp

static void kanna_rotate_3_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw;
    const int wgap = stride - w;

    // point to the last dst pixel
    // dst目标图像起始点调整为右下角
    unsigned char* dstend = dst + stride * h - wgap;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 1;

    int y = 0;
    for (; y < srch; y++)
    {
#if __ARM_NEON
        dst0 -= 15;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if !NCNN_GNU_INLINE_ASM || __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8_t _src = vld1_u8(src0);
            uint8x8_t _src2 = vld1_u8(src0 + 8);

             // 调换8个像素数值顺序
            _src = vrev64_u8(_src);
            _src2 = vrev64_u8(_src2);

            vst1_u8(dst0, _src2);
            vst1_u8(dst0 + 8, _src);

            // 内存增长方向相反
            src0 += 16;
            dst0 -= 16;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "mov        r4, #-16            \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.u8    {d0-d1}, [%1]!      \n"
                "vrev64.u8  d3, d0              \n"
                "vrev64.u8  d2, d1              \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d2-d3}, [%2], r4   \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1", "r4");
        }
#endif // __aarch64__

        dst0 += 15;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 -= 1;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}
```

## kanna_rotate_4_c1/flip vertical

相较于kanna_rotate_1_c1，kanna_rotate_4_c1不同点有：起始点发生变换，起始点由第一行起点调整为最后一行起点。

```
// src/mat_pixel_rotate.cpp

static void kanna_rotate_4_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw;
    const int wgap = stride + w;

    // point to the last dst pixel row
    // 起始点发生变换，由第一行起点调整为最后一行起点
    unsigned char* dstend = dst + stride * (h - 1);

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - stride;

    int y = 0;
    // 每次处理2行
    for (; y + 1 < srch; y += 2)
    {
#if __ARM_NEON
        // 每行每次处理32个像素
        int nn = srcw >> 5;
        int remain = srcw - (nn << 5);
#if !NCNN_GNU_INLINE_ASM || __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "pld        [%2, #256]          \n"
                "vld1.u8    {d4-d7}, [%2]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%3]!      \n"
                "vst1.u8    {d4-d7}, [%4]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1)
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 -= wgap + stride;
        dst1 -= wgap + stride;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = srcw >> 5;
        int remain = srcw - (nn << 5);
#if !NCNN_GNU_INLINE_ASM || __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1");
        }
#endif // __aarch64__
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}
```

## kanna_rotate_5_c1/transpose

相较于kanna_rotate_1_c1，kanna_rotate_5_c1不同点有：1）外层for循环每次处理的行数发生变换，有2行增加为8行；2）内层for循环每行处理的像素数量发生变换，由32缩减为8。

```
// src/mat_pixel_rotate.cpp

static void kanna_rotate_5_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    // 一次性处理8行
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst0 = dst + y;
        unsigned char* dst1 = dst + y + stride;

        int src_step = 2 * srcstride;
        int dst_step = 2 * stride;

        // 每行一次性处理8个元素
        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if !NCNN_GNU_INLINE_ASM || __aarch64__
        for (; nn > 0; nn--)
        {
            // 加载uint8x8_t
            // 0000,0000
            uint8x8_t _src0 = vld1_u8(src0);
            // 1111,1111
            uint8x8_t _src1 = vld1_u8(src1);

            // 2222,2222
            uint8x8_t _src2 = vld1_u8(src0 + src_step);
            // 3333,3333
            uint8x8_t _src3 = vld1_u8(src1 + src_step);

            // 4444,4444
            uint8x8_t _src4 = vld1_u8(src0 + 2 * src_step);
            // 5555,5555
            uint8x8_t _src5 = vld1_u8(src1 + 2 * src_step);

            // 6666,6666
            uint8x8_t _src6 = vld1_u8(src0 + 3 * src_step);
            // 7777,7777
            uint8x8_t _src7 = vld1_u8(src1 + 3 * src_step);

            // 相邻两行u8元素concat 	uint8x8x2_t
            // 01(0)01(1),01(2)01(3),01(4)01(5),01(6)01(7)
            uint8x8x2_t _src01t_r = vtrn_u8(_src0, _src1);
            // 23(0)23(1),2323,2323,2323
            uint8x8x2_t _src23t_r = vtrn_u8(_src2, _src3);
            // 45(0)45(1),4545,4545,4545
            uint8x8x2_t _src45t_r = vtrn_u8(_src4, _src5);
            // 67(0)67(1),6767,6767,6767
            uint8x8x2_t _src67t_r = vtrn_u8(_src6, _src7);

            // uint8x8_t -> uint16x4_t-> uint16x4x2_t  0123 行u16元素concat 
            // 01(0)23(0),01(2)23(2),0123,0123
            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[0]), vreinterpret_u16_u8(_src23t_r.val[0]));
            // 01(1)23(1),01(3)23(3),0123,0123
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[1]), vreinterpret_u16_u8(_src23t_r.val[1]));
            // 45(0)67(0),45(2)67(2),4567,4567
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[0]), vreinterpret_u16_u8(_src67t_r.val[0]));
            // 45(1)67(1),45(3)67(3),4567,4567
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[1]), vreinterpret_u16_u8(_src67t_r.val[1]));
            
            //  01(0)23(0)45(0)67(0),01234567
            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[0]), vreinterpret_u32_u16(_src46tt_r.val[0]));
            //  01(1)23(1)45(1)67(1),01234567
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[0]), vreinterpret_u32_u16(_src57tt_r.val[0]));
            //  01(2)23(2)45(2)67(2),01234567
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[1]), vreinterpret_u32_u16(_src46tt_r.val[1]));
            //  01(3)23(3)45(3)67(3),01234567
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[1]), vreinterpret_u32_u16(_src57tt_r.val[1]));

            uint8x8_t _dst0 = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            uint8x8_t _dst1 = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            uint8x8_t _dst2 = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            uint8x8_t _dst3 = vreinterpret_u8_u32(_src37ttt_r.val[0]);
            uint8x8_t _dst4 = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            uint8x8_t _dst5 = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            uint8x8_t _dst6 = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            uint8x8_t _dst7 = vreinterpret_u8_u32(_src37ttt_r.val[1]);

            vst1_u8(dst0, _dst0);
            vst1_u8(dst1, _dst1);
            vst1_u8(dst0 + dst_step, _dst2);
            vst1_u8(dst1 + dst_step, _dst3);
            vst1_u8(dst0 + 2 * dst_step, _dst4);
            vst1_u8(dst1 + 2 * dst_step, _dst5);
            vst1_u8(dst0 + 3 * dst_step, _dst6);
            vst1_u8(dst1 + 3 * dst_step, _dst7);

            src0 += 8;
            src1 += 8;

            dst0 += 4 * dst_step;
            dst1 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #64]           \n"
                "vld1.u8    {d0}, [%1], %10     \n"

                "pld        [%2, #64]           \n"
                "vld1.u8    {d1}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d2}, [%1], %10     \n"

                "vtrn.u8    d0, d1              \n" // _src01t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d3}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d4}, [%1], %10     \n"

                "vtrn.u8    d2, d3              \n" // _src23t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d5}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d6}, [%1], %10     \n"

                "vtrn.u8    d4, d5              \n" // _src45t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d7}, [%2], %10     \n"

                "vtrn.u8    d6, d7              \n" // _src67t_r

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q0, q1              \n" // _src02tt_r _src13tt_r

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q2, q3              \n" // _src13tt_r _src46tt_r

                "add        %1, #8              \n" // src0 += 8

                "vtrn.u32   q0, q2              \n" // _src04ttt_r _src15ttt_r

                "add        %2, #8              \n" // src1 += 8

                "vtrn.u32   q1, q3              \n" // _src26ttt_r _src37ttt_r
                "vst1.u8    {d0}, [%3], %11     \n"
                "vst1.u8    {d1}, [%4], %11     \n"

                "subs       %0, #1              \n"

                "vst1.u8    {d2}, [%3], %11     \n"
                "vst1.u8    {d3}, [%4], %11     \n"
                "vst1.u8    {d4}, [%3], %11     \n"
                "vst1.u8    {d5}, [%4], %11     \n"
                "vst1.u8    {d6}, [%3], %11     \n"
                "vst1.u8    {d7}, [%4], %11     \n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src1[0];
            dst0[2] = src0[0 + src_step];
            dst0[3] = src1[0 + src_step];
            dst0[4] = src0[0 + 2 * src_step];
            dst0[5] = src1[0 + 2 * src_step];
            dst0[6] = src0[0 + 3 * src_step];
            dst0[7] = src1[0 + 3 * src_step];

            src0 += 1;
            src1 += 1;

            dst0 += stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dst + y;

        int x = 0;
        for (; x < srcw; x++)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}
```

## kanna_rotate_6_c1/rotate 90

相较于kanna_rotate_1_c1，kanna_rotate_6_c1不同点有：1）外层for循环每次处理的行数发生变换，有2行增加为8行；2）内层for循环每行处理的像素数量发生变换，由32缩减为8; 3）起始点有第一行的第一个点调整为最后一个点。

```
// src/mat_pixel_rotate.cpp

static void kanna_rotate_6_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw;

    // point to the last dst pixel in row
    unsigned char* dstend = dst + w;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    // 一次性处理8行
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst0 = dstend - y - 8;
        unsigned char* dst1 = dstend - y - 8 + stride;

        int src_step = 2 * srcstride;
        int dst_step = 2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if !NCNN_GNU_INLINE_ASM || __aarch64__
        // 一次性处理8行
        for (; nn > 0; nn--)
        {
            uint8x8_t _src0 = vld1_u8(src0);
            uint8x8_t _src1 = vld1_u8(src1);

            uint8x8_t _src2 = vld1_u8(src0 + src_step);
            uint8x8_t _src3 = vld1_u8(src1 + src_step);

            uint8x8_t _src4 = vld1_u8(src0 + 2 * src_step);
            uint8x8_t _src5 = vld1_u8(src1 + 2 * src_step);

            uint8x8_t _src6 = vld1_u8(src0 + 3 * src_step);
            uint8x8_t _src7 = vld1_u8(src1 + 3 * src_step);

            //10(0)10(1),10(2)10(3),10(4)10(5),10(6)10(7)
            uint8x8x2_t _src01t_r = vtrn_u8(_src1, _src0);
            //32(0)32(1),32(2)32(3),32(4)32(5),32(6)32(7)
            uint8x8x2_t _src23t_r = vtrn_u8(_src3, _src2);
            //54(0)54(1),54(2)54(3),54(4)54(5),54(6)54(7)
            uint8x8x2_t _src45t_r = vtrn_u8(_src5, _src4);
            //76(0)76(1),76(2)76(3),76(4)76(5),76(6)76(7)
            uint8x8x2_t _src67t_r = vtrn_u8(_src7, _src6);

            // 32(1)10(1),32(3)10(3),32(5)10(5),32(7)10(7)
            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[1]), vreinterpret_u16_u8(_src01t_r.val[1]));
            // 32(0)10(0),32(2)10(2),32(4)10(4),32(6)10(6)
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[0]), vreinterpret_u16_u8(_src01t_r.val[0]));
            // 76(1)54(1),76(3)54(3),76(5)54(5),76(7)54(7)
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[1]), vreinterpret_u16_u8(_src45t_r.val[1]));
            // 76(0)54(0),76(2)54(2),76(4)54(4),76(6)54(6)
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[0]), vreinterpret_u16_u8(_src45t_r.val[0]));

            // 76(3)54(3),32(3)10(3),76(7)54(7),32(7)10(7)
            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[1]), vreinterpret_u32_u16(_src02tt_r.val[1]));
            // 76(2)54(2),32(2)10(2),76(6)54(6),32(6)10(6)
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[1]), vreinterpret_u32_u16(_src13tt_r.val[1]));
            // 76(1)54(1),32(1)10(1),76(5)54(5),32(5)10(5)
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[0]), vreinterpret_u32_u16(_src02tt_r.val[0]));
            // 76(0)54(0),32(0)10(0),76(4)54(4),32(4)10(4)
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[0]), vreinterpret_u32_u16(_src13tt_r.val[0]));

            // 76(7)54(7),32(7)10(7)
            uint8x8_t _dst0 = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            // 76(6)54(6),32(6)10(6)
            uint8x8_t _dst1 = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            // 76(5)54(5),32(5)10(5)
            uint8x8_t _dst2 = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            // 76(4)54(4),32(4)10(4)
            uint8x8_t _dst3 = vreinterpret_u8_u32(_src37ttt_r.val[1]);
            // 76(3)54(3),32(3)10(3)
            uint8x8_t _dst4 = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            // 76(2)54(2),32(2)10(2)
            uint8x8_t _dst5 = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            // 76(1)54(1),32(1)10(1)
            uint8x8_t _dst6 = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            // 76(0)54(0),32(0)10(0)
            uint8x8_t _dst7 = vreinterpret_u8_u32(_src37ttt_r.val[0]);

            vst1_u8(dst0, _dst7);
            vst1_u8(dst1, _dst6);
            vst1_u8(dst0 + dst_step, _dst5);
            vst1_u8(dst1 + dst_step, _dst4);
            vst1_u8(dst0 + 2 * dst_step, _dst3);
            vst1_u8(dst1 + 2 * dst_step, _dst2);
            vst1_u8(dst0 + 3 * dst_step, _dst1);
            vst1_u8(dst1 + 3 * dst_step, _dst0);

            src0 += 8;
            src1 += 8;

            dst0 += 4 * dst_step;
            dst1 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #64]           \n"
                "vld1.u8    {d0}, [%1], %10     \n"

                "pld        [%2, #64]           \n"
                "vld1.u8    {d1}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d2}, [%1], %10     \n"

                "vtrn.u8    d1, d0              \n" // _src01t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d3}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d4}, [%1], %10     \n"

                "vtrn.u8    d3, d2              \n" // _src23t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d5}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d6}, [%1], %10     \n"

                "vtrn.u8    d5, d4              \n" // _src45t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d7}, [%2], %10     \n"

                "vtrn.u8    d7, d6              \n" // _src67t_r

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q1, q0              \n" // _src02tt_r _src13tt_r

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q3, q2              \n" // _src46tt_r _src57tt_r

                "add        %1, #8              \n" // src0 += 8

                "vtrn.u32   q3, q1              \n" // _src26ttt_r _src37ttt_r

                "add        %2, #8              \n" // src1 += 8

                "vtrn.u32   q2, q0              \n" // _src04ttt_r _src15ttt_r
                "vst1.u8    {d6}, [%4], %11     \n"
                "vst1.u8    {d7}, [%3], %11     \n"

                "subs       %0, #1              \n"

                "vst1.u8    {d4}, [%4], %11     \n"
                "vst1.u8    {d5}, [%3], %11     \n"
                "vst1.u8    {d2}, [%4], %11     \n"
                "vst1.u8    {d3}, [%3], %11     \n"
                "vst1.u8    {d0}, [%4], %11     \n"
                "vst1.u8    {d1}, [%3], %11     \n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst0[0] = src1[0 + 3 * src_step];
            dst0[1] = src0[0 + 3 * src_step];
            dst0[2] = src1[0 + 2 * src_step];
            dst0[3] = src0[0 + 2 * src_step];
            dst0[4] = src1[0 + src_step];
            dst0[5] = src0[0 + src_step];
            dst0[6] = src1[0];
            dst0[7] = src0[0];

            src0 += 1;
            src1 += 1;

            dst0 += stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y - 1;

        int x = 0;
        for (; x < srcw; x++)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}
```

## 其他

anna_rotate_7_c1/transverse、anna_rotate_8_c1/rotate 270类似kanna_rotate_5_c1/transpose、kanna_rotate_6_c1/rotate 90，区别是起点、两重for训练内存增长方向差异。

