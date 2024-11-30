from_pixels函数用于把unsigned char*形式（例如：opencv读取）的图像数据转换为ncnn::Mat，通过组合pixels读取、图像格式转换、ROI提取、Resize实现不同的功能。

## pixels读取 + 图像格式转换
这里的from_pixels依据配置的type参数进行图像格式转换或直接pixels读取，涉及格式有GRAY、RGB、BGR、RGBA、BGRA；依据w、h、stride计算输入图像各个通道的起始地址，GRAY、RGB/BGR、RGBA/BGRA分别对应1、3、4个通道。

```
// src/mat.h

// convenient construct from pixel data
static Mat from_pixels(const unsigned char* pixels, int type, int w, int h, Allocator* allocator = 0);
// convenient construct from pixel data with stride(bytes-per-row) parameter
static Mat from_pixels(const unsigned char* pixels, int type, int w, int h, int stride, Allocator* allocator = 0);

// src/mat_pixel.cpp
Mat Mat::from_pixels(const unsigned char* pixels, int type, int w, int h, Allocator* allocator)
{
    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 3, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 1, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 4, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}
```

### 直接pixels读取
直接pixels读取包括三种情况，分别是from_rgb、from_gray、from_rgba。

```
// src/mat_pixel.cpp
Mat Mat::from_pixels(const unsigned char* pixels, int type, int w, int h, int stride, Allocator* allocator)
{
    Mat m;

    if (type & PIXEL_CONVERT_MASK)
    {
        // ... 
    }
    else
    {
        if (type == PIXEL_RGB || type == PIXEL_BGR)
            from_rgb(pixels, w, h, stride, m, allocator);

        if (type == PIXEL_GRAY)
            from_gray(pixels, w, h, stride, m, allocator);

        if (type == PIXEL_RGBA || type == PIXEL_BGRA)
            from_rgba(pixels, w, h, stride, m, allocator);
    }

    return m;
}
```

以from_rgb为例，主要流程是读取交换rgb三通道数值，并将unsigned char格式的图像数据强制转换为float格式。

```
// src/mat_pixel.cpp

static int from_rgb(const unsigned char* rgb, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 3, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON  
        // 这里采用ARM_NEON或内嵌汇编 同时处理8*3=24个数值
        int nn = w >> 3;
        // 每行剩余像素数量
        int remain = w - (nn << 3); 
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if !NCNN_GNU_INLINE_ASM || __aarch64__
        for (; nn > 0; nn--)
        {
            // Load multiple 3-element structures to three registers
            uint8x8x3_t _rgb = vld3_u8(rgb); 
            // uint8x8_t强转为uint16x8_t
            uint16x8_t _r16 = vmovl_u8(_rgb.val[0]); 
            uint16x8_t _g16 = vmovl_u8(_rgb.val[1]);
            uint16x8_t _b16 = vmovl_u8(_rgb.val[2]);

            // uint16x8_t->uint16x4_t->uint32x4_t->float32x4_t
            float32x4_t _rlow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16))); 
            float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
            float32x4_t _glow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
            float32x4_t _blow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

            // 逐个存储4个元素
            vst1q_f32(ptr0, _rlow);
            vst1q_f32(ptr0 + 4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1 + 4, _ghigh);
            vst1q_f32(ptr2, _blow);
            vst1q_f32(ptr2 + 4, _bhigh);

            rgb += 3 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld3.u8    {d0-d2}, [%1]!      \n"
                "vmovl.u8   q8, d0              \n"
                "vmovl.u8   q9, d1              \n"
                "vmovl.u8   q10, d2             \n"
                "vmovl.u16  q0, d16             \n"
                "vmovl.u16  q1, d17             \n"
                "vmovl.u16  q2, d18             \n"
                "vmovl.u16  q3, d19             \n"
                "vmovl.u16  q8, d20             \n"
                "vmovl.u16  q9, d21             \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "vcvt.f32.u32   q2, q2          \n"
                "vcvt.f32.u32   q3, q3          \n"
                "vcvt.f32.u32   q8, q8          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%2]!      \n"
                "vcvt.f32.u32   q9, q9          \n"
                "vst1.f32   {d4-d7}, [%3]!      \n"
                "vst1.f32   {d16-d19}, [%4]!    \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(rgb),  // %1
                "=r"(ptr0), // %2
                "=r"(ptr1), // %3
                "=r"(ptr2)  // %4
                : "0"(nn),
                "1"(rgb),
                "2"(ptr0),
                "3"(ptr1),
                "4"(ptr2)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr0 = rgb[0];
            *ptr1 = rgb[1];
            *ptr2 = rgb[2];

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgb += wgap;
    }

    return 0;
}
```

### 图像格式转换
图像格式转换涉及格式有GRAY、RGB、BGR、RGBA、BGRA，包括：from_rgb2bgr、from_rgb2gray、from_rgb2rgba、from_bgr2gray、from_bgr2rgba、from_gray2rgb、from_gray2rgba、from_rgba2rgb、from_rgba2bgr、from_rgba2gray、from_rgba2bgra、from_bgra2gray。

以from_rgb2gray为例，主要流程是读取交换rgb三通道数值，并将unsigned char格式的图像数据强制转换为float格式。


```
// src/mat_pixel.cpp

static int from_rgb2gray(const unsigned char* rgb, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8; //14
    // R、G、B转换系数
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        // 这里采用ARM_NEON或内嵌汇编 同时处理8*3=24个数值
        int nn = w >> 3;
        // 每行剩余像素数量
        int remain = w - (nn << 3); 
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if !NCNN_GNU_INLINE_ASM || __aarch64__
        // uint8_t->uint8x8_t
        uint8x8_t _R2Y = vdup_n_u8(R2Y);
        uint8x8_t _G2Y = vdup_n_u8(G2Y);
        uint8x8_t _B2Y = vdup_n_u8(B2Y);
        for (; nn > 0; nn--)
        {
            // Load multiple 3-element structures to three registers
            uint8x8x3_t _rgb = vld3_u8(rgb);
            
            // 两个uint8x8_t相乘，输出uint16x8_t
            uint16x8_t _y16 = vmull_u8(_rgb.val[0], _R2Y);
            // a + b * c
            _y16 = vmlal_u8(_y16, _rgb.val[1], _G2Y);
            _y16 = vmlal_u8(_y16, _rgb.val[2], _B2Y);
            _y16 = vshrq_n_u16(_y16, Y_shift);
            
            // uint16x8_t->uint16x4_t->uint32x4_t->float32x4_t
            float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
            float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));
            
            // 逐个存储4个元素
            vst1q_f32(ptr, _ylow);
            vst1q_f32(ptr + 4, _yhigh);

            rgb += 3 * 8;
            ptr += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "vdup.u8    d16, %6             \n"
                "vdup.u8    d17, %7             \n"
                "vdup.u8    d18, %8             \n"
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld3.u8    {d0-d2}, [%1]!      \n"
                "vmull.u8   q2, d0, d16         \n"
                "vmlal.u8   q2, d1, d17         \n"
                "vmlal.u8   q2, d2, d18         \n"
                "vshr.u16   q2, q2, #8          \n" // Y_shift
                "vmovl.u16  q0, d4              \n"
                "vmovl.u16  q1, d5              \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),  // %0
                "=r"(rgb), // %1
                "=r"(ptr)  // %2
                : "0"(nn),
                "1"(rgb),
                "2"(ptr),
                "r"(R2Y), // %6
                "r"(G2Y), // %7
                "r"(B2Y)  // %8
                : "cc", "memory", "q0", "q1", "q2", "q8", "q9");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr = static_cast<float>((rgb[0] * R2Y + rgb[1] * G2Y + rgb[2] * B2Y) >> Y_shift);

            rgb += 3;
            ptr++;
        }
        rgb += wgap;
    }
    return 0;
}
```

## resize + pixels读取
这里的from_pixels分两步执行，先对输入图像做resize，目标尺寸是target_width、target_height；再执行像素读取，依据配置的type参数进行图像格式转换或直接pixels读取。

```
// src/mat.h

// convenient construct from pixel data and resize to specific size
static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height, Allocator* allocator = 0);
// convenient construct from pixel data and resize to specific size with stride(bytes-per-row) parameter
static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height, Allocator* allocator = 0);

// src/mat_pixel.cpp
Mat Mat::from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height, Allocator* allocator)
{
    if (w == target_width && h == target_height)
        return Mat::from_pixels(pixels, type, w, h, stride, allocator);

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        Mat dst(target_width, target_height, (size_t)3u, 3);
        resize_bilinear_c3(pixels, w, h, stride, dst, target_width, target_height, target_width * 3);

        return Mat::from_pixels(dst, type, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        Mat dst(target_width, target_height, (size_t)1u, 1);
        resize_bilinear_c1(pixels, w, h, stride, dst, target_width, target_height, target_width * 1);
        return Mat::from_pixels(dst, type, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        Mat dst(target_width, target_height, (size_t)4u, 4);
        resize_bilinear_c4(pixels, w, h, stride, dst, target_width, target_height, target_width * 4);
        return Mat::from_pixels(dst, type, target_width, target_height, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}
```

这里的resize，不同格式的输入图像对应不同通道的resize操作，resize_bilinear_c1、resize_bilinear_c3、resize_bilinear_c4分别应对单通道、3通道、4通道输入图像的resize。

## ROI提取 + pixels读取
这里的from_pixels依据roi图像数据起始地址、roiw、roih进行pixels读取，参数stride影响roi起始地址计算，在没有参数stride的情况下，起始地址为pixels + (roiy * w + roix) * 3；否则，起始地址为pixels + roiy * stride + roix * 3。

```
// src/mat.h

// convenient construct from pixel data roi
static Mat from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, Allocator* allocator = 0);
// convenient construct from pixel data roi with stride(bytes-per-row) parameter
static Mat from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, Allocator* allocator = 0);

// src/mat_pixel.cpp
Mat Mat::from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, Allocator* allocator)
{
    if (roix < 0 || roiy < 0 || roiw <= 0 || roih <= 0 || roix + roiw > w || roiy + roih > h)
    {
        NCNN_LOGE("roi %d %d %d %d out of image %d %d", roix, roiy, roiw, roih, w, h);
        return Mat();
    }

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return from_pixels(pixels + (roiy * w + roix) * 3, type, roiw, roih, w * 3, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return from_pixels(pixels + (roiy * w + roix) * 1, type, roiw, roih, w * 1, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return from_pixels(pixels + (roiy * w + roix) * 4, type, roiw, roih, w * 4, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, Allocator* allocator)
{
    if (roix < 0 || roiy < 0 || roiw <= 0 || roih <= 0 || roix + roiw > w || roiy + roih > h)
    {
        NCNN_LOGE("roi %d %d %d %d out of image %d %d", roix, roiy, roiw, roih, w, h);
        return Mat();
    }

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return from_pixels(pixels + roiy * stride + roix * 3, type, roiw, roih, stride, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return from_pixels(pixels + roiy * stride + roix * 1, type, roiw, roih, stride, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return from_pixels(pixels + roiy * stride + roix * 4, type, roiw, roih, stride, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}
```

## ROI提取 + Resize + pixels读取

这里的from_pixels首先依据roi图像数据起始地址、roiw、roih进行pixels读取，参数stride影响roi起始地址计算；然后对ROI图像做resize，目标尺寸是target_width、target_height；最后执行像素读取，依据配置的type参数进行图像格式转换或直接pixels读取。

```
// src/mat.h

// convenient construct from pixel data roi and resize to specific size
static Mat from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator = 0);
// convenient construct from pixel data roi and resize to specific size with stride(bytes-per-row) parameter
static Mat from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator = 0);

// src/mat_pixel.cpp
Mat Mat::from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator)
{
    if (roix < 0 || roiy < 0 || roiw <= 0 || roih <= 0 || roix + roiw > w || roiy + roih > h)
    {
        NCNN_LOGE("roi %d %d %d %d out of image %d %d", roix, roiy, roiw, roih, w, h);
        return Mat();
    }

    int type_from = type & PIXEL_FORMAT_MASK;
    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return from_pixels_resize(pixels + (roiy * w + roix) * 3, type, roiw, roih, w * 3, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return from_pixels_resize(pixels + (roiy * w + roix) * 1, type, roiw, roih, w * 1, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return from_pixels_resize(pixels + (roiy * w + roix) * 4, type, roiw, roih, w * 4, target_width, target_height, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator)
{
    if (roix < 0 || roiy < 0 || roiw <= 0 || roih <= 0 || roix + roiw > w || roiy + roih > h)
    {
        NCNN_LOGE("roi %d %d %d %d out of image %d %d", roix, roiy, roiw, roih, w, h);
        return Mat();
    }

    int type_from = type & PIXEL_FORMAT_MASK;
    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return from_pixels_resize(pixels + roiy * stride + roix * 3, type, roiw, roih, stride, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return from_pixels_resize(pixels + roiy * stride + roix * 1, type, roiw, roih, stride, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return from_pixels_resize(pixels + roiy * stride + roix * 4, type, roiw, roih, stride, target_width, target_height, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}
```