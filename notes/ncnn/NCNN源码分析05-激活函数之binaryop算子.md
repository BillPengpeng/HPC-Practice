binaryop的功能类似bitwise，实现最多两个输入特征图逐像素的加减乘除运算，当两个输入特征图出现尺寸不等时，有相关的broadcasting操作。

## binaryop基础实现

```
// src/layer/binaryop.cpp

BinaryOp::BinaryOp()
{
    // 默认要求输入两个blob，不支持inplace
    one_blob_only = false;

    // 不支持inplace
    support_inplace = false;
}

int BinaryOp::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);
    with_scalar = pd.get(1, 0);
    b = pd.get(2, 0.f);

    // with_scalar不为0时，仅要求输入一个blob，由参数b替代第二个blob
    if (with_scalar != 0)
    {
        one_blob_only = true;
        support_inplace = true;
    }
    return 0;
}

template<typename Op>
static void binary_op_broadcast(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    // general broadcast
    const Op op;
    const int dims = c.dims;
    const int w = c.w;
    const int h = c.h;
    const int d = c.d;
    const int channels = c.c;

    if (dims == 1)
    {
        // ...
    }
    if (dims == 2)
    {
         // ...
    }

    if (dims == 3 || dims == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = c.channel(q);
            const int ainc = a.w > 1 ? 1 : 0;
            const int binc = b.w > 1 ? 1 : 0;
            for (int z = 0; z < d; z++)
            {
                for (int y = 0; y < h; y++)
                {
                    // shape为三维CHW或四维CDHW情况，当ptr1的CDH维度长度低于ptr时，会重复元素
                    const float* ptr = a.channel(std::min(q, a.c - 1)).depth(std::min(z, a.d - 1)).row(std::min(y, a.h - 1));
                    const float* ptr1 = b.channel(std::min(q, b.c - 1)).depth(std::min(z, b.d - 1)).row(std::min(y, b.h - 1));
                    for (int x = 0; x < w; x++)
                    {
                        outptr[x] = op(*ptr, *ptr1);
                        ptr += ainc;
                        ptr1 += binc;
                    }
                    outptr += w;
                }
            }
        }
    }
}
template<typename Op>
static void binary_op_scalar_inplace(Mat& a, float b, const Option& opt)
{
    const Op op;

    const int channels = a.c;
    const int size = a.w * a.h * a.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);
        // inplace情况采用配置参数b执行op计算
        for (int i = 0; i < size; i++)
        {
            ptr[i] = op(ptr[i], b);
        }
    }
}
// 支持运算类型
static void binary_op_broadcast(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    // a + b
    if (op_type == BinaryOp::Operation_ADD) return binary_op_broadcast<binary_op_add>(a, b, c, opt);
    // a - b
    if (op_type == BinaryOp::Operation_SUB) return binary_op_broadcast<binary_op_sub>(a, b, c, opt);
    // a * b
    if (op_type == BinaryOp::Operation_MUL) return binary_op_broadcast<binary_op_mul>(a, b, c, opt);
    // a / b
    if (op_type == BinaryOp::Operation_DIV) return binary_op_broadcast<binary_op_div>(a, b, c, opt);
    // max(a, b)
    if (op_type == BinaryOp::Operation_MAX) return binary_op_broadcast<binary_op_max>(a, b, c, opt);
    // min(a, b)
    if (op_type == BinaryOp::Operation_MIN) return binary_op_broadcast<binary_op_min>(a, b, c, opt);
    // powf(a, b);
    if (op_type == BinaryOp::Operation_POW) return binary_op_broadcast<binary_op_pow>(a, b, c, opt);
    // b - a
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_broadcast<binary_op_sub>(b, a, c, opt);
    // b / a
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_broadcast<binary_op_div>(b, a, c, opt);
    // powf(b, a);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_broadcast<binary_op_pow>(b, a, c, opt);
    // atan2f(a, b)
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_broadcast<binary_op_atan2>(a, b, c, opt);
    // atan2f(b, a)
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_broadcast<binary_op_atan2>(b, a, c, opt);
    // should never reach here
}

int BinaryOp::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A = bottom_blobs[0];
    const Mat& B = bottom_blobs[1];
    const int outdims = std::max(A.dims, B.dims);

    Mat A2 = A;
    Mat B2 = B;
    // B.dims较大情况，需要扩展A维度，A维度为1时需要特殊处理，否则直接增加长度为1的dim
    if (A.dims < outdims)
    {
        // expand inner axes
        if (outdims == 2)
        {
            if (A.w == B.h)
                A2 = A.reshape(1, A.w);
            else // if (A.w == B.w)
                A2 = A.reshape(A.w, 1);
        }
        if (outdims == 3 && A.dims == 1)
        {
            if (A.w == B.c)
                A2 = A.reshape(1, 1, A.w);
            else // if (A.w == B.w)
                A2 = A.reshape(A.w, 1, 1);
        }
        if (outdims == 3 && A.dims == 2)
            A2 = A.reshape(1, A.w, A.h);
        if (outdims == 4 && A.dims == 1)
        {
            if (A.w == B.c)
                A2 = A.reshape(1, 1, 1, A.w);
            else // if (A.w == B.w)
                A2 = A.reshape(A.w, 1, 1, 1);
        }
        if (outdims == 4 && A.dims == 2)
            A2 = A.reshape(1, 1, A.w, A.h);
        if (outdims == 4 && A.dims == 3)
            A2 = A.reshape(1, A.w, A.h, A.c);
    }
    // A.dims较大情况，需要扩展B维度
    if (B.dims < outdims)
    {
        // ...
    }
    const int outw = std::max(A2.w, B2.w);
    const int outh = std::max(A2.h, B2.h);
    const int outd = std::max(A2.d, B2.d);
    const int outc = std::max(A2.c, B2.c);
    Mat& top_blob = top_blobs[0];
    // ...
    if (outdims == 3)
    {
        top_blob.create(outw, outh, outc, 4u, opt.blob_allocator);
    }
    // ...
    if (top_blob.empty())
        return -100;
    binary_op_broadcast(A2, B2, top_blob, op_type, opt);
    return 0;
}

```

## x86平台加速版本

```
// src/layer/x86/binaryop_x86.cpp

BinaryOp_x86::BinaryOp_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

// no_broadcast情况
template<typename Op>
static void binary_op_vector_no_broadcast(const float* ptr, const float* ptr1, float* outptr, int size)
{
    const Op op;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        __m512 _b = _mm512_loadu_ps(ptr1);
        // 调用op对应pack函数
        __m512 _outp = op.func_pack16(_p, _b);
        _mm512_storeu_ps(outptr, _outp);
        ptr += 16;
        ptr1 += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
       // ...
       __m256 _outp = op.func_pack8(_p, _b);
       // ...
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        // ...
        __m128 _outp = op.func_pack4(_p, _b);
        // ...
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr = op.func(*ptr, *ptr1);
        ptr += 1;
        ptr1 += 1;
        outptr += 1;
    }
}

// ptr1是单个元素
template<typename Op>
static void binary_op_vector_broadcast_b(const float* ptr, const float* ptr1, float* outptr, int size, int elempack)
{
    const Op op;

    const float b = *ptr1;

    int i = 0;
#if __SSE2__
    __m128 _b_128 = (elempack == 4) ? _mm_loadu_ps(ptr1) : _mm_set1_ps(b);
#if __AVX__
    // _mm256_castps128_ps256把128bit强转为256bit
    // _mm256_insertf128_ps第3个参数为1时，dst[255:0] := a[255:0]，dst[255:128] := b[127:0]
    __m256 _b_256 = (elempack == 8) ? _mm256_loadu_ps(ptr1) : _mm256_insertf128_ps(_mm256_castps128_ps256(_b_128), _b_128, 1);
#if __AVX512F__
    __m512 _b_512 = (elempack == 16) ? _mm512_loadu_ps(ptr1) : _mm512_insertf32x8(_mm512_castps256_ps512(_b_256), _b_256, 1);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        __m512 _outp = op.func_pack16(_p, _b_512);
        _mm512_storeu_ps(outptr, _outp);
        ptr += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    // ...
#endif // __AVX__
    // ...
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr = op.func(*ptr, b);
        ptr += 1;
        outptr += 1;
    }
}

// 功能与binary_op_vector_broadcast_b重复
// ptr是单个元素
template<typename Op>
static void binary_op_vector_broadcast_a(const float* ptr, const float* ptr1, float* outptr, int size, int elempack)
{
    const Op op;

    const float a = *ptr;

    int i = 0;
#if __SSE2__
    __m128 _a_128 = (elempack == 4) ? _mm_loadu_ps(ptr) : _mm_set1_ps(a);
#if __AVX__
    __m256 _a_256 = (elempack == 8) ? _mm256_loadu_ps(ptr) : _mm256_insertf128_ps(_mm256_castps128_ps256(_a_128), _a_128, 1);
#if __AVX512F__
    __m512 _a_512 = (elempack == 16) ? _mm512_loadu_ps(ptr) : _mm512_insertf32x8(_mm512_castps256_ps512(_a_256), _a_256, 1);
    for (; i + 15 < size; i += 16)
    {
        __m512 _b = _mm512_loadu_ps(ptr1);
        __m512 _outp = op.func_pack16(_a_512, _b);
        _mm512_storeu_ps(outptr, _outp);
        ptr1 += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    // ...
#endif // __AVX__
    // ...
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr = op.func(a, *ptr1);
        ptr1 += 1;
        outptr += 1;
    }
}

// pack1 b，ptr、ptr1长度相同
template<typename Op>
static void binary_op_vector_broadcast_pb(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        int i = 0;
        for (; i < w; i++)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _b = _mm512_set1_ps(*ptr1);
            __m512 _outp = op.func_pack16(_p, _b);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            ptr1 += 1;
            outptr += 16;
        }
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        int i = 0;
#if __AVX512F__
        for (; i + 1 < w; i += 2)
        {
            // pack8的ptr1拼接成pack16
            __m512 _p = _mm512_loadu_ps(ptr);
            __m256 _b0 = _mm256_set1_ps(ptr1[0]);
            __m256 _b1 = _mm256_set1_ps(ptr1[1]);
            __m512 _b = _mm512_insertf32x8(_mm512_castps256_ps512(_b0), _b1, 1);
            __m512 _outp = op.func_pack16(_p, _b);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            ptr1 += 2;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i < w; i++)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _b = _mm256_set1_ps(*ptr1);
            __m256 _outp = op.func_pack8(_p, _b);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            ptr1 += 1;
            outptr += 8;
        }
    }
#endif // __AVX__
    if (elempack == 4)
    {
        int i = 0;
#if __AVX__
#if __AVX512F__
        for (; i + 3 < w; i += 4)
        {
            // pack4的ptr1拼接成pack16
            __m512 _p = _mm512_loadu_ps(ptr);
            __m128 _b0 = _mm_set1_ps(ptr1[0]);
            __m128 _b1 = _mm_set1_ps(ptr1[1]);
            __m128 _b2 = _mm_set1_ps(ptr1[2]);
            __m128 _b3 = _mm_set1_ps(ptr1[3]);
            __m256 _b01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_b0), _b1, 1);
            __m256 _b23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_b2), _b3, 1);
            __m512 _b = _mm512_insertf32x8(_mm512_castps256_ps512(_b01), _b23, 1);
            __m512 _outp = op.func_pack16(_p, _b);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            ptr1 += 4;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 1 < w; i += 2)
        {
            // pack4的ptr1拼接成pack8
            __m256 _p = _mm256_loadu_ps(ptr);
            __m128 _b0 = _mm_set1_ps(ptr1[0]);
            __m128 _b1 = _mm_set1_ps(ptr1[1]);
            __m256 _b = _mm256_insertf128_ps(_mm256_castps128_ps256(_b0), _b1, 1);
            __m256 _outp = op.func_pack8(_p, _b);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            ptr1 += 2;
            outptr += 8;
        }
#endif // __AVX__
        for (; i < w; i++)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _b = _mm_set1_ps(*ptr1);
            __m128 _outp = op.func_pack4(_p, _b);
            _mm_storeu_ps(outptr, _outp);
            ptr += 4;
            ptr1 += 1;
            outptr += 4;
        }
    }
#endif // __SSE2__
}

// pack1 single b，ptr1是一个float元素
template<typename Op>
static void binary_op_vector_broadcast_pb_b(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

    const int size = w * elempack;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _b_avx512 = _mm512_set1_ps(*ptr1);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        __m512 _outp = op.func_pack16(_p, _b_avx512);
        _mm512_storeu_ps(outptr, _outp);
        ptr += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    __m256 _b_avx = _mm256_set1_ps(*ptr1);
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        __m256 _outp = op.func_pack8(_p, _b_avx);
        _mm256_storeu_ps(outptr, _outp);
        ptr += 8;
        outptr += 8;
    }
#endif // __AVX__
    __m128 _b = _mm_set1_ps(*ptr1);
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        __m128 _outp = op.func_pack4(_p, _b);
        _mm_storeu_ps(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif // __SSE2__
}

// single a and pack1 b，ptr是一个float元素，ptr1一个pack后的元素
template<typename Op>
static void binary_op_vector_broadcast_pb_a(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        int i = 0;
        __m512 _p = _mm512_loadu_ps(ptr);
        for (; i < w; i++)
        {
            __m512 _b = _mm512_set1_ps(*ptr1);
            __m512 _outp = op.func_pack16(_p, _b);
            _mm512_storeu_ps(outptr, _outp);
            ptr1 += 1;
            outptr += 16;
        }
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        int i = 0;
        __m256 _p = _mm256_loadu_ps(ptr);
#if __AVX512F__
        __m512 _p_512 = _mm512_insertf32x8(_mm512_castps256_ps512(_p), _p, 1);
        for (; i + 1 < w; i += 2)
        {
            __m256 _b0 = _mm256_set1_ps(ptr1[0]);
            __m256 _b1 = _mm256_set1_ps(ptr1[1]);
            __m512 _b = _mm512_insertf32x8(_mm512_castps256_ps512(_b0), _b1, 1);
            __m512 _outp = op.func_pack16(_p_512, _b);
            _mm512_storeu_ps(outptr, _outp);
            ptr1 += 2;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i < w; i++)
        {
            __m256 _b = _mm256_set1_ps(*ptr1);
            __m256 _outp = op.func_pack8(_p, _b);
            _mm256_storeu_ps(outptr, _outp);
            ptr1 += 1;
            outptr += 8;
        }
    }
#endif // __AVX__
    if (elempack == 4)
    {
        int i = 0;
        __m128 _p = _mm_loadu_ps(ptr);
#if __AVX__
        __m256 _p_256 = _mm256_insertf128_ps(_mm256_castps128_ps256(_p), _p, 1);
#if __AVX512F__
        __m512 _p_512 = _mm512_insertf32x8(_mm512_castps256_ps512(_p_256), _p_256, 1);
        for (; i + 3 < w; i += 4)
        {
            __m128 _b0 = _mm_set1_ps(ptr1[0]);
            __m128 _b1 = _mm_set1_ps(ptr1[1]);
            __m128 _b2 = _mm_set1_ps(ptr1[2]);
            __m128 _b3 = _mm_set1_ps(ptr1[3]);
            __m256 _b01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_b0), _b1, 1);
            __m256 _b23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_b2), _b3, 1);
            __m512 _b = _mm512_insertf32x8(_mm512_castps256_ps512(_b01), _b23, 1);
            __m512 _outp = op.func_pack16(_p_512, _b);
            _mm512_storeu_ps(outptr, _outp);
            ptr1 += 4;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 1 < w; i += 2)
        {
            __m128 _b0 = _mm_set1_ps(ptr1[0]);
            __m128 _b1 = _mm_set1_ps(ptr1[1]);
            __m256 _b = _mm256_insertf128_ps(_mm256_castps128_ps256(_b0), _b1, 1);
            __m256 _outp = op.func_pack8(_p_256, _b);
            _mm256_storeu_ps(outptr, _outp);
            ptr1 += 2;
            outptr += 8;
        }
#endif // __AVX__
        for (; i < w; i++)
        {
            __m128 _b = _mm_set1_ps(*ptr1);
            __m128 _outp = op.func_pack4(_p, _b);
            _mm_storeu_ps(outptr, _outp);
            ptr1 += 1;
            outptr += 4;
        }
    }
#endif // __SSE2__
}

template<typename Op>
static void binary_op_vector(const float* ptr, const float* ptr1, float* outptr, int aw, int bw, int ap, int bp)
{
    const int w = std::max(aw, bw);
    const int elempack = std::max(ap, bp);
    const int size = w * elempack;

    if (ap == bp)
    {
        if (aw == bw)
        {
            // no broadcast
            // ap、bp相等，aw、bw相等
            return binary_op_vector_no_broadcast<Op>(ptr, ptr1, outptr, size);
        }

        if (bw == 1)
        {
            // broadcast single b
            // ap、bp相等，aw > 1，bw = 1
            return binary_op_vector_broadcast_b<Op>(ptr, ptr1, outptr, size, elempack);
        }

        if (aw == 1)
        {
            // broadcast single a
            // ap、bp相等，aw = 1，bw > 1
            return binary_op_vector_broadcast_a<Op>(ptr, ptr1, outptr, size, elempack);
        }
    }

    if (bp == 1)
    {
        if (aw == bw)
        {
            // broadcast pack1 b
            // ap > 1，bp = 1，aw、bw相等
            return binary_op_vector_broadcast_pb<Op>(ptr, ptr1, outptr, w, elempack);
        }

        if (bw == 1)
        {
            // broadcast pack1 single b
            // ap > 1，bp = 1，aw > 1，bw = 1
            return binary_op_vector_broadcast_pb_b<Op>(ptr, ptr1, outptr, w, elempack);
        }

        if (aw == 1)
        {
            // broadcast single a and pack1 b
            // ap > 1，bp = 1，aw = 1，bw > 1
            return binary_op_vector_broadcast_pb_a<Op>(ptr, ptr1, outptr, w, elempack);
        }
    }

    // shall never reach here
}

```

## ARM平台加速版本

```
// src/layer/arm/binaryop_arm.cpp

// no_broadcast情况
template<typename Op>
static void binary_op_vector_no_broadcast(const float* ptr, const float* ptr1, float* outptr, int size)
{
    const Op op;

    int i = 0;
#if __ARM_NEON
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _b = vld1q_f32(ptr1);
        float32x4_t _outp = op(_p, _b);
        vst1q_f32(outptr, _outp);
        ptr += 4;
        ptr1 += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr = op(*ptr, *ptr1);
        ptr += 1;
        ptr1 += 1;
        outptr += 1;
    }
}

// ptr1是单个元素
template<typename Op>
static void binary_op_vector_broadcast_b(const float* ptr, const float* ptr1, float* outptr, int size, int elempack)
{
    const Op op;

    const float b = *ptr1;

    int i = 0;
#if __ARM_NEON
    float32x4_t _b_128 = (elempack == 4) ? vld1q_f32(ptr1) : vdupq_n_f32(b);
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _outp = op(_p, _b_128);
        vst1q_f32(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr = op(*ptr, b);
        ptr += 1;
        outptr += 1;
    }
}

// 功能与binary_op_vector_broadcast_b重复
// ptr是单个元素
template<typename Op>
static void binary_op_vector_broadcast_a(const float* ptr, const float* ptr1, float* outptr, int size, int elempack)
{
    const Op op;

    const float a = *ptr;

    int i = 0;
#if __ARM_NEON
    float32x4_t _a_128 = (elempack == 4) ? vld1q_f32(ptr) : vdupq_n_f32(a);
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _b = vld1q_f32(ptr1);
        float32x4_t _outp = op(_a_128, _b);
        vst1q_f32(outptr, _outp);
        ptr1 += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr = op(a, *ptr1);
        ptr1 += 1;
        outptr += 1;
    }
}

// pack1 b，ptr、ptr1长度相同
template<typename Op>
static void binary_op_vector_broadcast_pb(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

#if __ARM_NEON
    if (elempack == 4)
    {
        int i = 0;
        for (; i < w; i++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _b = vdupq_n_f32(*ptr1);
            float32x4_t _outp = op(_p, _b);
            vst1q_f32(outptr, _outp);
            ptr += 4;
            ptr1 += 1;
            outptr += 4;
        }
    }
#endif // __ARM_NEON
}

// pack1 single b，ptr1是一个float元素
template<typename Op>
static void binary_op_vector_broadcast_pb_b(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

    const int size = w * elempack;

    int i = 0;
#if __ARM_NEON
    float32x4_t _b = vdupq_n_f32(*ptr1);
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _outp = op(_p, _b);
        vst1q_f32(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
}

// single a and pack1 b，ptr是一个float元素，ptr1一个pack后的元素
template<typename Op>
static void binary_op_vector_broadcast_pb_a(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

#if __ARM_NEON
    if (elempack == 4)
    {
        int i = 0;
        float32x4_t _p = vld1q_f32(ptr);
        for (; i < w; i++)
        {
            float32x4_t _b = vdupq_n_f32(*ptr1);
            float32x4_t _outp = op(_p, _b);
            vst1q_f32(outptr, _outp);
            ptr1 += 1;
            outptr += 4;
        }
    }
#endif // __ARM_NEON
}

```
