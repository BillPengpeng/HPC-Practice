NCNN中Convolution与ConvolutionDepthWise原理类似，convolution相当于group等于1情况下的ConvolutionDepthWise。

## 配置项
两种卷积的配置项基本一致，ConvolutionDepthWise在Convolution参数基础上，额外增加group参数，并要求输出通道num_output可以整除group。

```
// src/layer/convolution.cpp

Convolution::Convolution()
{
    one_blob_only = true;
    support_inplace = false;
}

int Convolution::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);

    // int8推理配置项
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    dynamic_weight = pd.get(19, 0);

    if (dynamic_weight)
    {
        one_blob_only = false;
    }

    if (int8_scale_term)
    {
#if NCNN_INT8
        support_int8_storage = true;
#else
        NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
        return -1;
#endif
    }

    return 0;
}

// src/layer/convolutiondepthwise.cpp
int ConvolutionDepthWise::load_param(const ParamDict& pd)
{
    // ...

    group = pd.get(7, 1);

    if (num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }
    // ...
    return 0;
}
```

## 模型参数加载
两种卷积的模型参数加载流程类似，依次加载weight数据、bias数据、weight对应int8 scale参数、bias对应int8 scale参数。区别是：Convolution对应weight_data_int8_scales的shape为num_output*1，ConvolutionDepthWise对应weight_data_int8_scales的shape有group*1、1*1两种情况，视配置参数而定。

```
// src/layer/convolution.cpp

int Convolution::load_model(const ModelBin& mb)
{
    if (dynamic_weight)
        return 0;

    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

#if NCNN_INT8
    if (int8_scale_term)
    {
        weight_data_int8_scales = mb.load(num_output, 1);
        bottom_blob_int8_scales = mb.load(1, 1);
    }

    if (int8_scale_term > 100)
    {
        top_blob_int8_scales = mb.load(1, 1);
    }
#endif // NCNN_INT8

    return 0;
}

// src/layer/convolutiondepthwise.cpp

int ConvolutionDepthWise::load_model(const ModelBin& mb)
{
    // ...

#if NCNN_INT8
    if (int8_scale_term == 1 || int8_scale_term == 101)
    {
        weight_data_int8_scales = mb.load(group, 1);
        bottom_blob_int8_scales = mb.load(1, 1);

        float bottom_blob_int8_scale = bottom_blob_int8_scales[0];
        bottom_blob_int8_scales = Mat(group);
        bottom_blob_int8_scales.fill(bottom_blob_int8_scale);
    }
    else if (int8_scale_term == 2 || int8_scale_term == 102)
    {
        weight_data_int8_scales = mb.load(1, 1);
        bottom_blob_int8_scales = mb.load(1, 1);

        // extend group if only one provided
        float weight_data_int8_scale = weight_data_int8_scales[0];
        weight_data_int8_scales = Mat(group);
        weight_data_int8_scales.fill(weight_data_int8_scale);

        float bottom_blob_int8_scale = bottom_blob_int8_scales[0];
        bottom_blob_int8_scales = Mat(group);
        bottom_blob_int8_scales.fill(bottom_blob_int8_scale);
    }

    if (int8_scale_term > 100)
    {
        top_blob_int8_scales = mb.load(1, 1);

        float top_blob_int8_scale = top_blob_int8_scales[0];
        top_blob_int8_scales = Mat(group);
        top_blob_int8_scales.fill(top_blob_int8_scale);
    }
#endif // NCNN_INT8

    return 0;
}
```

## create_pipeline
两种卷积的create_pipeline过程均依据weight_data_int8_scales对weight_data进行量化，Convolution所有

```
// src/layer/convolution.cpp

int Convolution::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

#if NCNN_INT8
    // runtime quantize the weight data
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)4u && int8_scale_term)
    {
        const int maxk = kernel_w * kernel_h;
        const int num_input = weight_data_size / num_output / maxk;

        // weight_data尺寸num_output*num_input*maxk
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        Mat weight_data_int8;

        Option opt_q = opt;
        opt_q.blob_allocator = weight_data.allocator;
        opt_q.use_packing_layout = false;
        quantize_to_int8(weight_data_r2, weight_data_int8, weight_data_int8_scales, opt_q);
        if (weight_data_int8.empty())
            return -100;

        weight_data = weight_data_int8.reshape(weight_data_size);
    }
#else
    (void)(opt);
#endif // NCNN_INT8

    return 0;
}

// src/layer/convolutiondepthwise.cpp

int ConvolutionDepthWise::create_pipeline(const Option& opt)
{
#if NCNN_INT8
    // runtime quantize the weight data
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)4u && int8_scale_term)
    {
        Mat int8_weight_data(weight_data_size, (size_t)1u);
        if (int8_weight_data.empty())
            return -100;

        // weight_data尺寸是group的整数倍
        const int weight_data_size_g = weight_data_size / group;

        for (int g = 0; g < group; g++)
        {
            Option opt_q = opt;
            opt_q.blob_allocator = int8_weight_data.allocator;
            opt_q.use_packing_layout = false;

            const Mat weight_data_g = weight_data.range(weight_data_size_g * g, weight_data_size_g);
            Mat int8_weight_data_g = int8_weight_data.range(weight_data_size_g * g, weight_data_size_g);
            const Mat weight_data_int8_scales_g = weight_data_int8_scales.range(g, 1);
            quantize_to_int8(weight_data_g, int8_weight_data_g, weight_data_int8_scales_g, opt_q);
        }

        weight_data = int8_weight_data;
    }
#else
    (void)(opt);
#endif // NCNN_INT8

    return 0;
}
```

## 前向推理
两种卷积的前向推理主要流程有：
1. 判断特殊条件释放触发：INT8推理、InnerProduct等效替换；
2. 对输入作padding操作；
3. 计算输出blob的shape，并申请内存初始化；
4. 调用卷积运算主处理函数。

```
// src/layer/convolution.cpp
int Convolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    // 判断释放触发int8推理
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8(bottom_blob, top_blob, opt);
    }
#endif

    // flattened blob, implement as InnerProduct
    // 判断是否可以等效替换成InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        int num_input = weight_data_size / num_output;
        if (bottom_blob.w * bottom_blob.elempack == num_input)
        {
            // call InnerProduct
            ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::InnerProduct);

            // set param
            ncnn::ParamDict pd;
            pd.set(0, num_output);
            pd.set(1, bias_term);
            pd.set(2, weight_data_size);
            pd.set(8, int8_scale_term);
            pd.set(9, activation_type);
            pd.set(10, activation_params);
            op->load_param(pd);

            // set weights
            ncnn::Mat weights[4];
            weights[0] = weight_data;
            weights[1] = bias_data;

#if NCNN_INT8
            if (int8_scale_term)
            {
                weights[2] = weight_data_int8_scales;
                weights[3] = bottom_blob_int8_scales;
            }
#endif
            op->load_model(ModelBinFromMatArray(weights));
            op->create_pipeline(opt);

            // forward
            op->forward(bottom_blob, top_blob, opt);
            op->destroy_pipeline(opt);
            delete op;
            return 0;
        }
    }

    // padding
    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const size_t elemsize = bottom_blob_bordered.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int ret = convolution(bottom_blob_bordered, top_blob, weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

// src/layer/convolutiondepthwise.cpp

int ConvolutionDepthWise::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8(bottom_blob, top_blob, opt);
    }
#endif

    // ...
    int ret = convolutiondepthwise(bottom_blob_bordered, top_blob, weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, group, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}
```

## 卷积运算函数
卷积运算函数实现是两卷积算子的核心，主要流程是：
1. kernel offsets计算，统计卷积核每个像素相对于左上角原点的相对坐标
2. 卷积运算循环，convolution函数为outch×outh×inch×maxk的O(n^4)复杂度的for循环，convolutiondepthwise分两种情况，depth-wise为group×outh×inch×maxk的O(n^4)复杂度的for循环，否则为group×outch_g×outh×inch×maxk的O(n^5)复杂度的for循环。

```
// src/layer/convolution.cpp
static int convolution(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;
    const int bias_term = bias_data.empty() ? 0 : 1;
    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        // 滑窗相邻两行坐标gap
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data[p];

                // 输出坐标 (j * stride_w, i * stride_h) 映射至 输入坐标(j, i)
                const float* kptr = (const float*)weight_data + maxk * inch * p;

                for (int q = 0; q < inch; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[space_ofs[k]]; // 20.72
                        float wt = kptr[k];
                        sum += val * wt; // 41.45
                    }

                    kptr += maxk;
                }
                // 激活函数
                outptr[j] = activation_ss(sum, activation_type, activation_params);
            }

            outptr += outw;
        }
    }

    return 0;
}

// src/layer/convolutiondepthwise.cpp

static int convolutiondepthwise(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int group, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    // ...

    // depth-wise
    if (inch == group && group == outch)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            float* outptr = top_blob.channel(g);
            // weight_data尺寸 outch*1*kernel_h*kernel_w
            const float* kptr = (const float*)weight_data + maxk * g;
            const Mat m = bottom_blob.channel(g);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[g];

                    // 输出坐标 (j * stride_w, i * stride_h) 映射至 输入坐标(j, i)
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        float w = kptr[k];
                        sum += val * w;
                    }

                    // 激活函数
                    outptr[j] = activation_ss(sum, activation_type, activation_params);
                }

                outptr += outw;
            }
        }
    }
    else
    {
        // group convolution
        const int inch_g = inch / group;
        const int outch_g = outch / group;

#ifdef _WIN32
        #pragma omp parallel for num_threads(opt.num_threads)
#else
        #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif
        for (int g = 0; g < group; g++)
        {
            for (int p = 0; p < outch_g; p++)
            {
                float* outptr = top_blob.channel(g * outch_g + p);

                // weight_data尺寸 group*outch_g*inch_g*kernel_h*kernel_w
                const float* weight_data_ptr = (const float*)weight_data + maxk * inch_g * outch_g * g;

                // shadowed variable for less openmp task args
                const int outw = top_blob.w;
                const int outh = top_blob.h;

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                            sum = bias_data[outch_g * g + p];

                        const float* kptr = weight_data_ptr + maxk * inch_g * p;

                        for (int q = 0; q < inch_g; q++)
                        {
                            const Mat m = bottom_blob.channel(inch_g * g + q);
                            // 输出坐标 (j * stride_w, i * stride_h) 映射至 输入坐标(j, i)
                            const float* sptr = m.row(i * stride_h) + j * stride_w;
                            for (int k = 0; k < maxk; k++)
                            {
                                float val = sptr[space_ofs[k]];
                                float w = kptr[k];
                                sum += val * w;
                            }
                            kptr += maxk;
                        }

                        // 激活函数
                        outptr[j] = activation_ss(sum, activation_type, activation_params);
                    }
                    outptr += outw;
                }
            }
        }
    }

    return 0;
}
```

## Int8推理补充
相对与float推理，int8推理主要区别是在其中插入量化、反量化操作，包括：
1. 依据bottom_blob_int8_scales对输入数据量化；
2. 卷积乘法运算由两个float相乘变为两个signed char相乘，结果反量化为float；
3. 依据配置参数决定计算结果是否量化为int8形式作输出。

```
// src/layer/convolution.cpp

int Convolution::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // ...

    Mat bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;

        quantize_to_int8(bottom_blob, bottom_blob_unbordered, bottom_blob_int8_scales, opt_g);
    }

    // ...

    // int8
    bool use_int8_requantize = int8_scale_term > 100;
    size_t out_elemsize = use_int8_requantize ? 1u : 4u;

    top_blob.create(outw, outh, num_output, out_elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        signed char* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                int sum = 0;

                const signed char* kptr = (const signed char*)weight_data + maxk * channels * p;

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    const signed char* sptr = m.row<signed char>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        int val = sptr[space_ofs[k]];
                        int wt = kptr[k];
                        sum += val * wt;
                    }

                    kptr += maxk;
                }

                float scale_in;
                if (weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

                float sumfp32 = sum * scale_in;

                if (bias_term)
                    sumfp32 += bias_data[p];

                sumfp32 = activation_ss(sumfp32, activation_type, activation_params);

                if (use_int8_requantize)
                {
                    // requantize
                    float scale_out = top_blob_int8_scales[0];
                    signed char sums8 = float2int8(sumfp32 * scale_out);
                    outptr[0] = sums8;
                    outptr += 1;
                }
                else
                {
                    // dequantize
                    ((float*)outptr)[0] = sumfp32;
                    outptr += 4;
                }
            }
        }
    }

    return 0;
}
```