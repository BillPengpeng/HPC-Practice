NCNN中Deconvolution与DeconvolutionDepthWise原理类似，Deconvolution相当于group等于1情况下的DeconvolutionDepthWise。

## 配置项
两种卷积的配置项基本一致，ConvolutionDepthWise在Convolution参数基础上，额外增加group参数，并要求输出通道num_output可以整除group。

```
// src/layer/deconvolution.cpp

Deconvolution::Deconvolution()
{
    one_blob_only = true;
    support_inplace = false;
}

int Deconvolution::load_param(const ParamDict& pd)
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
    output_pad_right = pd.get(18, 0);
    output_pad_bottom = pd.get(19, output_pad_right);
    output_w = pd.get(20, 0);
    output_h = pd.get(21, output_w);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    dynamic_weight = pd.get(28, 0);

    if (dynamic_weight)
    {
        one_blob_only = false;
    }

    return 0;
}

// src/layer/deconvolutiondepthwise.cpp
int DeconvolutionDepthWise::load_param(const ParamDict& pd)
{
    // ...
    group = pd.get(7, 1);
    // ...
    return 0;
}
```

## 模型参数加载
两种反卷积的模型参数加载流程一致，依次加载weight数据、bias数据。

```
// src/layer/deconvolution.cpp

int Deconvolution::load_model(const ModelBin& mb)
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

    return 0;
}
```

## 前向推理
反卷积相当于卷积的逆过程，两种反卷积的前向推理主要流程有：
1. 计算输出blob的shape，并申请内存初始化；
2. 调用卷积运算主处理函数;
2. 对输出作cut_padding操作；

```
// src/layer/deconvolution.cpp
int Deconvolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    // 反卷积输出shape计算
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;
    int outh = (h - 1) * stride_h + kernel_extent_h + output_pad_bottom;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    int ret = deconvolution(bottom_blob, top_blob_bordered, weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    // 输出裁边操作
    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}
```

## 反卷积运算函数
反卷积运算函数实现是两卷积算子的核心，主要流程是：
1. kernel offsets计算，统计卷积核每个像素相对于左上角原点的相对坐标
2. 卷积运算循环，deconvolution函数为outch×输入h×输入w×inch×maxk的O(n^5)复杂度的for循环，deconvolutiondepthwise分两种情况，depth-wise为group×输入h×输入w×maxk的O(n^4)复杂度的for循环，否则为group×outch_g×输入h×输入w×inch×maxk的O(n^6)复杂度的for循环。

```
// src/layer/deconvolution.cpp
static int deconvolution(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int outw = top_blob.w;
    const int outch = top_blob.c;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = outw * dilation_h - kernel_w * dilation_w;
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
        Mat out = top_blob.channel(p);
        const float bias = bias_data.empty() ? 0.f : bias_data[p];

        out.fill(bias);

        // shadowed variable for less openmp task args
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int inch = bottom_blob.c;
        const int outw = top_blob.w;
        const int outh = top_blob.h;

        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                // 输入坐标(j, i) 映射至输出坐标 (j * stride_w, i * stride_h)
                float* outptr = out.row(i * stride_h) + j * stride_w;

                const float* kptr = (const float*)weight_data + maxk * inch * p;

                for (int q = 0; q < inch; q++)
                {
                    const float val = bottom_blob.channel(q).row(i)[j];
                    for (int k = 0; k < maxk; k++)
                    {
                        float w = kptr[k];
                        outptr[space_ofs[k]] += val * w;
                    }
                    kptr += maxk;
                }
            }
        }

        {
            float* outptr = out;
            // 每个通道输出尺寸
            int size = outw * outh;
            for (int i = 0; i < size; i++)
            {
                // 激活函数
                outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
            }
        }
    }
    return 0;
}

// src/layer/deconvolutiondepthwise.cpp

static int deconvolutiondepthwise(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int group, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int inch = bottom_blob.c;
    const int outw = top_blob.w;
    const int outch = top_blob.c;
    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = outw * dilation_h - kernel_w * dilation_w;
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

    // depth-wise
    if (inch == group && group == outch)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            const float* inptr = bottom_blob.channel(g);
            // weight_data尺寸 outch*1*kernel_h*kernel_w
            const float* kptr = (const float*)weight_data + maxk * g;
            Mat out = top_blob.channel(g);
            const float bias = bias_data.empty() ? 0.f : bias_data[g];
            out.fill(bias);

            // shadowed variable for less openmp task args
            const int w = bottom_blob.w;
            const int h = bottom_blob.h;
            const int outw = top_blob.w;
            const int outh = top_blob.h;

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    // 输入坐标(j, i) 映射至输出坐标 (j * stride_w, i * stride_h)
                    float* outptr = out.row(i * stride_h) + j * stride_w;
                    const float val = inptr[i * w + j];
                    for (int k = 0; k < maxk; k++)
                    {
                        float w = kptr[k];
                        outptr[space_ofs[k]] += val * w;
                    }
                }
            }
            {
                float* outptr = out;
                int size = outw * outh;
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
                }
            }
        }
    }
    else
    {
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
                Mat out = top_blob.channel(g * outch_g + p);
                
                // weight_data尺寸 group*outch_g*inch_g*kernel_h*kernel_w
                const float* weight_data_ptr = (const float*)weight_data + maxk * inch_g * outch_g * g;
                const float bias = bias_data.empty() ? 0.f : bias_data[g * outch_g + p];
                out.fill(bias);

                // shadowed variable for less openmp task args
                const int w = bottom_blob.w;
                const int h = bottom_blob.h;
                const int outw = top_blob.w;
                const int outh = top_blob.h;

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        float* outptr = out.row(i * stride_h) + j * stride_w;
                        const float* kptr = weight_data_ptr + maxk * inch_g * p;
                        for (int q = 0; q < inch_g; q++)
                        {
                            const float val = bottom_blob.channel(inch_g * g + q).row(i)[j];
                            for (int k = 0; k < maxk; k++)
                            {
                                outptr[space_ofs[k]] += val * kptr[k];
                            }
                            kptr += maxk;
                        }
                    }
                }
                {
                    float* outptr = out;
                    int size = outw * outh;
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
                    }
                }
            }
        }
    }
    return 0;
}
```
