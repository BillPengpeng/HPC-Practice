NCNN模型加载主要通过类ncnn::Net的接口load_param/load_model，squeezenet.cpp示例如下。

```
// examples/squeezenet.cpp

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;

    squeezenet.opt.use_vulkan_compute = true;

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    if (squeezenet.load_param("squeezenet_v1.1.param"))
        exit(-1);
    if (squeezenet.load_model("squeezenet_v1.1.bin"))
        exit(-1);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}
```

## load_param
param文件描述神经网络的结构，包括层名称，层输入输出信息，层参数信息，queezenet_v1.1.param示例如下，其中：

第一行：7767517，通过这个数来区分版本信息。

第二行：75 83，75对应layer_count，83对应blob_count。

第三行及后续行：层类型、层名、层输入blob数量，层输出blob数量，层输入blob名称、输出blob名称、层特有参数。


```
// examples/squeezenet_v1.1.param
7767517
75 83
Input            data             0 1 data 0=227 1=227 2=3
Convolution      conv1            1 1 data conv1 0=64 1=3 2=1 3=2 4=0 5=1 6=1728
ReLU             relu_conv1       1 1 conv1 conv1_relu_conv1 0=0.000000
Pooling          pool1            1 1 conv1_relu_conv1 pool1 0=0 1=3 2=2 3=0 4=0
Convolution      fire2/squeeze1x1 1 1 pool1 fire2/squeeze1x1 0=16 1=1 2=1 3=1 4=0 5=1 6=1024
ReLU             fire2/relu_squeeze1x1 1 1 fire2/squeeze1x1 fire2/squeeze1x1_fire2/relu_squeeze1x1 0=0.000000
Split            splitncnn_0      1 2 fire2/squeeze1x1_fire2/relu_squeeze1x1 fire2/squeeze1x1_fire2/relu_squeeze1x1_splitncnn_0 fire2/squeeze1x1_fire2/relu_squeeze1x1_splitncnn_1
```

Convolution的0=64 1=3 2=1 3=2 4=0 5=1 6=1728，0=输出通道num_output 1=卷积核大小kernel_w 2=核膨胀dilation_w 3=步长stride_w 4=扩边pad_left 5=是否存在偏置bias_term 6=权重大小weight_data_size，此处1728=64*3*3*3，具体参数含义见Convolution.cpp。注意，此处未标识输入通道数量num_input，初始化时由weight_data_size、num_output、kernel_w、kernel_h计算所得;
```
// src/layer/convolution.cpp
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
```

ReLU的0=0.000000对应参数slope，当slope为为0时，NCNN的ReLU算子相当于Leaky ReLU。
```
// src/layer/relu.cpp
int ReLU::load_param(const ParamDict& pd)
{
    slope = pd.get(0, 0.f);
    return 0;
}
```

Pooling的0=0 1=3 2=2 3=0 4=0，0=池化类型（PoolMethod_MAX/PoolMethod_AVE） 1=卷积核大小 2=步长stride 3=padding 4=全局池化与否。另外，pad_mode扩边类型有full padding、valid padding、[SAME_UPPER、SAME_LOWER](https://onnx.ai/onnx/operators/onnx__MaxPool.html)，区别是扩边方向。
```
// src/layer/pooling.cpp
int Pooling::load_param(const ParamDict& pd)
{
    pooling_type = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    stride_w = pd.get(2, 1);
    stride_h = pd.get(12, stride_w);
    pad_left = pd.get(3, 0);
    pad_right = pd.get(14, pad_left);
    pad_top = pd.get(13, pad_left);
    pad_bottom = pd.get(15, pad_top);
    global_pooling = pd.get(4, 0);
    pad_mode = pd.get(5, 0);
    avgpool_count_include_pad = pd.get(6, 0);
    adaptive_pooling = pd.get(7, 0);
    out_w = pd.get(8, 0);
    out_h = pd.get(18, out_w);

    return 0;
}
```

Concat/Softmax的0=0对应参数axis。
```
// src/layer/concat.cpp
int Concat::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    return 0;
}

// src/layer/softmax.cpp
int Softmax::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    // the original softmax handle axis on 3-dim blob incorrectly
    // ask user to regenerate param instead of producing wrong result
    int fixbug0 = pd.get(1, 0);
    if (fixbug0 == 0 && axis != 0)
    {
        NCNN_LOGE("param is too old, please regenerate!");
        return -1;
    }

    return 0;
}
```

## load_model
bin文件则记录各个算子的具体数据信息（比如卷积层的权重、偏置信息等）。

典型的Convolution依次加载weight_data、bias_data、weight_data_int8_scales、bottom_blob_int8_scales、top_blob_int8_scales。

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
```

这里的mb.load有两个参数：w表示需要读取的元素个数、type表示是否去要按照不同的精度读取，当type为０的时候，需要读取数据的头部４个字节，然后由头部四个字节的数据来决定按照什么精度读取；当type为1的时候，则不读取头部的四个字节，直接按照默认的float精度(4字节)去读。

```
// src/modelbin.cpp
Mat ModelBinFromDataReader::load(int w, int type) const
{
    Mat m;

    if (type == 0)
    {
        size_t nread;

        union
        {
            struct
            {
                unsigned char f0;
                unsigned char f1;
                unsigned char f2;
                unsigned char f3;
            };
            unsigned int tag;
        } flag_struct;

        nread = d->dr.read(&flag_struct, sizeof(flag_struct));
        // ...

        unsigned int flag = (int)flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

        if (flag_struct.tag == 0x01306B47)
        {
            // half-precision data
            size_t align_data_size = alignSize(w * sizeof(unsigned short), 4);

#if !__BIG_ENDIAN__
            // try reference data
            const void* refbuf = 0;
            nread = d->dr.reference(align_data_size, &refbuf);
            if (nread == align_data_size)
            {
                m = Mat::from_float16((const unsigned short*)refbuf, w);
            }
            else
#endif
            {
                std::vector<unsigned short> float16_weights;
                float16_weights.resize(align_data_size);
                nread = d->dr.read(&float16_weights[0], align_data_size);
                // ...
            }

            return m;
        }
        else if (flag_struct.tag == 0x000D4B38)
        {
            // int8 data
            size_t align_data_size = alignSize(w, 4);

            // ...
            const void* refbuf = 0;
            nread = d->dr.reference(align_data_size, &refbuf);
            // ...

            return m;
        }
        else if (flag_struct.tag == 0x0002C056)
        {
            // ...
            const void* refbuf = 0;
            nread = d->dr.reference(w * sizeof(float), &refbuf);
            // ...
        }

        if (flag != 0)
        {
            // ...
            float quantization_value[256];
            nread = d->dr.read(quantization_value, 256 * sizeof(float));
            // ...

            nread = d->dr.read(&index_array[0], align_weight_data_size);
            // ...
        }
        else if (flag_struct.f0 == 0)
        {
            // ...
            const void* refbuf = 0;
            nread = d->dr.reference(w * sizeof(float), &refbuf);
            // ...
        }

        return m;
    }
    else if (type == 1)
    {
        // ...
        const void* refbuf = 0;
        size_t nread = d->dr.reference(w * sizeof(float), &refbuf);
        // ...
    }
    else
    {
        // ...
    }

    return Mat();
}
```



