本文主要整理qwen2.5_vl开源代码的主要内容。

## 2.0 - Qwen2_5_VLProcessor

Qwen2_5_VLProcessor继承自 ProcessorMixin，遵循 Hugging Face 处理器设计范式，核心目标是将 图像/视频处理器​ 与 文本分词器​ 封装成单一接口，简化多模态输入的处理流程。

```python
class Qwen2_5_VLProcessor(ProcessorMixin):

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]

    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        # 占位标记
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:

        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # 图像和视频数据统一由 self.image_processor处理。该处理器会进行动态分辨率调整、归一化等操作，并返回一个关键信息：grid_thw（即 (grid_t, grid_h, grid_w)）。这表示视觉数据在时间、高度和宽度维度上被划分成的网格数，决定了视觉信息将被编码成多少个特征向量（视觉Token）
        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        # 主要负责将视频数据转换为模型能够理解的时序特征
        if videos is not None:
            # 通过传入videos=videos参数，处理器会对视频进行帧采样、尺寸调整和归一化等操作，并将视频数据转换成一系列的视觉特征（patches）
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["images_kwargs"])
            # video_grid_thw是一个列表，其长度等于批量处理中视频的数量。列表中的每个元素是一个元组 (grid_t, grid_h, grid_w)，分别代表了单个视频在时间（T）、高度（H）​ 和宽度（W）​ 维度上被划分成的网格数。例如，(grid_t, grid_h, grid_w) = (8, 20, 36)表示该视频在时间维度被分成8段，空间上被划分为20x36的网格。grid_t直接关联到模型最终用于理解视频的视觉token数量
            video_grid_thw = videos_inputs["video_grid_thw"]

            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            # second_per_grid_ts，即每个时间网格所代表的实际秒数
            # 当 fps是单个数字（如 2.0）时：对所有视频片段采用统一的计算方式 self.image_processor.temporal_patch_size / fps。这里的 temporal_patch_size是模型的一个固有参数（通常为2），表示将多少帧合并为一个时间网格。假设 fps=2，则每个时间网格代表 2 / 2 = 1秒的视频内容
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            # 最终，包含 pixel_values_videos（视频像素数据）、video_grid_thw（空间时间网格信息）和 second_per_grid_ts（时间网格秒数）的 videos_inputs字典会与其他模态的特征合并，输入给下游的大语言模型（LLM）
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        # 计算了空间合并因子。在Qwen2.5-VL中，视觉编码器会将图像或视频帧划分为多个小块（patches），为了减少输入语言模型的视觉特征数量，常常将空间上相邻的多个小块（例如 2x2=4个）的特征合并后再送入语言模型。merge_length就是这个合并后的基本单位
        # 其核心目的是为了确保文本序列中的视觉标记数量与视觉编码器实际输出的特征向量数量实现精确的、动态的对齐。
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            # 第一步：临时扩展。使用 while循环查找文本中的每一个视觉占位符（如 <|image_pad|>），并将其替换为相应数量的临时标记 <|placeholder|>。例如，如果计算出的视觉标记数量是1440，就替换为1440个连续的 <|placeholder|>。
            # 第二步：最终定型。在遍历并替换完所有占位符后，再执行 text[i].replace("<|placeholder|>", self.image_token)，将所有临时标记一次性替换回标准的视觉标记。这确保了最终文本序列中视觉标记的数量与视觉编码器输出的特征向量数量完全一致
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # 特征合并与返回
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

    # batch_decode、decode和 post_process_image_text_to_text方法将模型生成的 token ID 转换回可读文本
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
        return names_from_processor + ["second_per_grid_ts"]
```

## 2.1 - Qwen2VLImageProcessor

Qwen2VLImageProcessor继承自 BaseImageProcessor，其主要目标是将任意尺寸的图像/视频输入，动态且规范地转换为模型可处理的格式。它通过一系列预处理步骤，确保视觉数据在送入视觉编码器（Vision Transformer）之前，满足其特定的输入要求，尤其是支持 Qwen2-VL 强调的原生动态分辨率特性。

```python
class Qwen2VLImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        self.do_convert_rgb = do_convert_rgb

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        images = make_list_of_images(images)

        # 基础图像处理：包括转换为RGB格式、转换为Numpy数组、动态尺寸调整（smart_resize，确保尺寸是patch_size * merge_size的倍数且像素总数在范围内）、像素值缩放（通常从[0,255]缩放到[0,1]）和归一化
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats = np.repeat(patches[-1][np.newaxis], self.temporal_patch_size - 1, axis=0)
            patches = np.concatenate([patches, repeats], axis=0)
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        # 通过一系列精妙的 reshape和 transpose操作，将图像数据重新组织。其核心目标是确保相邻的2x2图像块（由 merge_size=2定义）在最终展平的序列中是连续的。这为后续视觉编码器中通过MLP将4个（2x2）相邻小块特征合并成1个视觉Token做好了准备
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        # pixel_values：形状：(num_patches, patch_dim)。
        # num_patches是总块数，计算公式为 grid_t * grid_h * grid_w。
        # patch_dim是每个图像块展平后的维度，计算公式为 channel * temporal_patch_size * patch_size * patch_size（例如 3 * 2 * 14 * 14 = 1176）。
        # 每一行代表一个图像块的原始像素信息。
        # image_grid_thw：形状：(num_images, 3)。
        # 这是一个重要的元信息，记录了每个视觉输入在时间（T）、高度（H）、宽度（W）​ 三个维度上被划分的网格数 (grid_t, grid_h, grid_w)
        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = make_flat_list_of_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for image in images:
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}

        if videos is not None:
            pixel_values, vision_grid_thws = [], []
            for images in videos:
                patches, video_grid_thw = self._preprocess(
                    images,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(video_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {"pixel_values_videos": pixel_values, "video_grid_thw": vision_grid_thws}

        return BatchFeature(data=data, tensor_type=return_tensors)
```