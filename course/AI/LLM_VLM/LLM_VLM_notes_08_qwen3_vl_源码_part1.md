本文主要整理qwen3_vl开源代码的主要内容。

## 1 - extract_vision_info

```python
def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
) -> Tuple[Optional[List[Image.Image]], Optional[List[Union[torch.Tensor, List[Image.Image]]]], Optional[Dict[str, Any]]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True,
                        image_patch_size=image_patch_size, return_video_metadata=return_video_metadata)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    video_kwargs = {'do_sample_frames': False}
    if not return_video_metadata: # BC for qwen2.5vl
        video_kwargs.update({'fps': video_sample_fps_list})

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
    return image_inputs, video_inputs
```

`process_vision_info` 函数是 Qwen3-VL 多模态模型处理视觉输入（图像和视频）的核心入口。新增的 `return_video_metadata` 和 `image_patch_size` 参数，主要是为了**增强视频处理能力**和**提供更精细的图像控制**，使其能更好地适应复杂的多模态场景。下面这个流程图直观展示了该函数如何处理这些参数及视觉信息：

#### 1. `return_video_metadata`：视频元数据控制
这个参数决定了函数是否返回详细的视频元数据，用于**更精细的视频时序理解**。

- **当 `return_video_metadata=True` 时**：
    - 函数会保留并返回视频的元数据（如帧率 `fps`、采样后的帧索引 `frame_indices`、总帧数 `total_num_frames`）。
    - 这些元数据对于模型理解视频的时间动态至关重要，例如，将视觉特征与音频或字幕在时间线上对齐，或进行精确的时间定位（Temporal Grounding）。
    - 在代码中，此参数为 `True` 时，`video_kwargs` 字典不会用 `fps` 列表更新，暗示元数据将通过其他方式（如返回的 `video_inputs`）传递。

- **当 `return_video_metadata=False` 时（默认或为兼容性）**：
    - 代码执行 `video_kwargs.update({'fps': video_sample_fps_list})`。这是为了**向后兼容 Qwen2.5-VL** 的接口，当时可能仅将帧率信息通过 `video_kwargs` 传递。
    - 此时，更丰富的元数据（如具体帧索引）可能不会被返回或使用。

#### 2. `image_patch_size`：视觉编码基础
这个参数是 Vision Transformer (ViT) 模型的**核心配置**，它定义了模型如何“看”图像和视频的每一帧。

- **作用机制**：ViT 不像卷积神经网络那样直接处理整张图片，而是先将图像分割成多个大小相等的方块（Patch），每个 Patch 的大小就是 `image_patch_size`（例如 14x14 或 16x16 像素）。每个 Patch 会被展平并投影为一个特征向量，相当于一个“视觉词汇”（Vision Token）。
- **影响下游处理**：
    - **动态分辨率调整**：在 `fetch_image` 和 `fetch_video` 函数内部，输入图像或视频帧的尺寸会被调整，以确保其高和宽都是 `image_patch_size` 的整数倍。这是为了确保图像能被完整且整齐地分割，没有残缺的 Patch。
    - **决定视觉 Token 数量**：一张图像的视觉 Token 数量计算公式为：`(高度 / image_patch_size) * (宽度 / image_patch_size)`。因此，`image_patch_size` 直接决定了送入语言模型的视觉信息量。较小的 `patch_size` 会产生更多、更细粒度的 Token，可能带来更好细节但计算量更大；较大的 `patch_size` 则相反。

## 2.0 - Qwen3VLTextRotaryEmbedding

```python
class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3VLTextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        # 多模式RoPE支持：通过 rope_type参数支持多种RoPE变体（如"default", "linear", "dynamic", "yarn"等），默认使用标准实现。
        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        # freqs是一个形状为 (3, batch_size, seq_len, head_dim//2)的张量。
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        
        # 多模态分段配置：mrope_section = [24, 20, 20]是关键创新，它将位置编码的维度划分为三个部分，分别对应时间（Temporal）、高度（Height）、宽度（Width）​ 三个维度，这是处理视频和图像空间信息的基础
        self.mrope_section = config.rope_parameters.get("mrope_section", [24, 20, 20])

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen3VLTextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        # 扩展频率和位置ID为三维
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            # 虽然使用相同的基础频率，但乘以不同的位置坐标后，产生了完全不同的频率模式
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            # 计算频率和应用交错MRoPE
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    # freqs：是一个形状为 (3, bs, seq_len, head_dim//2)的张量，其中第一维的3个元素分别对应时间（T）、高度（H）、宽度（W）​ 三个维度预先计算好的旋转频率
    # mrope_section：通常为 [24, 20, 20]，定义了每个物理维度（T, H, W）在嵌入空间中的分配比例。例如，24+20+20=64，对应标准的64维注意力头
    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = freqs[0]  # just overwrite the first dimension T
        # 维度遍历：依次处理高度（dim=1）和宽度（dim=2）两个空间维度
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            # 长度计算：length = mrope_section[dim] * 3确定当前维度需要处理的索引范围
            length = mrope_section[dim] * 3
            # 交错索引：slice(offset, length, 3)创建等间隔的切片索引：
            # 高度维度：offset=1 → 索引 1, 4, 7, 10...
            # 宽度维度：offset=2 → 索引 2, 5, 8, 11...
            idx = slice(offset, length, 3)
            # 重组前：[TTT...TTT][HHH...HHH][WWW...WWW]（连续分块）
            # 重组后：[THW][THW][THW]...[THW][TT]（交错循环）

            # [ T0,  H0,  W0,  T1,  H1,  W1,  T2,  H2,  W2,  T3,  H3,  W3 ]
            # 低频←───────────────────────┼───────────────────────→高频
            # 在 [T, H, W, T, H, W, ...]的交错排列中，每个维度内部仍然保持从低频到高频的顺序。
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t
```

## 2.1 - Qwen3VLVisionModel

```python
class Qwen3VLVisionModel(Qwen3VLPreTrainedModel):
    config: Qwen3VLVisionConfig
    _no_split_modules = ["Qwen3VLVisionBlock"]
    _can_record_outputs = {
        "hidden_states": Qwen3VLVisionBlock,
        "attentions": Qwen3VLVisionAttention,
    }

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config=config,
        )

        # pos_embed：为每个 patch 学习一个位置向量（类似 ViT 的绝对位置编码）。
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        # num_grid_per_side：表示图像被切成的 patch 网格边长，例如 14×14=196 个 patch。
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        # Qwen3 使用 旋转式位置编码（RoPE）。这里每个 attention head 的维度是 head_dim，RoPE 使用一半维度进行旋转位置编码。可让模型学习相对位置关系（而不仅是绝对坐标）。
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        # use_postshuffle_norm=False 表示此合并层不执行后归一化步骤
        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

        self.post_init()

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    # 快速位置嵌入插值 (fast_pos_embed_interpolate)
    # 该方法实现了可学习位置嵌入的双线性插值，支持任意尺寸的输入
    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    @check_model_inputs
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # 更新deepstack_feature_lists
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )
```

## 2.2 - Qwen3VLVisionPatchMerger

```python
class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use_postshuffle_norm=False 表示此合并层不执行后归一化步骤
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x
```