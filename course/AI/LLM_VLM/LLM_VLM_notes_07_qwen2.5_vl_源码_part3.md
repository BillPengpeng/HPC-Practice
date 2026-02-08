本文主要整理qwen2.5_vl开源代码的主要内容。

## 3.0 - Qwen2_5_VisionPatchEmbed

Qwen2_5_VisionPatchEmbed类是 Qwen2.5-VL 模型中将图像或视频帧转换为视觉特征序列（Visual Tokens）的核心模块。其核心工作原理是将输入图像或视频在时间和空间维度上划分为规则的小块，并通过3D卷积将每个小块映射为一个特征向量。

```python
class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        # 这表示卷积核在时间、高度、宽度三个维度上的尺寸与步长完全相同，确保了无重叠的块提取。
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        # 将输入张量重塑为 (batch_size * num_patches, in_channels, temporal_patch_size, patch_size, patch_size)。这里的 num_patches是自动计算出的总块数。这一步的本质是将一个批次的多个样本的所有视觉块在批次维度上堆叠，以便进行并行卷积运算。
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        # 通过 .view(-1, self.embed_dim)去除冗余的维度，最终得到形状为 (batch_size * num_patches, embed_dim)的输出。这 num_patches个特征向量就是视觉输入对应的“视觉Token序列”，将被送入后续的Transformer层进行处理
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states
```

## 3.1 - Qwen2_5_VisionRotaryEmbedding & Qwen2_5_VLRotaryEmbedding

Qwen2_5_VisionRotaryEmbedding类是 Qwen2.5-VL 模型中实现旋转位置编码 (RoPE)​ 的核心组件，其作用是生成用于计算位置编码的基础频率张量。

Qwen2_5_VisionRotaryEmbedding的设计体现了 RoPE 的核心思想：将绝对位置信息通过旋转变换注入到注意力机制中，使注意力分数能够自动蕴含相对位置信息​。
- 多模态扩展 (MRoPE)：在 Qwen2.5-VL 中，这个基础的 1D RoPE 被扩展为多模态旋转位置编码。视觉数据（图像和视频）的位置是三维的 (temporal, height, width)。因此，模型会为每个维度独立生成类似 freqs的频率张量，并最终复合形成一个3D旋转位置编码。​ 
- 绝对时间对齐：对于视频，Qwen2.5-VL 的一个关键创新是将时间维度的位置索引与绝对时间戳（例如秒）对齐，而非简单的帧序号。这使得模型能理解不同帧率视频中的时间节奏和事件持续时间。

![3D mrope](https://pic1.zhimg.com/v2-b7658c166a07d63af591ab38875a493a_1440w.jpg)

```python
class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        # dim：代表位置编码的维度，通常对应于模型注意力机制中每个注意力头的维度。
        # theta：一个超参数，默认为 10000.0，用于控制频率的衰减速度。较大的 theta值会使频率随维度索引增加而衰减得更慢
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
```

```python
class Qwen2_5_VLRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen2_5_VLConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen2_5_VLConfig | None = None,
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

    # Ignore copy
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        # 目的：将基础的 inv_freq从形状 (head_dim//2,)扩展为 (3, batch_size, head_dim//2, 1)。
        # 意义：这里的维度 3直接对应位置ID的三个维度（时间T、高度H、宽度W）。这意味着模型会为每个维度独立计算一组旋转参数
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            # 这是最关键的步骤，它计算了 inv_freq和 position_ids的外积​ 。
            # 物理意义：对于序列中的每个位置 m和每个特征维度对 i，计算得出一个旋转相位 m * θ_i。这个相位决定了在该位置和该维度上，查询和键向量需要旋转多少角度
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

| 对比维度 | Qwen2_5_VLRotaryEmbedding | Qwen2_5_VisionRotaryEmbedding |
| :--- | :--- | :--- |
| **作用域与层级** | **语言模型（LLM）端**，处理融合后的多模态序列 | **视觉编码器（ViT）端**，处理原始的图像/视频块序列 |
| **核心目标** | 实现**多模态旋转位置编码（M-RoPE）**，统一处理文本、图像、视频的时空位置信息 | 为视觉输入生成基础的**1D序列位置编码** |
| **处理维度** | **3维**（时间、高度、宽度），支持绝对时间戳对齐 | **1维**（序列顺序） |
| **输入坐标** | 接收来自 `get_rope_index` 方法计算的复杂3D位置ID | 接收简单的1D序列长度 `seqlen` |
| **输出用途** | 用于语言模型的自注意力机制，使LLM能感知视觉特征的空间布局和视频时序 | 用于视觉编码器内部的自注意力机制，建立图像块间的初始空间关系 |

1.  **视觉特征提取**：输入图像或视频首先由视觉编码器（`Qwen2_5_VisionTransformerPretrainedModel`）处理。在此阶段，**`Qwen2_5_VisionRotaryEmbedding`** 首先工作，为分割后的图像块序列生成基础的1D位置编码，帮助视觉编码器初步理解图像块之间的相对位置。
2.  **特征融合与位置重标定**：视觉编码器输出的视觉特征与文本嵌入在语言模型中拼接成一个统一的序列。此时，**`Qwen2_5_VLRotaryEmbedding`** 开始发挥核心作用。它根据 `get_rope_index` 方法计算出的复杂3D位置ID（包含时间、高度、宽度信息），为整个融合序列生成位置编码。这使得语言模型不仅能理解文本的顺序，还能精确感知视觉特征在原始输入中的时空坐标。

## 3.2 - Qwen2_5_VisionTransformerPretrainedModel

Qwen2_5_VisionTransformerPretrainedModel是 Qwen2.5-VL 模型的视觉主干网络，其核心目标是将任意尺寸的图像或视频帧序列转换为一系列具有空间位置感知的视觉特征向量（Visual Tokens），为后续与语言模型的融合做准备。

该类通过一系列精心设计的组件，实现了对动态分辨率输入的原生支持。这意味着不同大小、不同长宽比的图像或视频，经过该编码器处理后，会产生数量不同但维度统一的视觉特征序列，而非传统ViT模型中的固定长度序列。

```python
class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
    config: Qwen2_5_VLVisionConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]
    _input_embed_layer = "patch_embed"
    _can_record_outputs = {
        "hidden_states": Qwen2_5_VLVisionBlock,
        "attentions": Qwen2_5_VLVisionAttention,
    }

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

        self.post_init()

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    # get_window_index方法是 Qwen2.5-VL 视觉编码器中实现窗口注意力（Window Attention）​ 的核心算法，其主要目标是将视觉特征序列智能地划分为局部窗口，并生成对应的索引，从而将全局注意力的计算复杂度从 O(N²)降低到 O(N)，这对于处理高分辨率图像至关重要
    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        # 计算在LLM（大语言模型）层面，每个注意力窗口应包含的基础视觉网格数量
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            # 计算LLM层面网格：根据 spatial_merge_size（空间合并因子，通常为2）对原始网格进行降采样，得到语言模型实际感知的网格尺寸 llm_grid_h和 llm_grid_w
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            # 窗口索引：index_new = index_padded[index_padded != -100]过滤掉填充值，得到真正需要处理的索引序列，并加上一个偏移量 window_index_id以确保不同输入图像的索引全局唯一
            window_index.append(index_new + window_index_id)
            # 计算每个窗口结束位置的累积索引
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        # window_index：一个张量，指示了原始特征序列中的每个元素，在经过窗口划分和重排后，应该位于新序列中的哪个位置。
        # cu_window_seqlens：一个列表，记录了每个窗口的序列长度累积和，用于后续高效的批量注意力计算
        return window_index, cu_window_seqlens

    @check_model_inputs
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        # 对于图像：将单张图像视为一个“2帧的短视频”，通过复制帧来适配3D卷积。
        # 卷积核：尺寸为 [temporal_patch_size, patch_size, patch_size]（例如 [2, 14, 14]）。卷积的步长与核大小相同，实现无重叠的块提取​ 。
        # 输出：将输入张量从 (B*T, C, H, W)转换为 (B*N, embed_dim)，其中 N是提取的块总数。这相当于将图像或视频在时空维度上网格化。
        hidden_states = self.patch_embed(hidden_states)

        # 该方法根据输入图像的实际网格划分​ (grid_thw) 生成精确的2D位置编码
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # 将全局的自注意力计算限制在一个个局部窗口内，大幅减少计算量
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        # 最终重塑：将重排后的张量恢复成 (seq_len, hidden_size)的形状，但此时序列的顺序已经按照窗口布局优化过了
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # 这行代码计算的是全局的累积序列长度​ cu_seqlens。它首先根据 grid_thw（每个视觉输入的网格信息）计算每个输入包含的块数量（grid_h * grid_w），然后通过 repeat_interleave和 cumsum进行累积求和
        # grid_thw[:, 0]（时间步数）是划分的主导维度。它直接决定了每个视觉输入的空间信息在最终序列中被复制的次数。一个时间为2帧的视频输入，其空间网格数就会被复制两次，从而在最终的 cu_seqlens中占据两个独立的位置范围。
        # cu_seqlens的最终长度等于 batch_size + 1，其中 batch_size在这里是所有时间步的总和（即 grid_thw[:, 0].sum()），而不是原始输入的数量。这正体现了其按时间步划分的本质。
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        # F.pad(cu_seqlens, (1, 0), value=0)在序列开头填充一个 0，使其变为 [0, 10, 25]。这个格式是许多标准注意力实现中用于标识批次内每个序列起始位置的常见格式
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)


        # 混合注意力策略：通过 fullatt_block_indexes配置，实现了全局注意力和窗口注意力的混合使用 。通常只有少数几层使用全局注意力以捕获远程依赖，大多数层使用高效的窗口注意力。
        # 动态控制：在 forward方法中，根据当前层编号决定使用 cu_seqlens（全局）还是 cu_window_seqlens（窗口）来指导注意力计算
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # 实现视觉Token压缩、减少后续语言模型计算负担的关键组件
        merged_hidden_states = self.merger(hidden_states)

        # 使用 torch.argsort(window_index)得到逆序索引，将因窗口划分而打乱的序列顺序恢复原状，确保输出的视觉特征序列与输入内容的空间位置对应关系是正确的。
        reverse_indices = torch.argsort(window_index)
        merged_hidden_states = merged_hidden_states[reverse_indices, :]

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )
```