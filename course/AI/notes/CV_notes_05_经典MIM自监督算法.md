本文主要整理经典的MIN自监督学习算法，主要应用于Transformer模型的预训练。

## MAE (2021)

<div align=center>
<img src="https://user-images.githubusercontent.com/30762564/150733959-2959852a-c7bd-4d3f-911f-3e8d8839fe67.png" width="80%"/>
</div>

### 1. **高比例掩码策略**
- **核心思想**：在训练过程中，随机掩盖输入图像中**75%以上的图像块**（patches），仅保留少量可见块作为上下文。
- **创新性**：
  - 远高于传统方法（如BERT的15%掩码率），迫使模型从极稀疏信息中学习全局语义推理能力。
  - 通过极端掩码比例，模型必须理解图像整体结构而非局部纹理，提升特征鲁棒性。

```
    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the mask for MAE Pre-training.

        Args:
            x (torch.Tensor): Image with data augmentation applied, which is
                of shape B x L x C.
            mask_ratio (float): The mask ratio of total patches.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked image, mask
            and the ids to restore original image.

            - ``x_masked`` (torch.Tensor): masked image.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
```

### 2. **非对称编码器-解码器架构**
- **编码器设计**：
  - 仅处理**未被掩码的可见块**，计算量大幅降低（如掩码75%时，编码器仅需处理25%的输入）。
  - 采用**Vision Transformer（ViT）**作为主干，将图像分块后编码为潜在表示。
```
        B = x.shape[0]
        x = self.patch_embed(x)[0]
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for _, layer in enumerate(self.layers):
            x = layer(x)
        # Use final norm
        x = self.norm1(x)

        return (x, mask, ids_restore)
```

- **解码器设计**：
  - 接收编码器的潜在表示与掩码标记（mask tokens），重建完整图像的像素值。
  - 解码器轻量化（如仅占模型总参数的10%），专注于重建任务。
```
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # add pos embed, different from that in encoder and is not learnable
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
```

- **优势**：编码器高效预训练，下游任务可仅用编码器，无需解码器。


### 3. **像素级重建任务**
- **训练目标**：直接预测被掩码区域的**原始像素值**（而非语义标签或特征向量）。
- **损失函数**：采用均方误差（MSE）计算掩码区域的像素重建误差：
  $$
  \mathcal{L} = \| x_{\text{masked}} - \hat{x}_{\text{masked}} \|_2^2
  $$
- **优势**：
  - 无需复杂的数据增强或对比学习策略，简化训练流程。
  - 重建任务迫使模型理解低级视觉特征（如形状、纹理）与高级语义的关联。


### 4. **基于ViT的长距离依赖建模**
- **块处理机制**：将图像分割为固定大小的块（如16×16像素），作为Transformer的输入序列。
- **全局注意力**：通过Transformer的自注意力机制，模型可捕捉图像块之间的长程依赖关系，增强对整体结构的理解。
- **兼容掩码策略**：掩码块的位置信息通过位置编码保留，避免空间信息丢失。


### 5. **高效计算与数据利用率**
- **计算优化**：
  - 编码器仅处理可见块，预训练速度提升3-4倍（相比处理全图）。
  - 解码器仅在预训练阶段使用，下游任务无需额外计算。
- **数据效率**：
  - 在ImageNet-1K上，MAE仅需30%的标注数据即可达到监督学习性能。
  - 在低数据场景（如1%标注）下，性能显著优于对比学习方法（如MoCo v3）。


### 6. **与传统自编码器的对比**
| **维度**         | **传统自编码器**               | **MAE**                        |
|------------------|-------------------------------|--------------------------------|
| **掩码比例**     | 低或无掩码                    | 高比例（75%以上）              |
| **计算效率**     | 全图处理，计算量大             | 编码器仅处理可见块，效率高      |
| **重建目标**     | 可能依赖特征级重建             | 直接像素级重建                 |
| **主干网络**     | CNN或浅层Transformer          | 深度Vision Transformer（ViT）  |


### 7. **影响与意义**
- **理论突破**：证明了掩码自编码器在视觉领域的有效性，填补了NLP与CV在自监督学习上的方法论鸿沟。
- **实用价值**：提供高效预训练方案，降低对标注数据的依赖，推动无监督学习在工业场景的应用。
- **启发后续工作**：如BEiT（引入语义码本）、CAE（上下文自编码器）等均基于MAE框架改进。

## BeiT (2022)

参考博文[如何评价微软提出的无监督视觉模型BEiT：ImageNet达到88.6，ADE20K达到57.0？](https://www.zhihu.com/question/478187326/answer/2666815139)。

<div align=center>
<img src="https://user-images.githubusercontent.com/36138628/203688351-adac7146-4e71-4ab6-8958-5cfe643a2dc5.png" width="70%"/>
</div>

BeiT（Bidirectional Encoder representation for Image Transformers）是一种基于**掩码图像建模**的自监督学习方法，其核心创新在于将自然语言处理中的掩码语言模型（如BERT）与图像离散表示结合，实现了高效的视觉特征学习。以下是其核心创新点概述：

### 1. **掩码图像建模（Masked Image Modeling, MIM）**
   - **任务设计**：  
     - 将输入图像划分为块（patches），随机掩码部分块（如掩码率40%），要求模型基于上下文预测被掩码区域的内容。  
     - 不同于MAE的直接像素重建，BeiT预测**离散视觉标记**（Discrete Visual Tokens），增强语义抽象能力。  
     - BEIT的MIM的输入是N个图像patches，输出是图被掩码的图像集合。
   - **优势**：  
     - 迫使模型理解全局与局部语义关系，而非低层次像素细节。  
     - 类似BERT的掩码预测机制，适用于视觉Transformer（ViT）的长程依赖建模。

```
    x, patch_resolution = self.patch_embed(x)

    # replace the masked visual tokens by mask_token
    B, L, _ = x.shape
    mask_token = self.mask_token.expand(B, L, -1)
    w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
    x = x * (1. - w) + mask_token * w
```

![参考图片](https://pica.zhimg.com/80/v2-14ecf3db24c190ab0759289656b5a6f3_720w.webp?source=2c26e567)

### 2. **视觉标记与码本（Visual Token & Codebook）**
   - **离散表示生成**：  
     - 使用预训练的**dVAE（Discrete Variational Autoencoder）**将图像块编码为离散标记（如8192类别的码本）。  
     - 码本中的每个标记对应一种视觉语义模式（如纹理、形状等）。  
```
target_generator=dict(
    type='DALL-E',
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/dalle_encoder.pth',  # noqa: E501
    ))
```
   - **目标重构**：  
     - 模型预测被掩码块对应的离散标记类别，而非原始像素值。  
     - 损失函数采用交叉熵：$\mathcal{L} = -\sum \log P(\text{Token} | \text{Context})$。  
   - **优势**：  
     - 避免像素级重建的计算冗余，提升语义表示的高效性。  
     - 码本提供紧凑的语义空间，增强模型对高层特征的捕捉能力。
```
    mask = torch.stack([data_sample.mask for data_sample in data_samples])

    img_latent = self.backbone(inputs[0], mask)

    # inputs[1] is the target image
    with torch.no_grad():
        target = self.target_generator(inputs[1])
        target = target.detach()

    if self.with_neck:
        # BEiT v2
        feats, feats_cls_pt = self.neck(
            img_latent, rel_pos_bias=self.backbone.shared_rel_pos_bias)
        loss = self.head.loss(feats, feats_cls_pt, target, mask)
    else:
        # BEiT v1
        loss = self.head.loss(img_latent[0], target, mask)
```

### 3. **两阶段预训练框架**
   - **阶段一：码本生成**  
     - 训练dVAE模型，学习图像块到离散标记的映射，构建视觉语义码本。  
   - **阶段二：掩码预训练**  
     - 冻结码本，使用ViT作为编码器，通过掩码预测任务学习图像表示。  
   - **优势**：  
     - 分离码本学习与特征学习，简化训练复杂度。  
     - 码本作为中间桥梁，连接低像素空间与高语义空间。


### 4. **与MAE的对比**
| **维度**         | **BeiT**                          | **MAE**                        |
|------------------|-----------------------------------|--------------------------------|
| **重建目标**     | 离散视觉标记（码本）              | 原始像素值                     |
| **训练阶段**     | 两阶段（码本训练 + 掩码预训练）   | 单阶段（端到端像素重建）        |
| **语义抽象**     | 高层语义（码本约束）              | 低层像素细节                   |
| **计算效率**     | 码本生成增加前期成本              | 编码器仅处理可见块，效率更高    |


## BeiT V2 (2022)

参考博文[图像预训练：BEiT v2](https://zhuanlan.zhihu.com/p/566511151)。

<div align=center>
<img src="https://user-images.githubusercontent.com/36138628/203912182-5967a520-d455-49ea-bc67-dcbd500d76bf.png" width="70%"/>
</div>

BEiT v2同BEiT v1一样，分为两阶段：第一阶段是VQ-KD的训练，第二阶段是预训练模型的训练。

### 1. **矢量量化-知识蒸馏（Vector-quantized Knowledge Distillation，VQ-KD）**

- 将原始图像作为输入，使用另外一个模型作为教师系统（Teacher）来引导视觉标志模型的训练。VQ-KD在这里重建的是教师系统编码的特征而非原始像素。
- 分为**Tokenizer以及Decoder**两部分，Tokenizer的计算分成两步：它首先使用ViT将输入图像编码成特征向量，然后使用从码本中查找最近的邻居；Tokenizer结果输入至Decoder中，得到最终输出。
- VQ-KD的损失函数可以表示最大化模型输出以及教师系统生成的特征相似度并最小化生成特征和视觉单词的距离。
- **码本衰减（Codebook Collapse）**，指的是码本中只有一少部分视觉单词被频繁使用，应对策略：使用了L2归一化进行码本查找；将查找空间降低到了32维，在特征被输入到解码器之前再映射回高维空间；使用滑动平均来进行码本更新。

![参考图片](https://pica.zhimg.com/80/v2-14ecf3db24c190ab0759289656b5a6f3_720w.webp?source=2c26e567)

### 2. **BEiT v2预训练**

- 为了学习图像的全局信息，BEiT v2在输入编码中拼接了[CLS]标志，然后通过对[CLS]标志的预训练来得到图像的全局信息。因此BEiT v2的预训练分成两个部分：分别是掩码图像模型的训练和[CLS]标志的训练。

```
    # 拼接CLS和浅层特征
    early_states, x = inputs[0], inputs[1]
    x_cls_pt = torch.cat([x[:, [0]], early_states[:, 1:]], dim=1)
    for layer in self.patch_aggregation:
        x_cls_pt = layer(x_cls_pt, rel_pos_bias=rel_pos_bias)

    # shared norm
    x, x_cls_pt = self.norm(x), self.norm(x_cls_pt)

    # remove cls_token
    x = x[:, 1:]
    x_cls_pt = x_cls_pt[:, 1:]
    return x, x_cls_pt
```

```
    # 两个不同的MIM任务共享同一个MIM输出头
    if self.with_neck:
        # BEiT v2
        feats, feats_cls_pt = self.neck(
            img_latent, rel_pos_bias=self.backbone.shared_rel_pos_bias)
        loss = self.head.loss(feats, feats_cls_pt, target, mask)
    else:
        # BEiT v1
        loss = self.head.loss(img_latent[0], target, mask)
```

## SimMIM (2022)

参考博文[如何评价微软亚洲研究院新提出的MIM方法：SimMIM？](https://www.zhihu.com/question/500105161/answer/2267169645)。

<div align=center>
<img src="https://user-images.githubusercontent.com/30762564/159404597-ac6d3a44-ee59-4cdc-8f6f-506a7d1b18b6.png" width="70%"/>
</div>

SimMIM（Simple Masked Image Modeling）是一种基于掩码图像建模的自监督学习方法，其核心创新在于通过**极简设计**和**高效实现**，验证了掩码预训练任务本身的有效性，而非依赖复杂的模型架构或训练策略。

SimMIM和MAE的区别主要有以下两点：
- SimMIM的encoder同时处理visible tokens和masked tokens，而MAE的encoder只处理visible tokens。
- SimMIM的decoder只采用一个线性层来回归像素值，而MAE的decoder采用transformer结构。


### 1. **极简的像素级重建任务**
   - **直接回归像素值**：  
     - 与BeiT的离散码本或MAE的像素重建不同，SimMIM直接预测被掩码区域的**原始RGB像素值**，无需引入额外模块（如dVAE或复杂解码器）。  
     - 损失函数采用**L1损失**（而非L2），对异常值更鲁棒：  
       $$
       \mathcal{L} = \frac{1}{N} \sum_{i=1}^N \| x_i - \hat{x}_i \|
       $$
   - **优势**：简化训练流程，降低实现复杂度。


### 2. **标准ViT架构的适配性**
   - **无需非对称设计**：  
     - 使用**完整的标准ViT**（无编码器-解码器分离），所有层参与掩码块和可见块的特征提取。  
     - 仅通过掩码token（可学习向量）替换被掩码的块，保持模型结构不变。  
   - **优势**：证明MIM的有效性不依赖于特殊架构设计，可直接迁移至现有ViT模型。


### 3. **高效的高比例随机掩码**
   - **高掩码比例**：  
     - 默认掩码率为**60%-80%**，高于MAE（75%），但通过实验验证其有效性。  
     - 采用**随机块掩码**（如32×32像素块），无需复杂的分层或语义引导策略。  
   - **优势**：减少计算量（仅处理部分块），同时迫使模型学习全局上下文推理能力。


### 4. **轻量级预测头设计**
   - **单层线性投影**：  
     - 在ViT输出后仅添加一个**线性层**，将特征映射到像素空间，而非多层MLP或Transformer解码器。  
   - **优势**：  
     - 参数量减少90%以上（相比MAE的解码器）。  
     - 推理时可直接移除预测头，下游任务仅保留ViT主干。


### 5. **训练效率优化**
   - **快速收敛**：  
     - 在ImageNet-1K上，仅需300 epoch预训练即可达到监督学习基线性能（ViT-B为83.8% Top-1）。  
   - **低资源适配**：  
     - 在较小ViT模型（如ViT-Tiny）上仍有效，验证方法的普适性。


### 6. **与MAE、BeiT的对比**
| **维度**         | **SimMIM**                      | **MAE**                        | **BeiT**                       |
|------------------|---------------------------------|--------------------------------|--------------------------------|
| **重建目标**     | 原始像素值（L1损失）            | 原始像素值（L2损失）           | 离散视觉标记（交叉熵损失）      |
| **架构设计**     | 标准ViT + 单层线性头            | 非对称编码器-解码器            | 标准ViT + 码本生成器           |
| **掩码策略**     | 高比例随机块掩码（60%-80%）     | 高比例随机块掩码（75%）        | 随机块掩码（40%）              |
| **计算开销**     | 极低（无复杂解码器）            | 中等（轻量解码器）             | 高（两阶段码本训练）           |

## MixMIM (2022)

MixMIM（Mixed and Masked Image Modeling）是一种基于**混合和掩码**的自监督学习方法。

<div align=center>
<img src="https://user-images.githubusercontent.com/56866854/202853730-d26fb3d7-e5e8-487a-aad5-e3d4600cef87.png"/>
</div>

### 1. ‌**混合图像双重重建机制‌‌**

- ‌跨图像可见标记融合‌：通过混合两幅图像的可见标记（visible tokens）生成训练输入，替代传统MIM方法中单一的掩码符号（[MASK]）填充，避免预训练与微调阶段输入分布不一致的问题。
- 双重像素级重建目标‌：要求模型同时恢复原始两幅图像的像素信息，增强对全局语义关系和局部细节的建模能力，解决传统方法仅重建单图的局限性。

```
def loss(self, x_rec: torch.Tensor, target: torch.Tensor,
            mask: torch.Tensor) -> torch.Tensor:
    """Generate loss.

    Args:
        pred (torch.Tensor): The reconstructed image.
        target (torch.Tensor): The target image.
        mask (torch.Tensor): The mask of the target image.

    Returns:
        torch.Tensor: The reconstruction loss.
    """
    target = self.construct_target(target)

    B, L, C = x_rec.shape

    # unmix tokens
    x1_rec = x_rec[:B // 2]
    x2_rec = x_rec[B // 2:]

    unmix_x_rec = x1_rec * mask + x2_rec.flip(0) * (1 - mask)

    loss_rec = self.loss_module(unmix_x_rec, target)

    return loss_rec
```

### 2. ‌**高效训练策略优化‌‌**

- 消除掩码符号计算冗余‌：传统MIM需保留大量掩码符号参与计算，而MixMIM仅对混合后的可见标记进行编码，显著降低训练时的显存占用和计算成本。

```
if mask is None or False:
    return super().forward(x)

else:
    mask_s1, mask_s2, mask_s3, mask_s4 = self.random_masking(
        x, self.mask_ratio)

    x, _ = self.patch_embed(x)

    x = x * (1. - mask_s1) + x.flip(0) * mask_s1
    x = x + self.absolute_pos_embed
    x = self.drop_after_pos(x)

    for idx, layer in enumerate(self.layers):
        if idx == 0:
            x = layer(x, attn_mask=mask_s1)
        elif idx == 1:
            x = layer(x, attn_mask=mask_s2)
        elif idx == 2:
            x = layer(x, attn_mask=mask_s3)
        elif idx == 3:
            x = layer(x, attn_mask=mask_s4)

    x = self.norm(x)

    return x, mask_s4
```

## MFF (2023)

参考博文[MFF：用于缓解 Pixel-based MIM 方法过度依赖低水平特征问题的特征融合策略](https://zhuanlan.zhihu.com/p/649288830)。

<div align=center>
<img src="https://user-images.githubusercontent.com/30762564/257412932-5f36b11b-ee64-4ce7-b7d1-a31000302bd8.png" width="80%"/>
</div>

论文通过实验论证MIM任务更偏向于低水平的细节，提出直接引入从浅层获取的低水平特征，所提出的MFF模块可以轻松地整合到许多现有的MIM方法中，是一种即插即用的方案。


```
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
        if mask is None or False:
            return super().forward(x)

        else:
            B = x.shape[0]
            x = self.patch_embed(x)[0]
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            res = []
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i in self.out_indices:
                    if i != self.out_indices[-1]:
                        proj_x = self.proj_layers[self.out_indices.index(i)](x)
                    else:
                        proj_x = x
                    res.append(proj_x)
            res = torch.stack(res)
            proj_weights = F.softmax(self.proj_weights, dim=0)
            res = res * proj_weights
            res = res.sum(dim=0)

            # Use final norm
            x = self.norm1(res)
            return (x, mask, ids_restore, proj_weights.view(-1))
```