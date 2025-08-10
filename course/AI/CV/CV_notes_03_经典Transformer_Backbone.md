本文持续总结典型的有监督训练范式的Transformer Backbone。

## ViT（2021）

Vision Transformer（ViT）是一种将Transformer架构引入计算机视觉任务的里程碑式模型，其核心创新在于完全摒弃传统卷积操作，通过全局自注意力机制实现图像理解。

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142579081-b5718032-6581-472b-8037-ea66aaa9e278.png" width="70%"/>
</div>

### **1. ViT基础架构**
ViT将图像分割为固定大小的图块（Patches），通过线性投影转换为序列输入，随后应用标准Transformer编码器进行特征提取。其核心组件包括：

#### **(1) 输入处理**
- **图像分块**：将输入图像（如224×224）分割为 $ P \times P $ 的图块（如16×16），得到 $ N = (224/16)^2 = 196 $ 个图块。
- **线性投影**：每个图块展平后（16×16×3=768维）通过 `nn.Linear` 映射到隐藏维度 $ D $（如768）。
- **Class Token**：在序列头部添加可学习的分类标记（Class Token），用于最终分类。
- **位置编码**：叠加可学习的1D位置编码（形状为 $ (N+1) \times D $）。

**输入序列构建**：  
$$
X = [x_{\text{class}}; x_{\text{patch}}^1; \dots; x_{\text{patch}}^N] + E_{\text{pos}}
$$

```
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, hidden_dim=768):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Linear(patch_size**2 * 3, hidden_dim)  # 3为RGB通道
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))  # 可学习位置编码

    def forward(self, x):
        # 分块并投影
        B, C, H, W = x.shape
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(B, -1, patch_size**2 * 3)
        patch_embeddings = self.patch_embed(x)  # (B, num_patches, D)

        # 添加 [class] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)  # (B, num_patches+1, D)

        # 添加位置编码
        embeddings += self.pos_embed  # 广播相加
        return embeddings
```

#### **(2) Transformer编码器**
由 $ L $ 层相同的Transformer层堆叠而成，每层包含：

##### **多头自注意力（MSA）**
- **输入**：形状为 `(B, N, D)`，其中 `N = num_patches + 1`，`D = hidden_dim`。
- **过程**：
  1. **线性投影**：将输入分为Q、K、V，每个头维度为 `D // num_heads`。
  2. **缩放点积注意力**：计算每个头的注意力分数。
  3. **多头拼接**：拼接所有头的输出并通过线性层融合。
- **公式**：
  $$
  \text{MSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
  $$
  $$
  \text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)
  $$

``` 
// MSA代码示例
def forward(self, x):
    B, N, _ = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                              self.head_dims).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn_drop = self.attn_drop if self.training else 0.
    x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
    x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

    x = self.proj(x)
    x = self.out_drop(self.gamma1(self.proj_drop(x)))

    if self.v_shortcut:
        x = v.squeeze(1) + x
    return x
```

##### **前馈神经网络（FFN）**
- **结构**：两层的MLP，中间扩展维度（通常为 `4×hidden_dim`）。
- **公式**：
  $$
  \text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
  $$
- **参数示例**（ViT-Base）：
  - 输入维度：768
  - 中间层维度：3072
  - 输出维度：768

##### 层归一化与残差连接**
- **Pre-LayerNorm**：在MSA和FFN之前应用层归一化（与原始Transformer的Post-LN不同）。
- **残差连接**：每个子层（MSA、FFN）的输出与输入相加。

```
def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = self.ffn(self.ln2(x), identity=x)
    return x
```

#### **(3) 分类头**
- **Class Token提取**：取最后一层的Class Token向量。
- **线性投影**：将隐藏维度 $ D $ 映射到类别数（如ImageNet-1K为1000）。

### **2. ViT标准架构变体**
ViT通过调整层数、隐藏维度等参数，定义了不同规模的模型：

| **模型**       | 层数 (L) | 隐藏维度 (D) | MLP维度 | 头数 (H) | 参数量 (M) | ImageNet Top-1 Acc（预训练数据） |
|----------------|----------|--------------|---------|----------|------------|----------------------------------|
| ViT-Tiny   | 12       | 192          | 768     | 3        | 5.7        | 75.5%（ImageNet-21K）            |
| ViT-Small  | 12       | 384          | 1536    | 6        | 22.0       | 81.2%（ImageNet-21K）            |
| **ViT-Base**   | 12       | 768          | 3072    | 12       | 86.4       | 84.2%（JFT-300M）                |
| **ViT-Large**  | 24       | 1024         | 4096    | 16       | 307.0      | 85.9%（JFT-300M）                |
| **ViT-Huge**   | 32       | 1280         | 5120    | 16       | 632.0      | 86.5%（JFT-300M）                |


## Swin Transformer (2021)

Swin-Transformer 的核心创新点在于通过**层次化窗口注意力机制**和**移位窗口设计**，解决了传统 Vision Transformer 在高分辨率图像任务中的计算复杂度和缺乏多尺度特征的问题，同时融合了 CNN 的层次化优势与 Transformer 的全局建模能力。

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142576715-14668c6b-5cb8-4de8-ac51-419fae773c90.png" width="90%"/>
</div>

### **1. 层次化窗口注意力（Hierarchical Window Attention）**
- **局部窗口划分**：  
  将图像划分为不重叠的局部窗口（如 7×7 像素），**仅在窗口内计算自注意力**，将计算复杂度从 $ O(N^2) $ 降至 $ O(N \times M) $（$ M $ 为窗口内块数）。
- **复杂度对比**（以 224×224 图像为例）：
  | **模型**       | 全局注意力复杂度 | Swin窗口注意力复杂度（M=49） |
  |----------------|-------------------|-----------------------------|
  | ViT            | $ (196)^2 = 38,416 $ | $ 196 \times 49 = 9,604 $ |
  | Swin-Transformer | -               | **计算量减少约 75%**        |

- **层级特征图**：通过 4 个阶段（Stage 1-4）逐步合并窗口，构建金字塔特征结构：
  | **阶段** | 分辨率      | 窗口大小 | 特征维度 |
  |----------|-------------|----------|----------|
  | Stage 1  | 56×56       | 7×7      | 128      |
  | Stage 2  | 28×28       | 7×7      | 256      |
  | Stage 3  | 14×14       | 7×7      | 512      |
  | Stage 4  | 7×7         | 7×7      | 1024     |

![参考图片](https://i-blog.csdnimg.cn/blog_migrate/7bace09246ce1de138a32c80a60a6458.png)
![参考图片](https://i-blog.csdnimg.cn/blog_migrate/f26512576881a6ce6449e82fa66d6647.png)


### **2. 移位窗口（Shifted Window）**
- **动机**：解决局部窗口间的信息隔离问题，增强跨窗口交互。
- **实现方式**：  
  在连续的两个 Transformer 层中交替使用两种窗口划分方式：
  1. **常规窗口划分**：不重叠的均匀网格。
  2. **移位窗口划分**：窗口向右下角偏移 $ \lfloor \frac{M}{2} \rfloor $ 个像素（$ M $ 为窗口大小），使相邻窗口部分重叠。
- **效果**：  
  - 引入跨窗口连接，无需额外计算开销。  
  - 相比全局注意力，移位窗口的复杂度仍为 $ O(N \times M) $。

```
# 交替使用两种窗口划分方式
for i in range(depth):
  _block_cfg = {
      'embed_dims': embed_dims,
      'num_heads': num_heads,
      'window_size': window_size,
      'shift': False if i % 2 == 0 else True,
      'drop_path': drop_paths[i],
      'with_cp': with_cp,
      'pad_small_map': pad_small_map,
      **block_cfgs[i]
  }
  block = SwinBlock(**_block_cfg)
  self.blocks.append(block)

```

### **3. 相对位置编码（Relative Position Bias）**
- **设计**：在自注意力计算中引入**可学习的相对位置偏置**，替代 ViT 的绝对位置编码。
- **公式**：  
  $$
  \text{Attention} = \text{Softmax}(QK^T / \sqrt{d} + B) V
  $$
  - $ B \in \mathbb{R}^{M^2 \times M^2} $：相对位置偏置矩阵，$ M $ 为窗口大小。
- **优势**：  
  - 更适配图像的 2D 空间结构。  
  - 提升模型对物体相对位置的感知能力。

![参考图片](https://pic2.zhimg.com/v2-0a9f8976c102f6176cbd10029c5772db_1440w.jpg)

```
def get_attn_mask(hw_shape, window_size, shift_size, device=None):
    if shift_size > 0:
        img_mask = torch.zeros(1, *hw_shape, 1, device=device)
        h_slices = (slice(0, -window_size), slice(-window_size,
                                                  -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size,
                                                  -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # nW, window_size, window_size, 1
        mask_windows = ShiftWindowMSA.window_partition(
            img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
    else:
        attn_mask = None
    return attn_mask
```

### **4. 灵活的多尺度特征融合**
- **Patch Merging**：  
  在阶段过渡时合并相邻 2×2 窗口的特征，通过拼接与线性投影实现下采样：
  - 分辨率减半，通道数翻倍（如 56×56×128 → 28×28×256）。
- **效果**：  
  构建类似 CNN 的金字塔特征，支持密集预测任务（如目标检测、分割）。

```
# 整体流程
if self.shift_size > 0:
    shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
else:
    shifted_x = x

# 窗口划分与注意力计算
x = window_partition(shifted_x, self.window_size)
...
x = window_reverse(x, self.window_size, H, W)

# 反向移位
if self.shift_size > 0:
    x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
```

### **5. 与 CNN 和 ViT 的对比**
| **特性**         | Swin-Transformer          | ViT                      | CNN                |  
|------------------|---------------------------|--------------------------|--------------------|  
| **计算复杂度**   | 线性增长（窗口注意力）     | 平方增长（全局注意力）    | 线性增长（卷积）   |  
| **多尺度特征**   | 层次化窗口合并             | 固定分辨率                | 池化下采样         |  
| **全局建模**     | 移位窗口实现隐式全局交互    | 显式全局注意力            | 受限（感受野累积） |  
| **位置编码**     | 相对位置偏置               | 绝对位置编码              | 无（通过零填充隐含） |  


## Swin Transformer V2 (2021)

Swin Transformer V2 在 Swin Transformer 的基础上，针对**大模型训练稳定性**和**超高分辨率图像处理**进行了多项关键改进，旨在提升模型容量和实用性。

<div align=center>
<img src="https://user-images.githubusercontent.com/42952108/180748696-ee7ed23d-7fee-4ccf-9eb5-f117db228a42.png" width="100%"/>
</div>

### **1. 残差后归一化（ResPostNorm）**

```
# Swin Transformer
def forward(self, x, hw_shape):
    def _inner_forward(x):
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)
        return x

# Swin Transformer V2
def forward(self, x, hw_shape):
    def _inner_forward(x):
        # Use post normalization
        identity = x
        x = self.attn(x, hw_shape)
        x = self.norm1(x)
        x = x + identity

        identity = x
        x = self.ffn(x)
        x = self.norm2(x)
        x = x + identity

        if self.extra_norm:
            x = self.norm3(x)

        return x
```

### **2. 缩放余弦注意力（Scaled Cosine Attention）**
- **问题**：传统点积注意力易受输入块间相似度差异影响，导致训练不稳定。
- **改进**：  
  用余弦相似度替代点积，并引入可学习的缩放因子 $ \tau $：
  $$
  \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\tau \cdot \|Q\| \|K\|}\right)V
  $$
- **优势**：降低相似度计算对幅度的敏感性，提升注意力权重分布的稳定性。

```
# cosine attention
attn = (
    F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
logit_scale = torch.clamp(
    self.logit_scale, max=np.log(1. / 0.01)).exp()
attn = attn * logit_scale
```

### **3. 对数间隔连续位置偏差（Log-Spaced Continuous Position Bias, LogCPB）**
- **问题**：原版相对位置偏置表$ B \in \mathbb{R}^{(2M-1)×(2M-1)} $无法泛化到训练时未见过的窗口大小。
- **改进**：  
  - 使用小型 MLP 网络生成位置偏置，输入为**对数间隔的相对坐标**。
  - 公式：  
    $$
    B(\Delta x, \Delta y) = \text{MLP}(\log(|\Delta x| + 1), \log(|\Delta y| + 1))
    $$
- **优势**：支持动态窗口大小和更高分辨率推理（如 1536×1536）。

```
# relative_coords_table定义，对相对坐标的绝对值取对数，以指数间隔替代线性间隔。这样做是为了更好地捕捉不同距离之间的关系，尤其是在较大距离时，对数变换可以压缩动态范围，使模型更容易学习不同尺度下的位置关系
relative_coords_table = torch.stack(
    torch_meshgrid([relative_coords_h, relative_coords_w])).permute(
        1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
if pretrained_window_size[0] > 0:
    relative_coords_table[:, :, :, 0] /= (
        pretrained_window_size[0] - 1)
    relative_coords_table[:, :, :, 1] /= (
        pretrained_window_size[1] - 1)
else:
    relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
    relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
# [-1, 1]映射到[-8, 8]
relative_coords_table *= 8  # normalize to -8, 8
relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
    torch.abs(relative_coords_table) + 1.0) / np.log2(8)


# relative_position_bias_table使用
relative_position_bias_table = self.cpb_mlp(
    self.relative_coords_table).view(-1, self.num_heads)
relative_position_bias = relative_position_bias_table[
    self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1],
        self.window_size[0] * self.window_size[1],
        -1)  # Wh*Ww,Wh*Ww,nH
relative_position_bias = relative_position_bias.permute(
    2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
# 范围扩充至[0, 16]
relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
```

### **4. 自监督预训练策略（Self-Supervised Pre-Training）**
- **任务设计**：  
  使用 **SimMIM**（遮蔽图像建模）进行预训练，随机遮蔽 60% 的图像块并预测原始像素。
- **改进**：  
  - 采用轻量化解码器（2 层 Transformer），降低预训练成本。
  - 结合 SwinV2 的 LogCPB，提升跨分辨率迁移能力。

### **5. 内存优化技术**
- **零冗余优化器（ZeRO-Offload）**：  
  将优化器状态卸载到 CPU，降低 GPU 显存占用。
- **梯度检查点（Gradient Checkpointing）**：  
  牺牲计算时间换取内存节省，支持更大 batch size。

### **6. 性能对比（ImageNet-22K 微调）**
| **模型**         | 参数量 | 分辨率 | Top-1 Acc | 训练内存（GB） |
|------------------|--------|--------|-----------|----------------|
| SwinV1-Giant     | 3B     | 224    | 89.5%     | OOM            |
| **SwinV2-Giant** | 3B     | 256    | **90.2%** | **40**         |
| SwinV2-Huge      | 658M   | 512    | **88.7%** | 22             |


| **模型变体**   | 层数（Layers） | 隐藏维度（C） | 头数（Heads） | MLP扩展比 | 窗口大小（Window） | 参数量（Million） | 
|----------------|----------------|---------------|---------------|-----------|--------------------|-------------------|
| **Swin-T**     | 4 stages       | 96            | [3, 6, 12, 24]| 4         | 7×7               | **28M**          |
| **Swin-S**     | 4 stages       | 96→192→384→768| [3, 6, 12, 24]| 4         | 7×7               | **50M**          |
| **Swin-B**     | 4 stages       | 128→256→512→1024| [4, 8, 16, 32]| 4         | 7×7               | **88M**          |
| **Swin-L**     | 4 stages       | 192→384→768→1536| [6, 12, 24, 48]| 4        | 7×7               | **197M**         |
| **SwinV2-T**   | 4 stages       | 96            | [3, 6, 12, 24]| 4         | 8→16→32            | **30M**          |
| **SwinV2-S**   | 4 stages       | 96→192→384→768| [3, 6, 12, 24]| 4         | 8→16→32            | **52M**          |
| **SwinV2-B**   | 4 stages       | 128→256→512→1024| [4, 8, 16, 32]| 4         | 8→16→32            | **91M**          |
| **SwinV2-L**   | 4 stages       | 192→384→768→1536| [6, 12, 24, 48]| 4        | 8→16→32            | **200M**         |
| **SwinV2-G**   | 4 stages       | 256→512→1024→2048| [8, 16, 32, 64]| 4       | 8→16→32            | **3B (3000M)**   |


## DeiT (2021)

DeiT（Data-efficient Image Transformers）的核心创新点在于通过**知识蒸馏**和**高效训练策略**，在无需海量预训练数据（仅使用ImageNet-1K）的情况下，使视觉Transformer模型达到与CNN相当甚至更优的性能。

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/143225703-c287c29e-82c9-4c85-a366-dfae30d198cd.png" width="40%"/>
</div>

### **1. 知识蒸馏策略（Knowledge Distillation）**
#### **(1) 蒸馏Token设计**
- **引入蒸馏Token**：在输入序列中增加一个可学习的**蒸馏Token**（与Class Token并列），专门用于接收教师模型（如RegNetY-16GF）的知识。
- **双Token输出**：模型同时输出两个预测结果：
  - **Class Token** → 真实标签监督。
  - **Distill Token** → 教师模型预测监督。

#### **(2) 蒸馏方式**
- **软蒸馏（Soft Distillation）**：最小化学生模型与教师模型输出的KL散度。
  $$
  \mathcal{L}_{\text{soft}} = (1 - \lambda) \cdot \mathcal{L}_{\text{CE}}(y, y_s) + \lambda \cdot \tau^2 \cdot \text{KL}(y_t / \tau, y_s / \tau)
  $$
- **硬蒸馏（Hard Distillation）**：直接匹配教师模型的硬标签。
  $$
  \mathcal{L}_{\text{hard}} = \frac{1}{2} \mathcal{L}_{\text{CE}}(y, y_s) + \frac{1}{2} \mathcal{L}_{\text{CE}}(y_t, y_{\text{distill}})
  $$
  - $ y_s, y_{\text{distill}} $：学生模型的Class Token和Distill Token预测。
  - $ y_t $：教师模型预测标签。

#### **(3) 教师模型选择**
- **CNN教师优势**：使用高效CNN（如RegNetY）作为教师，弥补Transformer在小数据下的局部性建模不足。

### **2. 高效训练策略**
- **强数据增强**：结合RandAugment、Mixup、CutMix和重复增强（Repeated Augmentation），提升数据多样性。
- **优化器配置**：使用AdamW优化器，搭配余弦学习率衰减，避免过拟合。
- **随机深度（Stochastic Depth）**：引入层随机丢弃（如20%概率），增强泛化能力。

### **3. 模型架构优化**
- **DeiT-Tiny/DeiT-Small**：提供参数量更小的模型（5.7M/22M参数），适应资源受限场景。
- **性能对比**（ImageNet-1K）：
  | 模型         | 参数量 (M) | Top-1 Acc | 预训练数据 |  
  |--------------|------------|-----------|------------|  
  | ViT-B/16     | 86         | 77.9%     | JFT-300M   |  
  | **DeiT-B**   | 86         | **83.4%** | ImageNet-1K|  
  | ResNet-50    | 25         | 76.1%     | ImageNet-1K|  

## DeiT 3 (2022)

参考博文[DeiT III：打造ViT最强基准](https://zhuanlan.zhihu.com/p/511159229)。

DeiT III（也称为DeiT III: Revenge of the ViT）在视觉Transformer（ViT）的训练策略上进行了多项创新，旨在提升ViT模型在有监督训练任务中的性能。

### **1. 简化数据增强策略（3-Augment）**
3-Augment包括三种简单的图像变换：灰度化（Grayscale）、过曝（Solarization）和高斯模糊（Gaussian blur）。对于每张图片，随机选择其中一种变换进行增强，同时保留常用的颜色抖动（Color Jitter）和水平翻转（Horizontal Flip）。  
效果：实验表明，3-Augment在ViT模型上的效果优于常用的自动/学习数据增强方法（如RandAugment），特别是在大数据集（如ImageNet-21k）上预训练时。
![参考图片](https://pica.zhimg.com/v2-51391cef5fd102ce4976a6372bb4944a_1440w.jpg)

### **2. 简单随机裁剪（Simple Random Cropping）**
裁剪方式：在大数据集上预训练时，DeiT III采用了一种更简单的随机裁剪方法（SRC），即先将图像的最短边调整到目标大小，然后进行4个像素的反射填充（reflect padding），最后随机裁剪出一个正方形区域。  
优势：与常用的随机调整大小后裁剪（Random Resize Cropping, RRC）相比，SRC在保持图像宽高比的同时，减少了裁剪区域与原始图像之间的差异，提高了裁剪区域与实际标签的一致性。  
效果：实验表明，在ImageNet-21k上预训练时，采用SRC的模型性能优于采用RRC的模型。
![参考图片](https://pic4.zhimg.com/v2-772915ef0ef9cb413f308587b1dc16e3_1440w.jpg)

### **3. 降低训练分辨率（FixRes）**
策略：DeiT III采用了一种称为FixRes的策略，即先在较低的图像分辨率下训练模型，然后在目标分辨率下进行微调。  
优势：这种策略减少了训练和测试阶段的分辨率差异，有助于提升模型的泛化能力。同时，降低训练分辨率还可以减少显存消耗，提高训练速度。  
效果：实验表明，采用FixRes策略的模型在ImageNet-1k上的性能优于直接在目标分辨率下训练的模型。

### **4. 正则化方法**
随机深度（Stochastic Depth）：DeiT III在所有层中采用统一的丢弃率（drop rate），并根据模型大小进行调整。这种方法有助于防止过拟合，提高模型的泛化能力。  
LayerScale：DeiT III采用了LayerScale技术，通过为每一层引入可学习的缩放因子，解决了更深ViT模型的收敛问题。尽管DeiT III的训练过程不存在收敛问题，但作者发现使用LayerScale可以获得更高的精度。

```
# LayerScale用于DeiT3FFN

@deprecated_api_warning({'residual': 'identity'}, cls_name='FFN')
def forward(self, x, identity=None):
    """Forward function for `FFN`.

    The function would add x to the output tensor if residue is None.
    """
    out = self.layers(x)
    out = self.gamma2(out)
    if not self.add_identity:
        return self.dropout_layer(out)
    if identity is None:
        identity = x
    return identity + self.dropout_layer(out)
```

### **5. 损失函数**
二元交叉熵（Binary Cross Entropy, BCE）：在ImageNet-1k上训练较小的ViT模型时，DeiT III采用了BCE损失函数而不是常用的交叉熵（Cross Entropy, CE）损失函数。作者发现，采用BCE损失函数可以在一定程度上提高模型的性能。然而，在ImageNet-21k上预训练时，BCE损失函数并没有带来明显的改进，因此仍采用CE损失函数。

### **6. 训练策略的优化**
训练时长：DeiT III默认的训练时长为400个epoch（在ImageNet-1k上），相比之前的DeiT模型有所增加。实验表明，进一步增加训练时长，模型性能仍然能够持续提升。  
