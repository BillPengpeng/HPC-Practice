本文主要整理Stable Diffusion implemented from scratch in PyTorch的主要内容。

## 5. - clip源码分析

### **整体架构概述**

这个CLIP文本编码器是一个**Transformer解码器风格的编码器**，专门为处理文本序列而设计。它将**文本标记序列**（如["a", "dog", "in", "the", "park"]）转换为**丰富的上下文感知的文本嵌入**，用于引导扩散模型生成图像。

### **关键特性**
- 12层Transformer块
- 768维嵌入空间
- 最大序列长度77个标记
- 词汇表大小49408（包括特殊标记）

---

### **核心组件详解**

#### **1. CLIPEmbedding（嵌入层）**

```python
class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        # 标记嵌入：将离散标记ID映射为连续向量
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # 位置嵌入：学习序列中每个位置的特征
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
```

**工作原理**：
1. **标记嵌入**：将整数标记ID（如304、5432）转换为768维向量
   - 词汇表大小49408：包含所有英语单词、子词和特殊标记
   - 嵌入维度768：与CLIP图像编码器输出维度对齐
2. **位置嵌入**：可学习的位置编码，告诉模型每个词在序列中的位置
   - 形状`(77, 768)`：支持最大77个标记
   - 在训练中学习，非正弦编码

**在Stable Diffusion中的输入**：
```python
# 示例：输入文本"A dog in the park"
tokens = [49406, 320, 1929, 530, 262, 2415, 49407]  # 添加了开始和结束标记
# 形状：(Batch_Size, 7)
```

---

### **CLIPLayer（Transformer块）**

这是CLIP的核心，每个块包含**自注意力机制**和**前馈网络**。

#### **1. 自注意力部分**

```python
# Pre-attention norm
self.layernorm_1 = nn.LayerNorm(n_embd)
# Self attention
self.attention = SelfAttention(n_head, n_embd)
```

**层归一化的作用**：
- 在每个子层（注意力、前馈）之前应用
- 稳定训练，加速收敛
- 在序列维度（dim=768）上归一化

**自注意力配置**：
```python
# CLIP使用12头注意力
self.attention = SelfAttention(12, 768, causal_mask=True)
```
- **12个头**：每个头关注不同的语义方面
- **因果掩码**：确保位置只能关注其左侧（包括自身）的位置
- 这是**解码器风格**的关键特征

#### **2. 前馈网络部分**

```python
# Feedforward layer
self.linear_1 = nn.Linear(n_embd, 4 * n_embd)  # 扩展
self.linear_2 = nn.Linear(4 * n_embd, n_embd)  # 压缩
```

**激活函数**：
```python
x = x * torch.sigmoid(1.702 * x)  # QuickGELU
```
这是CLIP特有的**QuickGELU激活函数**：
- 比标准GELU计算更快
- 参数1.702是经验值
- 公式：`x * sigmoid(1.702 * x)`
- 接近ReLU但更平滑

#### **3. 残差连接**

在每个子层后都使用残差连接：
```python
# 自注意力残差
x = x + residue

# 前馈网络残差  
x = x + residue
```

**为什么重要**：
- 防止梯度消失/爆炸
- 允许构建深层网络（12层）
- 保留原始信息的同时添加新特征

---

### **完整的CLIP编码器**

#### **1. 12层堆叠**

```python
self.layers = nn.ModuleList([
    CLIPLayer(12, 768) for i in range(12)
])
```
- 12层相同的结构
- 每层参数独立
- 随着深度增加，提取更抽象的语义特征

#### **2. 前向传播流程**

```python
def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
    # 1. 嵌入层
    state = self.embedding(tokens)  # (B, 77, 768)
    
    # 2. 12个Transformer块
    for layer in self.layers: 
        state = layer(state)  # 逐步细化表示
        
    # 3. 最终层归一化
    output = self.layernorm(state)  # (B, 77, 768)
    
    return output
```

**信息流动**：
1. 输入：`(Batch_Size, 77)` 整数标记
2. 嵌入后：`(Batch_Size, 77, 768)` 词向量
3. 每层处理后：维度不变，语义增强
4. 输出：`(Batch_Size, 77, 768)` 上下文感知的文本表示

---

### **在Stable Diffusion中的作用**

#### **1. 文本到图像的桥梁**

CLIP文本编码器将自然语言描述转换为扩散模型能理解的"语言"：

```
文本描述 → CLIP编码器 → 文本嵌入 → 扩散模型条件
"A dog"    → (77, 768)向量 → 引导图像生成
```

#### **2. 与交叉注意力配合**

在扩散模型的U-Net中，CLIP输出作为**交叉注意力的键和值**：

```python
# 在扩散模型中
class CrossAttention(nn.Module):
    def forward(self, x, y):
        # x: 图像潜在表示 (B, 4096, 320)
        # y: CLIP文本嵌入 (B, 77, 768) ← 这里！
        
        # 投影到相同维度
        k = self.k_proj(y)  # (B, 77, 320)
        v = self.v_proj(y)  # (B, 77, 320)
        
        # 计算注意力
        weight = q @ k.transpose(-1, -2)  # 图像查询与文本键的相似度
```

#### **3. 序列长度77的原因**

CLIP模型设计为处理最大77个标记：
- 第一个标记：`<|startoftext|>` (49406)
- 最后标记：`<|endoftext|>` (49407)
- 中间最多75个内容标记
- 这是BERT/GPT的标准长度，平衡计算和表达能力

---

### **关键技术细节**

#### **1. 因果掩码的作用**

```python
x = self.attention(x, causal_mask=True)
```

**为什么需要因果掩码**：
- CLIP文本编码器是**自回归解码器风格**
- 确保位置i只能看到位置≤i的标记
- 这与GPT系列模型一致
- 在推理时，支持逐步生成文本（虽然Stable Diffusion中不用于生成文本）

**掩码矩阵示例**（序列长度=5）：
```
[[1, 0, 0, 0, 0],
 [1, 1, 0, 0, 0],
 [1, 1, 1, 0, 0],
 [1, 1, 1, 1, 0],
 [1, 1, 1, 1, 1]]
```
1=可见，0=掩码（设为-inf）

#### **2. 层归一化 vs 批归一化**

CLIP使用**层归一化**而非批归一化：
```python
self.layernorm_1 = nn.LayerNorm(n_embd)
```

**原因**：
- 处理**变长序列**（实际文本长度不同）
- 对**批次大小不敏感**（可处理单个样本）
- 在序列维度归一化，保持**时间步独立性**

#### **3. 前馈网络的扩展因子**

```python
self.linear_1 = nn.Linear(n_embd, 4 * n_embd)  # 扩展4倍
```
- 从768维扩展到3072维
- 增加模型容量
- 然后在第二层压缩回768维
- 这是Transformer的标准设计

#### **4. 训练与推理差异**

**预训练阶段**（OpenAI）：
- 在4亿（图像，文本）对上训练
- 对比学习目标：匹配的图像和文本应该有相似的嵌入
- 学习丰富的视觉-语言对应关系

**在Stable Diffusion中**（冻结使用）：
- CLIP权重固定，不参与扩散模型训练
- 只作为**文本特征提取器**
- 提供稳定、高质量的文本表示

---

### **与图像编码器的关系**

CLIP模型实际上是**双塔结构**：
- **文本编码器**（此代码）：处理文本输入
- **图像编码器**（ViT或CNN）：处理图像输入
- 两者输出在共享的768维空间中对齐

在Stable Diffusion中，**只使用文本编码器**，因为：
1. 目标是**文本到图像生成**
2. 图像编码器用于CLIP预训练，但生成时不需
3. 文本编码器提供足够的语义指导

---

### **输出格式与应用**

#### **1. 输出形状**

```python
# 输入 tokens: (Batch_Size, 77)
# 输出: (Batch_Size, 77, 768)
```

**每个标记的768维向量包含**：
- 词汇语义信息
- 位置信息
- 上下文信息（与句子中其他词的关系）
- 潜在的视觉关联

#### **2. 在扩散模型中的使用**

```python
# 简化版扩散模型前向传播
def forward(self, x, t, text_tokens):
    # 编码文本
    text_emb = clip_encoder(text_tokens)  # (B, 77, 768)
    
    # 在U-Net的多个层中注入文本条件
    for block in unet_blocks:
        if has_cross_attention(block):
            # 通过交叉注意力注入文本信息
            x = block.cross_attn(x, text_emb)
    
    return x
```

**注入点**：
- 在U-Net的下采样和上采样路径中
- 通常在残差块之后
- 多层注入，从粗到细控制

## 6. - diffusion U-Net源码解释

这是**Stable Diffusion 核心U-Net模型**的完整实现代码，它是**扩散模型的去噪网络**。这个复杂的网络结构负责在潜在空间中逐步去噪，从纯噪声生成有意义的图像，同时融合**时间步信息**和**文本条件**。

### **整体架构概述**

这个U-Net是一个**编码器-解码器结构**，具有**跳跃连接**，专门为扩散模型的去噪任务设计。与传统的U-Net相比，它整合了：

1. **时间步嵌入**：控制去噪过程的进度
2. **文本条件注入**：通过交叉注意力实现文本引导
3. **多分辨率特征提取**：在不同尺度上处理图像信息

### **输入输出**
- **输入**：带噪声的潜在表示 `(B, 4, 64, 64)` + 文本嵌入 `(B, 77, 768)` + 时间步 `(1, 320)`
- **输出**：预测的噪声 `(B, 4, 64, 64)`

---

### **核心组件详解**

#### **1. TimeEmbedding（时间步嵌入）**

```python
class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)  # 320 -> 1280
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)  # 1280 -> 1280
```

**作用**：将标量时间步`t`转换为高维向量表示

**工作原理**：
1. 输入：时间步`t`（0-1000）通过正弦位置编码转换为320维向量
2. 第一次线性变换：320 → 1280维
3. SiLU激活：引入非线性
4. 第二次线性变换：1280 → 1280维

**为什么需要时间嵌入**：
- 告诉网络当前处于去噪过程的哪个阶段
- 不同阶段需要不同的去噪策略
- 早期：去除粗粒度噪声
- 后期：恢复细粒度细节

---

### **残差块与注意力块**

#### **1. UNET_ResidualBlock（残差块）**

这是U-Net的**基本构建单元**，负责特征提取和时间条件融合。

```python
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        # 特征处理路径
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # 时间条件处理
        self.linear_time = nn.Linear(n_time, out_channels)
        
        # 合并后的处理
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, 3, padding=1)
```

**关键创新：时间条件融合**
```python
# 时间条件处理
time = F.silu(time)  # 激活
time = self.linear_time(time)  # 投影到特征维度

# 广播相加：将时间条件加到每个空间位置
merged = feature + time.unsqueeze(-1).unsqueeze(-1)
# feature: (B, C, H, W) + time: (1, C, 1, 1) -> (B, C, H, W)
```

**残差连接处理**：
```python
if in_channels == out_channels:
    self.residual_layer = nn.Identity()  # 恒等映射
else:
    self.residual_layer = nn.Conv2d(in_channels, out_channels, 1)  # 1×1卷积调整维度
```

#### **2. UNET_AttentionBlock（注意力块）**

这是U-Net的**核心条件注入模块**，结合了**自注意力**和**交叉注意力**。

```python
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        channels = n_head * n_embd  # 计算总通道数
        
        # 自注意力部分
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        
        # 交叉注意力部分（文本条件注入）
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        
        # 前馈网络（使用GeGLU）
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)  # 输出维度×2
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
```

**GeGLU激活函数**：
```python
# GeGLU: Gated Linear Unit with GELU
x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)  # 分割为两部分
x = x * F.gelu(gate)  # 逐元素相乘
```

**为什么用GeGLU**：
- 比标准前馈网络更强大
- 门控机制可以控制信息流
- 在实践中表现更好

**注意力块的工作流程**：
1. **空间自注意力**：图像内部不同位置的关系
2. **文本交叉注意力**：图像位置与文本标记的对齐
3. **前馈网络**：特征变换和非线性增强

---

### **SwitchSequential（智能序列容器）**

这是U-Net的**关键设计创新**，能够根据层类型自动传递不同的参数。

```python
class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)  # 注意力块需要context
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)  # 残差块需要time
            else:
                x = layer(x)  # 普通层（如卷积）
        return x
```

**为什么需要SwitchSequential**：
- U-Net由不同类型层混合组成
- 每层需要不同的输入参数
- 简化前向传播逻辑
- 提高代码可读性和可维护性

---

### **U-Net整体架构**

#### **1. 编码器（下采样路径）**

编码器逐步**降低空间分辨率**，增加通道数，提取多尺度特征。

```python
self.encoders = nn.ModuleList([
    # 输入层: 4通道 -> 320通道
    SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
    
    # 多个块，每个块包含：残差块 + 注意力块
    SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
    SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
    
    # 下采样: 320通道，尺寸减半
    SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
    
    # 增加通道数: 320 -> 640
    SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
    # ... 继续下采样和增加通道
])
```

**下采样流程**：
| 阶段 | 输入尺寸 | 输出尺寸 | 通道数 | 注意力头数 | 头部维度 |
|------|----------|----------|--------|------------|----------|
| 1    | 64×64    | 64×64    | 320    | 8          | 40       |
| 2    | 64×64    | 32×32    | 640    | 8          | 80       |
| 3    | 32×32    | 16×16    | 1280   | 8          | 160      |
| 4    | 16×16    | 8×8      | 1280   | -          | -        |

**注意力头计算**：
- 总维度 = 头数 × 每头维度
- 320 = 8 × 40
- 640 = 8 × 80
- 1280 = 8 × 160

#### **2. 瓶颈层**

这是编码器和解码器之间的**连接层**，在最低分辨率（8×8）上进行深度处理。

```python
self.bottleneck = SwitchSequential(
    UNET_ResidualBlock(1280, 1280), 
    UNET_AttentionBlock(8, 160),  # 在最低分辨率使用注意力
    UNET_ResidualBlock(1280, 1280), 
)
```

**作用**：
- 整合所有编码器特征
- 在全局上下文上进行处理
- 准备上采样所需的特征

#### **3. 解码器（上采样路径）**

解码器逐步**提高空间分辨率**，减少通道数，同时**融合编码器的跳跃连接**。

```python
self.decoders = nn.ModuleList([
    # 第一层：拼接编码器特征 (2560 = 1280 + 1280)
    SwitchSequential(UNET_ResidualBlock(2560, 1280)),
    
    # 后续层：继续拼接和上采样
    SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
    
    # 上采样后减少通道数
    SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
    
    # 最终层：320通道，准备输出
    SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
])
```

**上采样模块**：
```python
class Upsample(nn.Module):
    def forward(self, x):
        # 最近邻插值，尺寸×2
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # 3×3卷积平滑特征
        return self.conv(x)
```

**跳跃连接融合**：
```python
# 在解码器中
for layers in self.decoders:
    # 从编码器取出对应层的特征
    x = torch.cat((x, skip_connections.pop()), dim=1)  # 通道维度拼接
    x = layers(x, context, time)
```

**拼接模式**：
- 解码器输入 = 当前特征 + 编码器对应层特征
- 通道数倍增：1280 + 1280 = 2560
- 通过残差块减少到合适维度

---

### **完整前向传播流程**

#### **1. Diffusion类（顶层封装）**

```python
class Diffusion(nn.Module):
    def forward(self, latent, context, time):
        # 1. 时间步嵌入
        time = self.time_embedding(time)  # (1, 320) -> (1, 1280)
        
        # 2. U-Net去噪
        output = self.unet(latent, context, time)  # (B, 4, 64, 64) -> (B, 320, 64, 64)
        
        # 3. 输出层：预测噪声
        output = self.final(output)  # (B, 320, 64, 64) -> (B, 4, 64, 64)
        
        return output
```

#### **2. 信息流动示例**

假设生成512×512图像：
1. VAE编码：512×512×3 → 64×64×4
2. 扩散过程：在64×64×4的潜在空间添加噪声
3. U-Net输入：带噪的64×64×4潜在表示
4. 经过U-Net：预测噪声
5. 采样过程：从噪声中减去预测噪声，得到更清晰的潜在表示
6. 重复步骤3-5，直到得到干净的潜在表示
7. VAE解码：64×64×4 → 512×512×3

---

### **关键技术细节**

#### **1. 组归一化的使用**

在整个U-Net中大量使用`GroupNorm(32, channels)`：
- 稳定训练，不依赖批量大小
- 适合扩散模型的小批量训练
- 与注意力机制兼容

#### **2. 残差连接的两种形式**

**层内残差**：
- 在注意力块和残差块内部使用
- 防止梯度消失，稳定深度网络

**跳跃连接**：
- 编码器到解码器的连接
- 保留低级特征信息
- 帮助解码器恢复细节

#### **3. 注意力机制的位置策略**

**编码器**：
- 在每个分辨率都使用注意力
- 但不在最低分辨率（8×8）使用，因为计算成本高

**解码器**：
- 在上采样后使用注意力
- 帮助整合全局信息

**瓶颈层**：
- 在最低分辨率使用注意力
- 计算成本可接受（8×8 → 64个位置）

#### **特征维度设计**

**通道数变化**：
- 输入：4通道（潜在空间）
- 编码器：320 → 640 → 1280 → 1280
- 解码器：对称减少
- 输出：320通道（最终投影到4通道）

**为什么用这些数字**：
- 320是VAE潜在维度的80倍（4×80）
- 遵循2的幂次，计算高效
- 经验验证的最佳平衡点

---

### **在扩散模型训练中的作用**

#### **1. 训练目标**

U-Net学习预测噪声：
```
损失 = ||真实噪声 - 预测噪声||²
```

**前向传播**：
1. 采样时间步`t`，噪声`ε`
2. 构造带噪样本：`x_t = √āₜ x₀ + √(1-āₜ) ε`
3. U-Net预测噪声：`ε_θ = U-Net(x_t, t, context)`
4. 计算损失：`L = MSE(ε, ε_θ)`

#### **2. 条件注入机制**

**两种条件同时注入**：
1. **时间条件**：通过残差块中的加法注入
2. **文本条件**：通过交叉注意力注入

**注入时机**：
- 每个残差块都接收时间条件
- 注意力块在特定层接收文本条件
- 条件信息在网络中逐层传播

#### **3. 与Classifier-Free Guidance的关系**

在推理时，通过**无分类器引导**增强文本控制：
```
预测噪声 = 无条件预测 + w × (条件预测 - 无条件预测)
```

U-Net同时计算：
- 条件预测：`ε_θ(x_t, t, context)`
- 无条件预测：`ε_θ(x_t, t, ∅)`（空文本）

---

### **总结：U-Net在扩散模型中的核心作用**

#### **1. 架构设计的智慧**

1. **U型结构**：编码器-解码器，多尺度特征提取
2. **跳跃连接**：保留细节，帮助重建
3. **条件融合**：时间和文本条件自然融入
4. **注意力机制**：全局关系和文本对齐

#### **2. 与原始U-Net的差异**

| 特性 | 原始U-Net（医学分割） | 扩散模型U-Net |
|------|----------------------|--------------|
| **输入输出** | 图像到分割掩码 | 噪声潜在到预测噪声 |
| **条件信息** | 无 | 时间步 + 文本 |
| **注意力** | 无 | 自注意力 + 交叉注意力 |
| **归一化** | BatchNorm | GroupNorm |
| **深度** | 较浅 | 很深（数十层） |

#### **3. 成功的关键因素**

1. **多尺度处理**：同时捕捉全局结构和局部细节
2. **条件适应**：灵活响应不同时间步和文本提示
3. **深度可训练**：通过残差连接和归一化稳定训练
4. **计算效率**：在潜在空间操作，大幅减少计算量

**结论**：这个U-Net是**Stable Diffusion的大脑**，它将扩散理论、注意力机制和条件生成完美结合。通过精巧的架构设计，它能够在低维潜在空间中理解时间动态、整合文本条件，并执行复杂的去噪操作，最终实现从文本描述生成高质量图像的奇迹。