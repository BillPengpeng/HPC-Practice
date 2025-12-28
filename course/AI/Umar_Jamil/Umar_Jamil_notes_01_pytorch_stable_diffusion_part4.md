本文主要整理Stable Diffusion implemented from scratch in PyTorch的主要内容。

## 7. - pipeline源码分析

这是**Stable Diffusion 完整图像生成流程**的实现代码，它整合了CLIP文本编码器、VAE编码器/解码器、扩散模型U-Net，实现了**文本到图像**和**图像到图像**的生成。这是整个Stable Diffusion系统的**顶层接口**。

---

### **整体架构概述**

这个`generate`函数是Stable Diffusion的**主入口点**，它实现了以下功能：

1. **文本到图像生成**：从纯噪声开始，根据文本提示生成图像
2. **图像到图像生成**：基于输入图像，根据文本提示进行编辑
3. **条件控制**：支持无分类器引导（CFG）增强文本对齐
4. **灵活配置**：支持不同采样器、步数、种子等参数

### **工作流程概览**
```
1. 文本编码 → 2. 潜在初始化 → 3. 迭代去噪 → 4. 图像解码
```

---

### **输入参数详解**

**1. 核心参数**

```python
def generate(
    prompt,                 # 文本提示
    uncond_prompt=None,     # 无条件提示（用于CFG）
    input_image=None,       # 输入图像（img2img模式）
    strength=0.8,           # 图像编辑强度
    do_cfg=True,           # 是否使用无分类器引导
    cfg_scale=7.5,         # CFG引导强度
    sampler_name="ddpm",   # 采样器类型
    n_inference_steps=50,  # 去噪步数
    models={},             # 预加载的模型组件
    seed=None,             # 随机种子
    device=None,           # 计算设备
    idle_device=None,      # 闲置设备（内存管理）
    tokenizer=None,        # CLIP分词器
):
```

**2. 参数意义**

| 参数 | 类型 | 默认值 | 作用 |
|------|------|--------|------|
| `prompt` | str | 必需 | 文本描述，指导图像生成 |
| `uncond_prompt` | str | None | 负面提示，引导模型避开某些内容 |
| `input_image` | PIL.Image | None | 输入图像（img2img模式） |
| `strength` | float | 0.8 | 控制img2img时添加的噪声量 |
| `do_cfg` | bool | True | 启用/禁用无分类器引导 |
| `cfg_scale` | float | 7.5 | 引导强度，值越大越遵循文本 |
| `sampler_name` | str | "ddpm" | 采样器类型（如DDPM、DDIM） |
| `n_inference_steps` | int | 50 | 去噪迭代次数 |
| `seed` | int | None | 随机种子，控制可重复性 |

---

### **完整流程分步解析**

#### **阶段1：初始化与设备管理**

```python
# 设备管理：将不用的模型移到空闲设备节省显存
if idle_device:
    to_idle = lambda x: x.to(idle_device)
else:
    to_idle = lambda x: x

# 随机种子设置
generator = torch.Generator(device=device)
if seed is None:
    generator.seed()  # 随机种子
else:
    generator.manual_seed(seed)  # 固定种子
```

**设备管理策略**：
- **活跃设备**：当前正在计算的设备（通常是GPU）
- **空闲设备**：存储不活跃模型的设备（CPU或其他GPU）
- **惰性加载**：只有需要时才将模型移到活跃设备
- **显存优化**：在处理大模型时至关重要

---

#### **阶段2：文本编码（CLIP处理）**

**1. 分词与编码**

```python
# 正面提示词编码
cond_tokens = tokenizer.batch_encode_plus(
    [prompt], padding="max_length", max_length=77
).input_ids
cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
cond_context = clip(cond_tokens)  # (1, 77, 768)

# 无条件提示词编码（用于CFG）
uncond_tokens = tokenizer.batch_encode_plus(
    [uncond_prompt], padding="max_length", max_length=77
).input_ids
uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
uncond_context = clip(uncond_tokens)  # (1, 77, 768)

# 拼接条件与无条件上下文
context = torch.cat([cond_context, uncond_context])  # (2, 77, 768)
```

**分词器工作流程**：
```
文本 → 分词 → 标记ID → 填充/截断到77 → 张量化
示例："a dog" → ["a", "dog"] → [320, 1929] → [320, 1929, 0, 0, ...]
```

**CLIP输出**：
- 形状：`(batch_size, 77, 768)`
- 77是CLIP最大序列长度
- 768是CLIP嵌入维度
- 每个位置包含该标记的上下文感知表示

**2. 无分类器引导机制**

```python
if do_cfg:
    # 拼接条件和无条件上下文
    context = torch.cat([cond_context, uncond_context])  # (2, 77, 768)
else:
    # 只使用条件上下文
    context = clip(tokens)  # (1, 77, 768)
```

**CFG的数学原理**：
```
pred_noise = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
```
- `uncond_pred`：无条件预测（多样性基础）
- `cond_pred`：条件预测（文本对齐）
- `cfg_scale`：控制文本遵循程度

**典型CFG值**：
- `cfg_scale=1.0`：标准条件生成
- `cfg_scale=7.5`：Stable Diffusion默认，强文本对齐
- `cfg_scale>10`：过强引导，可能降低图像质量

---

#### **阶段3：潜在空间初始化**

**1. 纯文本生成模式（txt2img）**

```python
if input_image is None:
    # 从标准正态分布采样随机噪声
    latents = torch.randn(latents_shape, generator=generator, device=device)
    # 形状: (1, 4, 64, 64)
```

**潜在空间维度**：
- 输入图像：512×512×3 = 786,432像素
- 潜在空间：64×64×4 = 16,384值
- **压缩率**：约48:1

**2. 图像到图像模式（img2img）**

```python
if input_image:
    # 1. 图像预处理
    input_image_tensor = input_image.resize((WIDTH, HEIGHT))  # 调整尺寸
    input_image_tensor = np.array(input_image_tensor)  # PIL转numpy
    input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
    
    # 2. 像素值归一化 [-1, 1]
    input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
    
    # 3. 维度重排 (H,W,C) -> (B,C,H,W)
    input_image_tensor = input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    
    # 4. VAE编码到潜在空间
    encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
    latents = encoder(input_image_tensor, encoder_noise)  # (1, 4, 64, 64)
    
    # 5. 添加噪声（控制编辑强度）
    sampler.set_strength(strength=strength)
    latents = sampler.add_noise(latents, sampler.timesteps[0])
```

**强度参数`strength`的作用**：
- `strength=0.0`：完全保留原图
- `strength=0.5`：中等程度编辑
- `strength=1.0`：完全重绘（类似txt2img）
- 控制添加到编码图像中的噪声量

**数学原理**：
```
# 根据strength选择起始时间步
start_timestep = int(strength * total_timesteps)
# 添加对应噪声
noisy_latents = add_noise(latents, start_timestep)
```

---

#### **阶段4：时间步嵌入生成**

```python
def get_time_embedding(timestep):
    # 创建频率向量
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # 频率与时间步相乘
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # 正弦和余弦编码
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)  # (1, 320)
```

**正弦位置编码原理**：
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**参数解释**：
- `160`：频率向量长度
- `320`：最终时间嵌入维度（160×2）
- 这种编码能捕捉时间步的相对位置信息

---

#### **阶段5：迭代去噪（扩散模型推理）**

这是整个流程的**核心部分**，U-Net逐步去除潜在空间中的噪声。

**1. 采样器初始化**

```python
if sampler_name == "ddpm":
    sampler = DDPMSampler(generator)
    sampler.set_inference_timesteps(n_inference_steps)
```

**DDPM采样器功能**：
- 管理时间步调度
- 实现去噪更新公式
- 处理不同类型的噪声调度

**时间步调度**：
```python
# 通常使用线性调度
timesteps = torch.linspace(999, 0, n_inference_steps, dtype=torch.long)
# 或余弦调度（更平滑）
```

**2. 主去噪循环**

```python
timesteps = tqdm(sampler.timesteps)  # 进度条
for i, timestep in enumerate(timesteps):
    # 1. 生成时间嵌入
    time_embedding = get_time_embedding(timestep).to(device)  # (1, 320)
    
    # 2. 准备模型输入
    model_input = latents
    if do_cfg:
        # CFG需要两份输入：条件+无条件
        model_input = model_input.repeat(2, 1, 1, 1)  # (2, 4, 64, 64)
    
    # 3. U-Net预测噪声
    model_output = diffusion(model_input, context, time_embedding)  # 预测噪声
    
    # 4. 应用CFG（如果启用）
    if do_cfg:
        output_cond, output_uncond = model_output.chunk(2)
        model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
    
    # 5. 采样器更新潜在表示
    latents = sampler.step(timestep, latents, model_output)
```

**U-Net前向传播细节**：
```
输入: 
  - latents: (B, 4, 64, 64) 带噪潜在
  - context: (B, 77, 768) 文本嵌入
  - time_embedding: (B, 320) 时间嵌入
  
输出: 
  - predicted_noise: (B, 4, 64, 64) 预测的噪声
```

**CFG应用步骤**：
1. **重复输入**：`latents`重复2次
2. **同时推理**：U-Net同时处理条件和无条件输入
3. **分离输出**：`model_output`包含两个预测
4. **线性组合**：按公式组合两个预测

**采样器更新**（DDPM公式）：
```
# DDPM采样公式简化版
predicted_original = (latents - sqrt(1-alpha_bar[t]) * pred_noise) / sqrt(alpha_bar[t])
latents_prev = (predicted_original + sqrt(1-alpha_bar[t-1]) * pred_noise) * 系数
```

---

#### **阶段6：图像解码与后处理**

**1. VAE解码**

```python
decoder = models["decoder"]
decoder.to(device)
# 潜在空间 -> 像素空间
images = decoder(latents)  # (1, 3, 512, 512)
```

**VAE解码器作用**：
- 将64×64×4的潜在表示上采样到512×512×3
- 使用反卷积和上采样层
- 学习从特征到像素的映射

**2. 后处理**

```python
# 1. 像素值反归一化 [-1, 1] -> [0, 255]
images = rescale(images, (-1, 1), (0, 255), clamp=True)

# 2. 维度重排 (B,C,H,W) -> (B,H,W,C)
images = images.permute(0, 2, 3, 1)

# 3. 数据类型转换和设备转移
images = images.to("cpu", torch.uint8).numpy()

# 4. 返回第一个（也是唯一一个）图像
return images[0]  # (512, 512, 3) uint8 numpy数组
```

**像素值范围处理**：
- 训练时：图像归一化到[-1, 1]
- 推理后：反归一化到[0, 255]
- 钳制：确保值在有效范围内

**rescale函数**：
```python
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x
```

---

### **模型组件交互图**

```
用户输入
    │
    ├──文本提示──────────────┐
    │                       │
    └──输入图像（可选）───────┤
                            │
                    CLIP文本编码器
                            ↓
                    文本嵌入 (77, 768)
                            │
                    ┌───────┴───────┐
                    │               │
            无输入图像        有输入图像
                    │               │
            随机噪声 ───┐     VAE编码器
                    │  │        │
                    ↓  ↓        ↓
                潜在空间 (4, 64, 64)
                            │
                    DDPM采样器 + U-Net
                    （迭代去噪50步）
                            ↓
                    干净潜在表示
                            │
                    VAE解码器
                            ↓
                    像素图像 (3, 512, 512)
                            │
                    后处理与输出
```

---

### **性能优化策略**

#### **1. 计算优化**

```python
# 使用半精度推理（如果支持）
with torch.autocast(device_type='cuda', dtype=torch.float16):
    model_output = diffusion(model_input, context, time_embedding)
```

**优化技巧**：
1. **混合精度**：FP16训练/推理
2. **CUDA图**：捕获和重用计算图
3. **内核融合**：自定义CUDA内核
4. **注意力优化**：Flash Attention等

#### **2. 内存优化**

```python
# 梯度检查点（节省显存）
from torch.utils.checkpoint import checkpoint

def custom_forward(module, *inputs):
    def inner(*inputs):
        return module(*inputs)
    return checkpoint(inner, *inputs)
```

**内存节省技术**：
1. **梯度检查点**：用时间换空间
2. **CPU卸载**：中间特征存CPU
3. **模型分片**：多GPU流水线
4. **激活重计算**：需要时重新计算

#### **3. 延迟优化**

**预热与缓存**：
```python
# 预热模型
with torch.no_grad():
    _ = model(torch.randn(1,4,64,64).to(device), 
              torch.randn(1,77,768).to(device),
              torch.randn(1,320).to(device))

# 缓存CLIP嵌入（相同提示词）
clip_cache = {}
if prompt in clip_cache:
    context = clip_cache[prompt]
else:
    context = clip(encode(prompt))
    clip_cache[prompt] = context
```
