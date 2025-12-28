本文主要整理Stable Diffusion implemented from scratch in PyTorch的主要内容。

## 8. - ddpm源码分析

这是**去噪扩散概率模型采样器**的完整实现，负责控制扩散模型的**噪声调度**、**前向加噪**和**反向去噪**过程。它是Stable Diffusion等扩散模型生成图像时的**核心调度引擎**。

### **整体架构与作用**

DDPMSampler 是**扩散模型采样的控制中心**，主要负责：

1. **噪声调度管理**：定义如何随时间添加/移除噪声
2. **前向加噪**：为图像到图像生成准备噪声样本
3. **反向去噪**：执行DDPM采样算法，从噪声生成图像
4. **强度控制**：调节图像编辑的程度

---

### **初始化参数与噪声调度**

**1. 噪声调度初始化**

```python
def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
    # β调度：线性插值
    self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # ᾱ_t
```

**参数意义**：
- `beta_start=0.00085`, `beta_end=0.0120`：来自Stable Diffusion官方配置
- `num_training_steps=1000`：标准DDPM训练步数
- **β_t**：第t步的噪声方差，控制加噪强度
- **α_t = 1 - β_t**：保留信号的比例
- **ᾱ_t = ∏_{s=1}^{t} α_s**：累积乘积，用于直接计算任意步的加噪结果

**为什么对β开方再平方**：
```python
# 线性插值在平方根空间，确保β单调递增但变化平缓
betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps) ** 2
```

**噪声调度曲线**：
```
β_t: 从0.00085逐渐增加到0.0120
α_t: 从0.99915逐渐减小到0.9880
ᾱ_t: 从接近1逐渐衰减到接近0
```

**2. 累积乘积的重要性**

`alphas_cumprod` 是**扩散模型的核心参数**，它允许我们**直接计算任意时间步的加噪结果**，无需逐步迭代：

```
q(x_t | x_0) = N(√(ᾱ_t) x_0, (1-ᾱ_t)I)
```

这意味着：
- 已知原始图像 `x_0` 和噪声 `ε`
- 可直接计算第t步的加噪图像：`x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε`

---

### **推理时间步设置**

**时间步采样策略**

```python
def set_inference_timesteps(self, num_inference_steps=50):
    self.num_inference_steps = num_inference_steps
    step_ratio = self.num_train_timesteps // self.num_inference_steps
    timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
    self.timesteps = torch.from_numpy(timesteps)
```

**工作原理**：
- 训练时：1000个时间步（0-999）
- 推理时：通常用更少的步数（如50步）加速生成
- **均匀采样**：从1000步中均匀选取50个时间步
- **逆序排列**：从大（噪声多）到小（噪声少）

**示例**（1000训练步，50推理步）：
```python
# 步长比 = 1000 // 50 = 20
# 时间步序列 = [0, 20, 40, ..., 980]
# 逆序后 = [980, 960, ..., 0]
```

**为什么可以跳过时间步**：
- 扩散模型学习的是**连续的噪声预测**
- 在合理的时间步采样下，模型能泛化到中间状态
- 这是**加速扩散模型推理**的关键技术

---

### **图像到图像强度控制**

**强度参数的作用**

```python
def set_strength(self, strength=1):
    # 计算跳过的步数
    start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
    # 截断时间步序列
    self.timesteps = self.timesteps[start_step:]
    self.start_step = start_step
```

**强度`strength`的物理意义**：
- `strength=0.0`：从最后一步开始（t=0），几乎不添加噪声
- `strength=0.5`：从中间步开始
- `strength=1.0`：从第一步开始（t≈980），添加最大噪声

**图像到图像的工作流**：
```
原始图像 → VAE编码 → 添加噪声（到指定强度）→ 去噪生成 → VAE解码
```

**数学原理**：
```
# 根据强度选择起始步
start_t = int(strength * T)
# 对编码图像添加对应噪声
noisy_latents = add_noise(latents, start_t)
```

---

### **前向加噪过程**

**加噪函数实现**

```python
def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor):
    # 获取对应时间步的ᾱ_t
    alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    
    # 调整维度以支持广播
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    
    # 采样噪声
    noise = torch.randn(original_samples.shape, generator=self.generator, 
                       device=original_samples.device, dtype=original_samples.dtype)
    
    # 计算加噪结果：x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples
```

**维度调整技巧**：
- 输入：`original_samples` 形状为 `(B, C, H, W)`
- `sqrt_alpha_prod` 初始形状为 `(B,)` 或 `(1,)`
- 通过`unsqueeze(-1)`逐步扩展为 `(B, 1, 1, 1)`
- 支持与图像张量的广播运算

**加噪公式的物理意义**：
```
x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
```
- **信号部分**：`√(ᾱ_t) * x_0`，保留原始信息
- **噪声部分**：`√(1-ᾱ_t) * ε`，添加的高斯噪声
- 当`t→∞`时，`ᾱ_t→0`，`x_t→纯噪声`

---

### **反向去噪过程（核心）**

**1. 单步去噪算法**

这是DDPM采样的**核心实现**，对应论文中的公式(7)和(15)。

```python
def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
    t = timestep
    prev_t = self._get_previous_timestep(t)  # 计算上一步的时间
```

**2. 计算相关参数**

```python
# 1. 计算α和β的各种累积形式
alpha_prod_t = self.alphas_cumprod[t]          # ᾱ_t
alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one  # ᾱ_{t-1}
beta_prod_t = 1 - alpha_prod_t                 # 1-ᾱ_t
beta_prod_t_prev = 1 - alpha_prod_t_prev       # 1-ᾱ_{t-1}
current_alpha_t = alpha_prod_t / alpha_prod_t_prev  # α_t
current_beta_t = 1 - current_alpha_t           # β_t
```

**参数关系**：
- `ᾱ_t = ∏_{s=1}^{t} α_s`
- `α_t = ᾱ_t / ᾱ_{t-1}`
- `β_t = 1 - α_t`

**3. 预测原始样本（去噪方向）**

```python
# 2. 从预测噪声计算预测的原始样本 x_0
# 公式(15)：x_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √(ᾱ_t)
pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
```

**物理意义**：
- 给定带噪图像 `x_t` 和预测的噪声 `ε_θ`
- 反推原始干净图像 `x_0` 的估计值
- 这是**去噪过程的关键中间步骤**

**4. 计算混合系数**

```python
# 4. 计算 x_0 和 x_t 的混合系数
# 公式(7)中的系数分解
pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t
```

**系数来源**（公式7推导）：
```
x_{t-1} = μ_θ(x_t, t) + σ_t z
其中：
μ_θ = (√(ᾱ_{t-1})β_t)/(1-ᾱ_t) * x_0_est + (√(α_t)(1-ᾱ_{t-1}))/(1-ᾱ_t) * x_t
```

**5. 计算预测的均值**

```python
# 5. 计算预测的上一步样本 μ_t
# μ_θ = coeff1 * x_0_est + coeff2 * x_t
pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
```

**均值`μ_θ`的物理意义**：
- 是 `x_{t-1}` 分布的**均值**
- 由两部分线性组合：
  1. 预测的干净图像 `x_0_est`（去噪目标）
  2. 当前带噪图像 `x_t`（当前状态）

**6. 添加随机性（方差）**

```python
# 6. 添加噪声（方差项）
variance = 0
if t > 0:
    # 采样随机噪声
    noise = torch.randn(model_output.shape, generator=self.generator, 
                       device=model_output.device, dtype=model_output.dtype)
    # 计算方差：σ_t^2 = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
    variance = (self._get_variance(t) ** 0.5) * noise

# 最终采样：x_{t-1} = μ_θ + σ_t * z
pred_prev_sample = pred_prev_sample + variance
```

**方差计算函数**：
```python
def _get_variance(self, timestep: int) -> torch.Tensor:
    prev_t = self._get_previous_timestep(timestep)
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
    current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
    
    # 方差公式：σ_t^2 = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
    variance = torch.clamp(variance, min=1e-20)  # 防止数值下溢
    
    return variance
```

**为什么`t=0`时不加噪声**：
- 最后一步（t=0）应该是**确定性**的
- 确保生成结果的**可重复性**
- 对应干净的图像估计

---

### **DDPM采样算法的数学原理**

**1. 完整采样公式（论文公式7）**

```
x_{t-1} = 1/√(α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t z
其中：
σ_t^2 = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
```

**2. 代码实现与公式的对应关系**

| 数学公式 | 代码实现 | 说明 |
|----------|----------|------|
| `ε_θ(x_t, t)` | `model_output` | U-Net预测的噪声 |
| `x_0 ≈ (x_t - √(1-ᾱ_t)ε_θ)/√(ᾱ_t)` | `pred_original_sample` | 估计的干净图像 |
| `μ_θ` 的系数计算 | `pred_original_sample_coeff`, `current_sample_coeff` | 线性组合系数 |
| `σ_t^2` | `_get_variance(t)` | 方差计算 |
| `z ∼ N(0, I)` | `noise` | 随机噪声采样 |

**3. 两种等价表示**

**表示1（原始论文）**：
```python
# 直接使用公式(7)
pred_prev_sample = 1/√(α_t) * (latents - (1-α_t)/√(1-ᾱ_t) * model_output) + σ_t * z
```

**表示2（代码实现）**：
```python
# 先估计x_0，再线性组合
x0_est = (latents - √(1-ᾱ_t)*model_output) / √(ᾱ_t)
pred_prev_sample = coeff1 * x0_est + coeff2 * latents + σ_t * z
```

**两种表示的等价性**：
- 数学上完全等价
- 代码实现选择了第二种，更清晰体现物理意义
- 将"预测x_0"和"混合当前状态"分离

---

### **关键技术细节**

**1. 时间步管理**

**获取前一时间步**：
```python
def _get_previous_timestep(self, timestep: int) -> int:
    # 根据推理步数计算步长
    prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
    return prev_t
```

**示例**：
- 训练步：1000，推理步：50
- 步长：1000//50 = 20
- 当前步t=980，前一步prev_t=960

**2. 数值稳定性**

**方差钳制**：
```python
variance = torch.clamp(variance, min=1e-20)
```
- 防止方差为0导致除零错误
- 确保对数运算的数值稳定性

**设备与数据类型管理**：
```python
# 确保张量在正确设备和数据类型
alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, 
                                         dtype=original_samples.dtype)
```

**3. 随机性控制**

**生成器传递**：
```python
noise = torch.randn(..., generator=self.generator, ...)
```
- 使用统一的随机数生成器
- 确保可重复性（相同种子产生相同结果）
- 支持条件生成中的确定性采样

---

### **在Stable Diffusion中的工作流程**

**1. 文本到图像生成**

```python
# 简化流程
for timestep in sampler.timesteps:  # 从大到小迭代
    # 预测噪声
    model_output = unet(latents, text_embeddings, time_embedding)
    # 采样器更新
    latents = sampler.step(timestep, latents, model_output)
```

**2. 图像到图像生成**

```python
# 添加指定强度的噪声
sampler.set_strength(strength)
latents = sampler.add_noise(encoded_image, start_timestep)
# 然后执行去噪循环
```

**3. 与CFG的配合**

```python
# 在生成函数中
if do_cfg:
    # 重复潜在表示
    model_input = latents.repeat(2, 1, 1, 1)
    # U-Net预测（条件和无条件）
    model_output = diffusion(model_input, context, time_embedding)
    # 分离结果
    output_cond, output_uncond = model_output.chunk(2)
    # CFG组合
    model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
# 采样器更新
latents = sampler.step(timestep, latents, model_output)
```

---

### **总结：DDPMSampler的核心作用**

**1. 理论到实践的桥梁**

DDPMSampler 将**扩散模型的数学理论**转化为**可执行的代码**：

| 理论概念 | 代码实现 |
|----------|----------|
| 前向扩散过程 | `add_noise` 方法 |
| 反向生成过程 | `step` 方法 |
| 噪声调度 | `betas`, `alphas_cumprod` |
| 采样算法 | DDPM公式实现 |

**2. 关键设计决策**

1. **模块化设计**：采样器独立于U-Net模型
2. **灵活性**：支持不同推理步数、强度控制
3. **数值稳定性**：钳制、设备管理、类型转换
4. **可重复性**：随机生成器控制

**3. 在生成式AI中的重要性**

DDPMSampler 是**扩散模型生成质量的关键**：
- **噪声调度**：影响生成图像的清晰度和多样性
- **采样策略**：决定去噪的平滑性和稳定性
- **强度控制**：实现图像到图像的灵活编辑
- **加速推理**：通过减少步数平衡质量与速度

**结论**：这个DDPMSampler实现是**DDPM算法的精确工程化表达**，它将复杂的概率扩散过程封装为简洁高效的接口，是Stable Diffusion等现代生成模型能够稳定、高效工作的基石。通过精心的数学实现和工程优化，使得从文本描述生成高质量图像成为可能。