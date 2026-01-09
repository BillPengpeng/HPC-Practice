本文主要整理pytorch-quantization的主要内容。

## 6 - quantization_from_scratch

这段代码实现了两种常见的神经网络量化方法：**非对称量化**和**对称量化**。下面我将逐部分解释其工作原理。

### 🔧 辅助函数：clamp

```python
def clamp(params_q: np.array, lower_bound: int, upper_bound: int) -> np.array:
    params_q[params_q < lower_bound] = lower_bound
    params_q[params_q > upper_bound] = upper_bound
    return params_q
```
这个函数确保量化后的值落在指定范围内：
- **功能**：将数组中小于下限的值设为下限，大于上限的值设为上限
- **用途**：防止量化溢出，确保所有值都在有效的整数表示范围内

### ⚖️ 非对称量化

```python
def asymmetric_quantization(params: np.array, bits: int) -> Tuple[np.array, float, int]:
    alpha = np.max(params)  # 最大值
    beta = np.min(params)   # 最小值
    scale = (alpha - beta) / (2**bits-1)  # 缩放因子
    zero = -1*np.round(beta / scale)       # 零点偏移
    lower_bound, upper_bound = 0, 2**bits-1  # 8比特范围[0, 255]
    
    quantized = clamp(np.round(params / scale + zero), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale, zero
```

**工作原理**：
- **缩放因子计算**：`scale = (最大值 - 最小值) / (2^bits - 1)`，将浮点范围映射到整数范围
- **零点偏移**：`zero = -round(最小值 / scale)`，将浮点零点对齐到整数零点
- **量化公式**：`quantized = round(浮点数 / scale + zero)`

**适用场景**：数据分布不对称时效果更好

### ⚖️ 对称量化

```python
def symmetric_quantization(params: np.array, bits: int) -> Tuple[np.array, float]:
    alpha = np.max(np.abs(params))  # 最大绝对值
    scale = alpha / (2**(bits-1)-1)  # 缩放因子
    lower_bound = -2**(bits-1)       # 有符号整数下限
    upper_bound = 2**(bits-1)-1       # 有符号整数上限
    
    quantized = clamp(np.round(params / scale), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale
```

**工作原理**：
- **基于对称范围**：使用最大绝对值确定范围 `[-α, α]`
- **量化公式**：`quantized = round(浮点数 / scale)`，无需零点偏移
- **整数范围**：8比特时为 `[-128, 127]` 或 `[-127, 127]`

**优势**：计算更简单，硬件实现更高效

### 🔄 反量化函数

```python
def asymmetric_dequantize(params_q: np.array, scale: float, zero: int) -> np.array:
    return (params_q - zero) * scale

def symmetric_dequantize(params_q: np.array, scale: float) -> np.array:
    return params_q * scale
```

**功能**：将量化后的整数恢复为浮点数，用于推理计算

### 📊 量化误差评估

```python
def quantization_error(params: np.array, params_q: np.array):
    return np.mean((params - params_q)**2)
```

**用途**：计算原始浮点数与反量化后数值的均方误差，评估量化质量

### 🚀 实际调用示例

```python
(asymmetric_q, asymmetric_scale, asymmetric_zero) = asymmetric_quantization(params, 8)
(symmetric_q, symmetric_scale) = symmetric_quantization(params, 8)
```

**执行流程**：
1. 对输入参数 `params` 分别进行非对称和对称量化
2. 返回量化后的整数数组、缩放因子和零点（非对称量化）
3. 8比特量化将32位浮点数压缩为8位整数，减少75%存储空间

### 💡 核心区别总结

| 特性 | 非对称量化 | 对称量化 |
|------|------------|----------|
| **范围映射** | `[β, α]` → `[0, 2^bits-1]` | `[-α, α]` → `[-2^(bits-1), 2^(bits-1)-1]` |
| **零点偏移** | 需要 | 不需要 |
| **计算复杂度** | 较高 | 较低 |
| **适用场景** | 数据分布不对称 | 数据分布对称或接近对称 |

## 7 - quantization_compare_minmax_percentile

### 📊 非对称量化函数（百分位数法）

**改进点**：使用百分位数替代最小/最大值，减少异常值对量化范围的干扰，提高主体数据的精度。

```python
def asymmetric_quantization_percentile(params: np.array, bits: int, percentile: float = 99.99) -> Tuple[np.array, float, int]:
    alpha = np.percentile(params, percentile)       # 上百分位数（如99.99%）
    beta = np.percentile(params, 100 - percentile)  # 下百分位数（如0.01%）
    scale = (alpha - beta) / (2**bits - 1)
    zero = -1 * np.round(beta / scale)
    lower_bound, upper_bound = 0, 2**bits - 1
    quantized = clamp(np.round(params / scale + zero), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale, zero
```
- **百分位数的优势**：例如，`percentile=99.99` 会忽略分布中最高0.01%和最低0.01%的极端值，使缩放因子更贴合主体数据分布，降低异常值引起的量化误差。
- **适用场景**：当输入数据包含显著离群点时（如模型激活值），这种方法通常比最小-最大值法更鲁棒。

## 8 - post_training_quantization

这是一个完整的PyTorch训练后静态量化（Post-Training Static Quantization）实现代码。

### 1. 量化网络定义

```python
class QuantizedVerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(QuantizedVerySimpleNet,self).__init__()
        self.quant = torch.quantization.QuantStub()    # 量化入口
        self.linear1 = nn.Linear(28*28, hidden_size_1) 
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()  # 反量化出口

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = x.contiguous()    # 确保张量内存连续
        x = self.quant(x)      # 将输入量化为int8
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.dequant(x)    # 将输出反量化为float32
        return x
```

- **QuantStub()**: 在推理时会将float32输入转换为int8
- **DeQuantStub()**: 在输出前将int8转换回float32，便于后续处理
- **contiguous()**: 确保张量在内存中连续存储，避免某些量化操作出错

### 2. 模型初始化与权重复制

```python
device = "cpu"
net_quantized = QuantizedVerySimpleNet().to(device)
# Copy weights from unquantized model
net_quantized.load_state_dict(net.state_dict())
net_quantized.eval()
```

这部分代码将预训练好的浮点模型权重加载到量化模型中，为后续的量化做准备。

### 3. 量化配置与准备阶段

```python
net_quantized.qconfig = torch.ao.quantization.default_qconfig
net_quantized = torch.ao.quantization.prepare(net_quantized)  # 插入观察器
```

- **qconfig设置**: 使用默认量化配置，指定如何量化激活值和权重
- **prepare()**: 在模型中插入**观察器（Observer）**，用于在校准阶段收集张量的统计信息（如最小值、最大值）

### 4. 模型转换（实际量化）

```python
net_quantized = torch.ao.quantization.convert(net_quantized)
```

这是最关键的一步，执行以下操作：
1. 使用观察器收集的统计信息计算**scale（缩放因子）**和**zero_point（零点）**
2. 将权重从float32量化为int8
3. 将观察器替换为实际的量化操作
4. 生成真正执行低精度计算的量化模型

### 5. 量化原理与优势

量化基于线性映射：`quantized_value = round(float_value / scale) + zero_point`

## 9 - quantization_aware_training

这段代码演示了如何使用PyTorch实现**量化感知训练（QAT）**，完整展示了从模型定义、QAT准备、训练到最终转换的整个流程。

### 🧠 模型定义与量化存根

```python
class VerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(VerySimpleNet,self).__init__()
        self.quant = torch.quantization.QuantStub()    # 量化入口
        self.linear1 = nn.Linear(28*28, hidden_size_1) 
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()  # 反量化出口

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.quant(x)      # 将输入量化为int8
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.dequant(x)    # 将输出反量化为float32
        return x
```
- **QuantStub与DeQuantStub**：这两个存根分别标记了模型的**量化起点**和**反量化终点**。在前向传播中，它们会在训练阶段模拟量化过程（称为伪量化），即对数据执行量化再立即反量化，以引入量化误差，但保持浮点计算。
- **网络结构**：这是一个简单的全连接网络，适用于MNIST数据集（输入尺寸28×28，输出10类）。

### ⚙️ 量化配置与QAT准备

```python
net = VerySimpleNet().to(device)  
net.qconfig = torch.ao.quantization.default_qconfig  # 设置量化配置
net.train()  # 设置为训练模式
net_quantized = torch.ao.quantization.prepare_qat(net)  # 准备QAT
```
- **qconfig**：此为**量化配置文件**，它决定了如何量化权重和激活值（如对称/非对称量化、量化位数等）。`default_qconfig` 是PyTorch提供的默认配置，通常针对x86 CPU（使用`fbgemm`后端）或ARM CPU（使用`qnnpack`后端）进行优化。
- **prepare_qat()**：这是QAT的核心准备步骤。它会：
    - 在网络中插入**伪量化节点**，这些节点在前向传播时模拟INT8量化的舍入和截断误差。
    - 替换特定的模块（如`nn.Linear`）为支持量化感知训练的版本。
    - 设置**观测器**，用于在校准阶段收集张量的数值范围（min/max），从而计算缩放因子（scale）和零点（zero_point）。

### 🏋️ QAT训练过程

```python
def train(train_loader, net, epochs=5, total_iterations_limit=None):
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # ... 训练循环 ...
    for data in train_loader:
        x, y = data
        optimizer.zero_grad()
        output = net(x.view(-1, 28*28))  # 前向传播（包含伪量化）
        loss = cross_el(output, y)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
```
- **关键特性**：在QAT模式下，训练过程与标准训练类似，但前向传播中包含了**伪量化操作**。这意味着模型权重在反向传播和更新时，能“感知”到量化带来的精度损失，从而学习调整以适应低精度表示，这通常比训练后量化（PTQ）获得更好的精度。
- **目的**：通过训练让模型权重**适应量化噪声**，找到对量化不敏感的平坦最优区域，使得最终转换为真正INT8模型时精度损失最小。

### 🔄 模型转换与评估

```python
net_quantized.eval()  # 切换为评估模式
net_quantized = torch.ao.quantization.convert(net_quantized)  # 转换为量化模型
```
- **convert()**：这是QAT流程的最后一步。它会：
    - 移除训练时插入的伪量化节点。
    - 将FP32权重**永久转换为INT8**（使用训练和校准过程中确定的量化参数）。
    - 将模块替换为真正的量化实现，生成一个**可用于高效推理的INT8模型**。
- **模型大小**：`print_size_of_model`函数展示了量化后的模型大小，INT8模型相比原始FP32模型**通常可减少约75%的存储空间**。

### 💎 核心总结

这段代码完整实现了量化感知训练（QAT）的核心流程：**准备模型 → 配置量化 → 插入伪量化节点并训练 → 转换为最终INT8模型**。QAT通过在训练中模拟量化误差，让模型自适应调整，是平衡模型精度与推理效率的有效方法，特别适合对精度要求较高的部署场景。与训练后量化（PTQ）相比，QAT通常精度更高，但需要额外的训练时间。