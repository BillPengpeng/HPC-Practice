本文主要整理Assignment 1 (basics): Building a Transformer LM的主要内容。

## 4 Training a Transformer LM

## 4.1 Cross-entropy loss

### **1. 训练Transformer语言模型的核心组件（第1张图）**  
1. **训练所需的基础模块**  
   在完成数据预处理（通过tokenizer）和模型定义（Transformer架构）后，需进一步构建支持训练的完整代码，主要包括：  
   - **损失函数**：标准交叉熵（Negative Log-Likelihood）；  
   - **优化器**：AdamW（用于最小化损失函数）；  
   - **训练循环**：包含数据加载、检查点保存及训练过程管理的基础框架。  

2. **交叉熵损失函数的定义与推导**  
   - **模型输出的分布定义**：对于长度为 $m+1$ 的输入序列 $x$（索引 $i=1,...,m$），Transformer语言模型为每个位置 $i$ 定义了一个条件概率分布 $p_{\theta}(x_{i+1} \mid x_{1:i})$，即基于前 $i$ 个token预测第 $i+1$ 个token的概率。  
   - **标准交叉熵损失公式**：  
     $$
     \ell(\theta;D)=\frac{1}{|D|m}\sum_{x\in D}\sum_{i=1}^{m}-\log p_{\theta}(x_{i+1}\mid x_{1:i})
     $$  
     其中 $|D|$ 是训练集样本数，$m$ 是序列长度，求和覆盖所有样本的所有位置。  
   - **与模型输出的关系**：  
     - Transformer的前向传播会直接输出每个位置 $i$ 的 **logits向量 $o_i \in \mathbb{R}^{\text{vocab\_size}}$**（词汇表大小的实数向量）；  
     - 通过softmax函数将logits转换为概率分布：  
       $$
       p(x_{i+1} \mid x_{1:i}) = \text{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\text{vocab\_size}}\exp(o_i[a])}
       $$  
       即第 $i+1$ 个token的预测概率是其对应logits值的指数除以所有vocab词对应logits指数的总和。  


### **2. 模型评估的关键指标：困惑度（第2张图）**  
1. **困惑度的定义与作用**  
   - 交叉熵损失可直接用于模型训练，但在评估模型性能时，通常需要额外报告 **困惑度（Perplexity）**——这是一个更直观的指标，反映模型对测试数据的“预测不确定性”（值越低，模型越好）。  

2. **困惑度的计算公式**  
   对于长度为 $m$ 的序列，若其各位置的交叉熵损失值为 $\ell_1, \ell_2, ..., \ell_m$，则困惑度定义为：  
   $$
   \text{perplexity} = \exp\left(\frac{1}{m}\sum_{i=1}^{m}\ell_i\right)
   $$  
   即所有位置交叉熵损失的平均值取指数。  

3. **公式注释与概率解释**  
   - **注释6**：logits向量 $o_i$ 中索引 $k$ 处的值记为 $o_i[k]$（即第 $k$ 个词汇对应的logit值）。  
   - **注释7**：该交叉熵实际对应于 **真实token $x_{i+1}$ 的狄拉克δ分布（即确定性地选择正确token）与模型预测的softmax分布之间的差异**，本质是衡量模型预测与真实标签的匹配程度。  

### **总结**  
这两张图片完整呈现了Transformer语言模型从训练到评估的核心技术细节：训练阶段通过交叉熵损失指导模型学习序列中token的条件概率分布（依赖logits和softmax），而评估阶段则通过困惑度（基于交叉熵的平均指数）量化模型的整体预测能力。两者共同构成了语言模型开发的关键闭环。

## 4.2 The SGD Optimizer

### 内容概括  
在定义好损失函数后，从随机初始化参数出发，每一步迭代都依据当前小批量（random batch）数据计算的梯度来更新模型参数。  

### 要点总结  
1. **背景**：在有了损失函数（loss function）之后，开始探讨优化器（optimizer）；SGD 是最简单的基于梯度的优化器。  
2. **初始化**：参数 $\theta$ 从随机初始值 $\theta_0$ 开始。  
3. **迭代更新**：在第 $t$ 步（$t = 0, \dots, T-1$），按如下规则更新参数：  
   $$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t)$$  
   其中：  
   - $B_t$ 是从整个数据集 $D$ 中采样得到的**随机小批量（random batch）**数据；  
   - $\alpha_t$ 是**学习率（learning rate）**，控制每一步更新的步长；  
   - $\nabla L(\theta_t; B_t)$ 是损失函数 $L$ 在当前参数 $\theta_t$ 和小批量 $B_t$ 上的梯度；  
   - 学习率 $\alpha_t$ 和小批量大小（batch size）都属于**超参数（hyperparameters）**，需要在训练前设定或调整。  

## 4.2.1 Implementing SGD in PyTorch

### 内容概括  
这两张图片围绕 **PyTorch 中随机梯度下降（SGD）优化器的自定义实现与典型训练流程** 展开，核心内容包括两部分：  
1. **自定义 SGD 优化器的实现方法**（第1张图）：介绍如何通过继承 `torch.optim.Optimizer` 基类，实现包含 `__init__()`（初始化参数与超参数）和 `step()`（执行单步参数更新）的核心方法，并以“学习率随时间衰减的 SGD 变体”为例，展示关键代码逻辑（如学习率公式 $\theta_{t+1}=\theta_{t}-\frac{\alpha}{\sqrt{t+1}}\nabla L(\theta_{t};B_{t})$）。  
2. **优化器的参数更新逻辑与训练循环示例**（第2张图）：详细说明优化器内部如何管理参数状态（如迭代次数 `t`）、基于公式更新权重（如 $p.data -= lr / \sqrt{t + 1} * grad$），并提供一个完整的 **最小训练循环示例**（包括梯度清零、损失计算、反向传播、优化器执行步骤），强调该结构与语言模型训练的基本流程一致。  

---

### 要点总结  

#### 一、自定义 SGD 优化器的实现（第1张图）  
1. **核心方法要求**  
   - 继承 `torch.optim.Optimizer` 基类，必须实现两个方法：  
     - `__init__(self, params, ...)`：初始化优化器，接收待优化参数 `params`（参数组集合）和其他超参数（如学习率 `lr`），并通过 `super().__init__(params, defaults)` 将参数注册到基类（`defaults` 是存储超参数的字典，例如 `{"lr": lr}`）。  
     - `step(self, closure=None)`：执行单步参数更新（在训练循环中，此方法在反向传播后调用，可直接访问参数梯度 `p.grad`），需遍历参数组并修改参数张量 `p.data`（原地更新）。  

2. **学习率衰减的 SGD 变体示例**  
   - 目标：实现学习率随时间衰减的 SGD，更新公式为：  
     $$
     \theta_{t+1}=\theta_{t}-\frac{\alpha}{\sqrt{t+1}} abla L(\theta_{t};B_{t})
    $$  
     其中 $\alpha$ 是初始学习率，$t$ 是迭代次数，$\nabla L$ 是当前小批量数据的梯度。  
   - 代码关键点：  
     - `__init__()` 中检查学习率合法性（如 `lr < 0` 报错），并通过 `defaults` 存储超参数。  
     - `step()` 中遍历参数组（`param_groups`），获取当前学习率 `lr`，并针对每个参数 `p`（检查 `p.grad` 是否存在），结合迭代次数 `t` 计算衰减后的学习率（$\alpha / \sqrt{t+1}$）更新参数。  

---

#### 二、优化器的参数更新逻辑与训练循环（第2张图）  
1. **参数状态管理与更新细节**  
   - **状态记录**：每个参数 `p` 的迭代次数 `t` 通过 `self.state[p]` 字典管理（初始值为 0，每次更新后递增）。  
   - **更新公式**：基于学习率衰减的 SGD，权重更新逻辑为：  
     $$
     p.data -= \frac{lr}{\sqrt{t + 1}} \times grad
     $$  
     其中 `lr` 是初始学习率，`t` 是当前迭代次数（从状态中读取并更新）。  

2. **最小训练循环示例**  
   - **典型结构**：  
     ```python
     # 初始化参数和优化器
     weights = torch.nn.Parameter(5 * torch.randn(10, 10))  # 待优化参数
     opt = SGD([weights], lr=1)  # 自定义 SGD 优化器

     # 训练循环（100 次迭代）
     for t in range(100):
         opt.zero_grad()      # 清零梯度
         loss = (weights**2).mean()  # 计算损失（示例：均方误差）
         print(loss.cpu().item())
         loss.backward()      # 反向传播计算梯度
         opt.step()           # 执行优化器更新步骤
     ```  
   - **关键步骤**：  
     - `zero_grad()`：清除上一轮迭代的梯度（避免累积）。  
     - `loss.backward()`：计算当前损失对参数的梯度（存储在 `p.grad` 中）。  
     - `opt.step()`：调用自定义优化器的 `step()` 方法，按公式更新参数。  
   - **通用性说明**：语言模型训练时，参数通过 `model.parameters()` 获取，损失基于小批量数据计算，但训练循环的基本结构（梯度清零→损失计算→反向传播→优化器更新）完全一致。  

---

### Problem (learning_rate_tuning): Tuning the learning rate

As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s
see that in practice in our toy example. Run the SGD example above with three other values for the
learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each
of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of
training)?
- Deliverable: A one-two sentence response with the behaviors you observed. 
- 三种lr 现象类似，1e2和le3 loss收敛状况好于1e1

## 4.3 AdamW

[Adam与AdamW](https://zhuanlan.zhihu.com/p/1932578309281158770)

### 内容概括

这张图片详细介绍了 **AdamW 优化器**，这是现代语言模型训练中广泛使用的先进优化算法。主要内容包括：

1. **AdamW 的背景与重要性**：作为 Adam 优化器的改进版本，AdamW 通过解耦权重衰减机制来增强正则化效果，在现代深度学习特别是大语言模型（如 LLaMA、GPT-3）训练中占据主导地位。

2. **算法原理与实现**：提供了完整的 AdamW 算法伪代码，详细说明了其状态管理机制（维护一阶矩和二阶矩估计）、参数更新规则以及权重衰减的应用方式。

3. **超参数配置**：介绍了关键超参数的典型设置，包括动量参数 (β₁, β₂) 的常用值（0.9, 0.999）以及大语言模型中采用的调整值（0.9, 0.95）。

4. **内存与性能权衡**：指出 AdamW 需要额外内存来存储状态信息，但能提供更好的训练稳定性和收敛性能。

### 要点总结

#### 一、AdamW 的核心特性
- **改进目的**：解决 Adam 优化器中权重衰减与梯度更新耦合的问题，提供更有效的正则化
- **关键创新**：将权重衰减与梯度更新解耦，单独应用于参数更新
- **应用领域**：现代大语言模型（LLaMA、GPT-3等）的标准优化器

#### 二、算法实现细节
1. **状态管理**：
   - 维护两个状态向量：**一阶矩估计 m**（动量）和 **二阶矩估计 v**（自适应学习率）
   - 初始值均为 0，形状与参数 θ 相同

2. **更新步骤**：
   - **梯度计算**：$g \leftarrow \nabla_{\theta}\ell(\theta;B_{t})$
   - **矩估计更新**：
     - $m \leftarrow \beta_{1}m + (1-\beta_{1})g$ （一阶矩，动量更新）
     - $v \leftarrow \beta_{2}v + (1-\beta_{2})g^{2}$ （二阶矩，平方梯度更新）
   - **学习率调整**：$\alpha_{t} \leftarrow \alpha\frac{\sqrt{1-(\beta_{2})^{t}}}{1-(\beta_{1})^{t}}$ （偏差校正）
   - **参数更新**：$\theta \leftarrow \theta - \alpha_{t}\frac{m}{\sqrt{v}+\epsilon}$ （自适应更新）
   - **权重衰减**：$\theta \leftarrow \theta - \alpha\lambda\theta$ （**解耦的正则化**）

#### 三、超参数设置
- **学习率 α**：主要调节参数，控制更新步长
- **动量参数**：
  - $\beta_1$：一阶矩衰减率，通常设为 0.9
  - $\beta_2$：二阶矩衰减率，通常设为 0.999（标准）或 0.95（大语言模型）
- **权重衰减率 λ**：控制正则化强度
- **数值稳定常数 ε**：极小值（如 $10^{-8}$），防止除零错误

#### 四、优势与代价
- **优势**：
  - 自适应学习率，适合不同参数
  - 动量加速收敛
  - 解耦权重衰减提供更好正则化
- **代价**：
  - 需要额外内存存储 m 和 v 状态
  - 计算复杂度略高于 SGD

#### 五、实际应用
- **典型设置**：$(\beta_{1},\beta_{2}) = (0.9,0.999)$
- **大语言模型设置**：$(\beta_{1},\beta_{2}) = (0.9,0.95)$ （LLaMA、GPT-3）
- **数值稳定性**：通过 ε 参数避免数值问题

AdamW 通过结合动量、自适应学习率和解耦权重衰减，为现代深度神经网络训练提供了稳定高效的优化方案，成为大语言模型训练的事实标准。

### Problem (adamwAccounting): Resource accounting for training with AdamW

(a) How much peak memory does running AdamW require? Decompose your answer based on the
memory usage of the parameters, activations, gradients, and optimizer state. Express your answer
in terms of the batch_size and the model hyperparameters (vocab_size, context_length,
num_layers, d_model, num_heads). Assume d_ff = 4 × d_model.
- For simplicity, when calculating memory usage of activations, consider only the following compo-
nents:
   - Transformer block: RMSNorm(s); Multi-head self-attention sublayer: QKV projections, $Q^TK$ matrix multiply, softmax,
      weighted sum of values, output projection; Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply.
   - final RMSNorm; output embedding; cross-entropy on logits
- Deliverable: An algebraic expression for each of parameters, activations, gradients, and optimizer state, as well as the total.

- 参数内存 (Parameters Memory): num_layers * (4*d_model^2 + 2*d_model + 3*d_model*d_ff) + d_model + d_model*vocab_size = num_layers * (16*d_model^2 + 2*d_model) + d_model + d_model*vocab_size = num_layers * 16*d_model^2 + (2*num_layers + 1 + vocab_size) * d_model
- 激活值内存 (Activations Memory): num_layers * (seq_len*3*d_model + num_heads*seq_len*seq_len + seq_len*d_model + seq_len*d_model) + num_layers * (2*seq_len*d_ff + 2*seq_len*d_model) + seq_len*d_model + seq_len*vocab_size + seq_len*vocab_size = num_layers*num_heads*seq_len*seq_len + (15*num_layers + 1)*seq_len*d_model + 2*seq_len*vocab_size
- 梯度内存 (Gradients Memory) = 参数内存 (Parameters Memory)
- 优化器状态内存 (Optimizer State Memory) = 2 * 参数内存 (Parameters Memory)
- Peak Memory Total = 4 * 参数内存 (Parameters Memory) + 激活值内存 (Activations Memory) 

(b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on
the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?
Deliverable: An expression that looks like a · batch_size + b for numerical values a, b, and a
number representing the maximum batch size.

- a = 激活值内存 (Activations Memory)  = 9.47Gb
- b = 4*参数内存 (Parameters Memory) = 23.16GB
- batchsize = 6

(c) How many FLOPs does running one step of AdamW take?
- Deliverable: An algebraic expression, with a brief justification.
- ​常用近似值：10N​FLOPs​​

(d) Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second)
relative to the hardware’s theoretical peak FLOP throughput [Chowdhery et al., 2022]. An
NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming
you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a
batch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022],
assume that the backward pass has twice the FLOPs of the forward pass.
- Deliverable: The number of days training would take, with a brief justification.
- 'total_flops': 0.45133365248, 'adam_flops': 5.793709635734558e-12
- (0.45133365248 * 1024 * 400 * 1000) / (19.5 * 0.5) = 18904615s = 218 days
- (0.9555 * 1024 * 400 * 1000) / (19.5 * 0.5) = 465 days


## 4.4 Learning rate scheduling

### 内容概况

这张图片是关于 **Transformer模型训练中的学习率调度策略** 的技术文档，重点介绍了 **余弦退火学习率调度（cosine annealing learning rate schedule）** 方法。该方法是LLaMA等现代大语言模型训练中采用的关键技术，通过动态调整学习率来优化训练过程。

### 要点总结

#### 1. 学习率调度的重要性
1. **动态调整需求**：在训练过程中，能够最快降低损失的最佳学习率值会不断变化。
2. **Transformer训练惯例**：通常采用**先大后小**的策略——开始时使用较大的学习率进行快速更新，随后逐渐衰减到较小的值。
3. **调度器定义**：一个以当前训练步数t和其他相关参数为输入，返回该步应使用的学习率的函数。

#### 2. 余弦退火调度器的具体实现
该调度器需要五个参数：
- 当前迭代步数 `t`
- 最大学习率 `α_max`
- 最小（最终）学习率 `α_min`
- 预热迭代次数 `T_w`
- 余弦退火迭代次数 `T_c`

##### 三阶段学习率计算：

1. **预热阶段（Warm-up）** - `t < T_w`
   - 公式：`α_t = (t / T_w) * α_max`
   - 作用：学习率从0线性增加到最大值，避免训练初期的不稳定。

2. **余弦退火阶段（Cosine annealing）** - `T_w ≤ t ≤ T_c`
   - 公式：`α_t = α_min + 0.5 * (1 + cos(π*(t - T_w)/(T_c - T_w))) * (α_max - α_min)`
   - 作用：学习率按余弦函数从最大值平滑衰减到最小值。

3. **退火后阶段（Post-annealing）** - `t > T_c`
   - 公式：`α_t = α_min`
   - 作用：保持最小学习率进行后续训练。

#### 3. 技术背景与应用
- **学术引用**：该方法引用自Touvron et al. 2023的论文，是训练LLaMA模型采用的策略。
- **对比基础**：最简单的调度器是常数函数，始终返回相同的学习率。
- **实践意义**：这种调度策略在现代大语言模型训练中已成为标准实践，能够有效平衡训练速度与最终性能。

## 4.5 Gradient clipping

### 内容概况

这张图片详细介绍了 **梯度裁剪（Gradient Clipping）** 技术，这是深度学习中用于稳定训练过程的重要方法。主要内容包括梯度裁剪的原理、数学实现方式以及具体的编程任务要求。该技术通过限制梯度的大小来防止训练过程中因过大梯度而导致的训练不稳定问题。

### 要点总结

#### 一、梯度裁剪的核心概念
1. **问题背景**：在训练过程中，某些训练样本可能产生**过大的梯度**，这些大梯度会破坏训练的稳定性，导致模型难以收敛。
2. **解决方案**：梯度裁剪通过**强制限制梯度范数**来缓解这个问题，在优化器执行更新步骤之前对梯度进行处理。

#### 二、数学原理与实现
1. **计算梯度范数**：对于所有参数的梯度 $g$，计算其 $\ell_2$-范数 $\left\|g\right\|_{2}$。
2. **裁剪条件判断**：
   - 如果 $\left\|g\right\|_{2} \leq M$（最大允许范数），保持梯度不变
   - 如果 $\left\|g\right\|_{2} > M$，按比例缩放梯度
3. **缩放公式**：$g_{\text{clipped}} = g \times \frac{M}{\left\|g\right\|_{2} + \epsilon}$
   - 其中 $\epsilon = 10^{-6}$ 是为了数值稳定性添加的小常数
   - 裁剪后的梯度范数将略小于 $M$

#### 三、编程任务要求
1. **函数实现**：需要编写一个实现梯度裁剪的函数
   - **输入**：参数列表和最大 $\ell_2$-范数 $M$
   - **操作**：原地修改每个参数的梯度（in-place modification）
   - **数值稳定性**：使用 $\epsilon = 10^{-6}$（PyTorch默认值）
2. **适配器实现**：需要实现 `adapters.run_gradient_clipping` 适配器
3. **测试验证**：确保实现能够通过特定的测试用例（`uv run pytest -k test_gradient_clipping`）

#### 四、技术特点
- **原位操作**：直接修改参数的梯度值，不创建新的张量
- **范数计算**：使用 $\ell_2$-范数（欧几里得范数）来衡量梯度大小
- **比例缩放**：保持梯度方向不变，只调整幅度大小
- **数值安全**：通过 $\epsilon$ 参数避免除零错误和数值不稳定

## 5 Training loop

## 5.1 Data Loader

### **内容概括**
两张图片分别描述了训练循环中的数据加载流程及大数据集的内存优化方案：
1. **第一张图**聚焦于训练循环的构建，重点说明如何将标记化数据、模型和优化器整合，并详细介绍了数据加载器（Data Loader）的设计逻辑，包括数据格式、批处理生成方式及其对训练效率的优化价值。
2. **第二张图**针对大规模数据集无法完全载入内存的问题，提出了基于Unix系统调用`mmap`的解决方案，并推荐了在Python中通过Numpy（`np.memmap`或`mmap_mode='r'`）实现内存映射加载的具体方法，同时强调了数据类型匹配与数据验证的重要性。

---

### **核心要点总结**

#### **第一张图要点：**
1. **数据整合**：标记化数据通常被拼接为单一序列（不同文档间添加分隔符），简化后续处理。
2. **批处理生成**：数据加载器将长序列转换为固定长度（$m$）的批次（$B$个序列），每个批次包含输入序列及对应的下一标记目标。
3. **训练优化优势**：
   - 无需填充：统一序列长度提升硬件利用率。
   - 无需全量加载：支持流式读取，适应超大规模数据集。

#### **第二张图要点：**
1. **内存映射技术（mmap）**：通过虚拟内存机制延迟加载磁盘文件，实现“伪内存载入”。
2. **Numpy实现方案**：
   - 使用`np.memmap`或`np.load(..., mmap_mode='r')`按需加载数据。
   - 需严格匹配数据类型（`dtype`）以避免错误。
3. **验证建议**：检查加载数据是否在预期范围内（如词汇表大小），确保数据完整性。

## 5.2 Checkpointing

### **内容概括**
该文档阐述了在模型训练过程中实施检查点机制的核心目的与技术方法。检查点用于定期保存训练状态，使得训练任务可从意外中断（如超时或硬件故障）中恢复，同时也为获取训练过程中的中间模型提供支持。文档指出，一个完整的检查点必须包含模型权重、优化器状态（如AdamW中的动量估计）及当前迭代次数等关键状态信息，并强调了PyTorch框架为此提供的原生工具链（如`state_dict()`、`load_state_dict()`及`torch.save()`/`torch.load()`函数）的便利性。

---

### **核心要点总结**

#### **1. 检查点的核心价值**
- **容错恢复**：确保训练任务因超时、机器故障等中断后可从最近状态继续，避免从头开始。
- **中间模型访问**：支持后续分析训练动态（如损失变化、权重演化）或在不同训练阶段获取模型快照（例如用于测试或生成样本）。

#### **2. 检查点应包含的完整状态**
- **模型权重**：通过`nn.Module.state_dict()`获取所有可学习参数。
- **优化器状态**：对于有状态优化器（如AdamW），需保存其内部变量（例如一阶和二阶动量估计）。
- **训练进度**：当前迭代次数（用于恢复学习率调度器等依赖迭代次数的组件）。

#### **3. PyTorch的技术实现**
- **状态序列化**：
  - `model.state_dict()` → 获取模型参数字典。
  - `optimizer.state_dict()` → 获取优化器状态字典。
- **存储与加载**：
  - `torch.save(obj, path)`：将对象（可包含张量或Python原生对象如整数）保存至文件。
  - `torch.load(path)`：从文件加载对象回内存。
- **状态恢复**：
  - `model.load_state_dict(state_dict)` → 恢复模型参数。
  - `optimizer.load_state_dict(state_dict)` → 恢复优化器状态。

#### **4. 设计建议**
- 检查点通常以字典形式组织（例如`{"model": model_state, "optimizer": optimizer_state, "iteration": step}`），通过`torch.save()`统一保存。
- 需确保保存和加载时的模型/优化器结构一致，否则`load_state_dict()`可能失败。

---

### **关键启示**
检查点机制是生产环境及长时训练任务中**不可或缺的可靠性保障措施**，PyTorch通过高度封装接口降低了实现门槛，开发者应优先将状态管理框架化而非临时实现。

## 6. Generating text

## 📋 内容概况

这张图片是文档的第6节"Generating text"，主要介绍了**如何从训练好的语言模型中生成文本**。内容包括：
- 语言模型输出的概率分布转换
- 基本的解码（生成）过程
- 两种改进生成质量的技巧：温度缩放和核采样

## 🎯 要点总结

### 1. **文本生成基本原理**
- **输入**：整数序列（可能批量处理）
- **输出**：概率分布矩阵（序列长度 × 词汇表大小）
- **核心操作**：通过softmax将logits转换为概率分布

### 2. **解码过程**
- **步骤**：
  1. 提供前缀词元（prompt）
  2. 模型预测下一个词元的概率分布
  3. 从分布中采样得到下一个词元
  4. 重复直到生成结束标记或达到最大长度

- **数学表达**：
  $$P(x_{t+1}=i\mid x_{1\ldots t}) = \frac{\exp(v_i)}{\sum_j\exp(v_j)}$$
  其中 $v = \text{TransformerLM}(x_{1\ldots t})_t$

### 3. **改进生成质量的技巧**

#### 🔥 温度缩放（Temperature Scaling）
- **公式**：
  $$\text{softmax}(v,\tau)_i = \frac{\exp(v_i/\tau)}{\sum_j\exp(v_j/\tau)}$$
- **作用**：
  - $\tau \rightarrow 0$：输出趋于one-hot（确定性更强）
  - $\tau \rightarrow 1$：保持原始分布
  - $\tau > 1$：分布更平滑（多样性更高）

#### 🎯 核采样/Top-p采样（Nucleus Sampling）
- **原理**：截断低概率词元，只从高概率集合中采样
- **计算步骤**：
  1. 对概率分布$q$按大小排序
  2. 选择最小的索引集合$V(p)$，使得$\sum_{j\in V(p)}q_j \geq p$
  3. 重新归一化概率分布并采样

### 4. **应用场景**
- **小模型问题**：小模型可能生成低质量文本，这些技巧可以改善质量
- **控制生成**：通过调整参数控制生成文本的创造性和确定性

## 💡 关键启示

1. **文本生成是迭代过程**：基于前面生成的词元预测下一个词元
2. **采样策略影响质量**：不同的解码策略会产生不同风格的文本
3. **参数调节重要**：温度参数$\tau$和核参数$p$需要根据任务调整
4. **实用性导向**：这些技巧特别有助于提升小模型的生成效果

## 7 Experiments

### Problem (batch_size_experiment): Batch size variations

| lr | batchsize | total_tokens | validation loss | perplexity | 
| :--: | :--: | :--: | :--: | :--: |
| 0.001 | 32 | 40960000 | 1.64 | 5.15 |
| 0.001 | 32 | 327680000 | 1.34 | 3.84 |
| 0.001 | 64 | 327680000 | 1.35 | 3.86 |
| 0.001 | 128 | 327680000 | 1.40 | 4.04 |

### Problem (generate): Generatetext

```python
uv run problem/pred.py --yaml_path=./cfg/tinystories_v1.0.yaml --model_path=./workdir/tinystories_v1.0_lr0.001_wd0.1_mln1.0_wi100_2/iter_40000.pth --prompt="Once upon a time"

>> Once upon a time, there was a little boy named Tim. Tim was an adventurous boy. He loved to play outside. One sunny day, Tim went to the park with his mom to play.
At the park, Tim saw a mosquito. He asked his mom, "Mom, can you help me with this mosquito?" His mom smiled and said, "Of course, Tim. Let's be one and catch it." They played near a bush and had a lot of fun.
Later, it started to get dark. Tim's mom said, "Tim, it's time to go home now." Tim picked up his kayak and began to paddle home. His mom helped him get away from the dark fireplace. They laughed and waved goodbye to the exciting park town.
 ```

## 7.3 Ablationsandarchitecturemodification

### Ablation1: layer normalization

| lr | batchsize | total_tokens | validation loss | perplexity | 备注 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 0.001 | 64 | 327680000 | 1.39 | 4.01 | Remove RMSNorm and train |
| 0.001 | 64 | 327680000 | 1.36 | 3.88 | Post norm |
| 0.001 | 64 | 327680000 | 1.33 | 3.80 | Add norm |

### Ablation 2: position embeddings

| lr | batchsize | total_tokens | validation loss | perplexity | 备注 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 0.001 | 64 | 327680000 | 1.39 | 4.01 | NoPE |

### Ablation 3: SwiGLU vs. SiLU

| lr | batchsize | total_tokens | validation loss | perplexity | 备注 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 0.001 | 64 | 327680000 | 1.38 | 3.99 | SiLU |