本文主要整理Assignment 5 (alignment)的主要内容。

## 7 Group Relative Policy Optimization

## 7.1 GRPO Algorithm

### 📊 内容概况

本段详细介绍了**GRPO（Group Relative Policy Optimization，组相对策略优化）算法**，这是一种专为语言模型设计的强化学习算法。内容涵盖了**优势估计、高层算法流程和GRPO目标**三个核心部分，重点解释了GRPO如何通过**组归一化奖励**来避免训练单独的价值函数，并结合了离线策略梯度和PPO裁剪机制来实现稳定高效的训练。

### 🎯 要点总结

#### 1. **优势估计的创新**
- **核心思想**：为每个问题从策略$π_θ$采样多个输出（G个），利用这些输出计算基线
- **避免价值函数**：不需要训练神经网络价值函数$V_φ(s)$，既简化了训练又避免了系统复杂性
- **组归一化奖励**：通过同一组输出的奖励计算标准化优势

#### 2. **高层算法流程**
- 参考Shao等人2024年的工作
- 整体训练循环包括：采样多个输出、计算组归一化优势、应用GRPO目标更新策略
- 允许在单批数据上进行多次梯度更新，提高数据效率

#### 3. **GRPO目标的三大思想**
1. **离线策略梯度**：使用重要性采样，允许用旧策略数据更新当前策略
2. **组归一化优势**：通过组内输出的奖励均值和标准差计算标准化优势
3. **裁剪机制**：借鉴PPO的裁剪思想，防止策略更新过大，保持训练稳定性

### 📈 打印公式

#### 公式(28)：组归一化优势计算
$$
A^{(i)} = \frac{r^{(i)} - \text{mean}(r^{(1)}, r^{(2)}, \ldots, r^{(G)})}{\text{std}(r^{(1)}, r^{(2)}, \ldots, r^{(G)}) + \text{advantage\_eps}}
$$

**符号说明**：
- $r⁽ⁱ⁾$ = $R(q, o⁽ⁱ⁾)$：第i个输出的奖励
- $mean(r⁽¹⁾, ..., r⁽ᴳ⁾)$：组内所有输出奖励的均值
- $std(r⁽¹⁾, ..., r⁽ᴳ⁾)$：组内所有输出奖励的标准差
- $advantage_eps$：防止除零的小常数

**注意**：这个优势$A⁽ⁱ⁾$在响应的每个token上都相同，因此在后续讨论中省略时间下标t。

### 🔬 技术细节深入

#### 1. **优势估计的工作原理**
```python
def compute_group_normalized_advantage(rewards, advantage_eps=1e-8):
    """
    计算组归一化优势
    """
    # 转换为张量
    rewards = torch.tensor(rewards)
    
    # 计算均值和标准差
    mean_reward = torch.mean(rewards)
    std_reward = torch.std(rewards)
    
    # 组归一化
    advantages = (rewards - mean_reward) / (std_reward + advantage_eps)
    
    return advantages

# 示例
rewards = [0.9, 0.8, 0.7, 0.6, 0.5]  # 5个输出的奖励
advantages = compute_group_normalized_advantage(rewards)
print(f"优势值: {advantages}")
```

#### 2. **与PPO的关系**
GRPO结合了PPO的两个关键思想：
```python
ppo_concepts_in_grpo = {
    "裁剪机制": "防止新旧策略差异过大，保持训练稳定",
    "重要性采样": "允许离线策略学习，提高数据效率", 
    "多步优化": "在单批数据上进行多次梯度更新"
}
```

#### 3. **离线策略梯度公式（提及的Eq. 27）**
虽然没有完全显示，但Eq. 27很可能指的是**离线策略梯度的一般形式**：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{(s,a)\sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \nabla_\theta \log \pi_\theta(a|s) A(s,a) \right]
$$

### 🎪 GRPO算法流程

```python
def grpo_algorithm():
    """
    GRPO算法高层流程
    """
    steps = [
        "1. 对每个提示q，从当前策略π_θ采样G个输出{o⁽ⁱ⁾}",
        "2. 计算每个输出的奖励r⁽ⁱ⁾ = R(q, o⁽ⁱ⁾)",
        "3. 使用公式(28)计算组归一化优势A⁽ⁱ⁾",
        "4. 构建损失函数，结合离线策略梯度和裁剪机制",
        "5. 在单批数据上进行多次梯度更新",
        "6. 重复上述过程直到收敛"
    ]
    
    return steps
```

### 📊 GRPO的优势分析

```python
grpo_advantages = {
    "简化架构": "无需单独的价值函数网络，减少参数量和训练复杂度",
    "稳定训练": "组归一化提供自适应的基线，裁剪防止策略突变",
    "数据高效": "离线策略允许数据重用，组内比较提高样本效率",
    "适应性强": "适用于各种奖励函数，特别适合语言模型任务"
}
```

### 🔧 实现注意事项

#### 1. **超参数选择**
```python
grpo_hyperparameters = {
    "组大小G": "通常8-32，需平衡计算成本和统计可靠性",
    "裁剪系数ε": "通常0.1-0.3，控制策略更新幅度", 
    "优势常数eps": "通常1e-8，防止除零错误",
    "批次大小": "根据GPU内存和任务复杂度调整"
}
```

#### 2. **实际实现示例**
```python
import torch
import torch.nn.functional as F

def grpo_loss(new_log_probs, old_log_probs, advantages, clip_epsilon=0.2):
    """
    计算GRPO损失（结合重要性采样和裁剪）
    """
    # 重要性权重
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 裁剪的重要性权重
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    
    # 裁剪的损失
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    
    return loss.mean()

# 使用示例
def compute_grpo_update(batch_data, policy_network, reward_function):
    """
    计算GRPO更新
    """
    prompts, old_outputs, old_log_probs = batch_data
    
    # 重新采样新输出
    with torch.no_grad():
        new_outputs = policy_network.sample(prompts, num_samples=8)  # G=8
    
    # 计算奖励
    rewards = []
    for prompt, outputs in zip(prompts, new_outputs):
        group_rewards = [reward_function(prompt, output) for output in outputs]
        rewards.append(group_rewards)
    
    # 计算组归一化优势
    advantages = []
    for group_rewards in rewards:
        group_advantages = compute_group_normalized_advantage(group_rewards)
        advantages.append(group_advantages)
    
    # 计算新策略的对数概率
    new_log_probs = policy_network.get_log_probs(prompts, new_outputs)
    
    # 计算GRPO损失
    loss = grpo_loss(
        new_log_probs.flatten(),
        old_log_probs.flatten(),
        torch.tensor(advantages).flatten()
    )
    
    return loss
```

## 7.1 GRPO优化目标

### 📊 内容概况

本图详细阐述了**GRPO-Clip算法的目标函数**，这是GRPO算法中引入**裁剪机制**的核心部分。内容从完整的GRPO-Clip目标函数（公式29）开始，逐步拆解分析每个token级别的目标，定义了控制策略更新幅度的裁剪函数g(ε, A⁽ⁱ⁾)（公式30），并分情况讨论了优势A⁽ⁱ⁾为正或负时目标函数的行为机制及其对策略更新的约束作用，最终揭示了裁剪机制如何确保新策略不过度偏离旧策略，从而保持训练稳定性。

### 🎯 要点总结

1. **GRPO-Clip目标函数结构**：
   - 是GRPO算法的裁剪版本，借鉴了PPO的裁剪思想
   - 在**每个生成的token级别**定义目标函数
   - 包含**重要性采样比**和**组归一化优势**的乘积

2. **裁剪函数g(ε, A⁽ⁱ⁾)的设计**：
   - 超参数ε>0控制策略可变化的最大幅度
   - 优势A⁽ⁱ⁾为正时，裁剪上界为(1+ε)
   - 优势A⁽ⁱ⁾为负时，裁剪下界为(1-ε)
   - 这种非对称裁剪适应了优势的符号

3. **目标函数的行为分析**：
   - **优势为正时**：鼓励增加对应token的概率，但被(1+ε)上限限制
   - **优势为负时**：鼓励减少对应token的概率，但被(1-ε)下限限制
   - 最终效果：防止单次更新中策略变化过大，保持训练稳定性

4. **裁剪机制的直观理解**：
   - 当新策略对某token的概率超过旧策略的(1+ε)倍时，目标函数不再增加
   - 当新策略对某token的概率低于旧策略的(1-ε)倍时，目标函数不再减少
   - 这形成了策略更新的"信赖域"，避免策略崩溃

### 📈 打印公式

#### 公式29：完整的GRPO-Clip目标函数
$$
J_{\text{GRPO-Clip}}(\theta) = \mathbb{E}_{q\sim\mathcal{D},\,\{o^{(i)}\}_{i=1}^{G}\sim\pi_{\theta}(\cdot\mid q)}
\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\left|o^{(i)}\right|}\sum_{t=1}^{\left|o^{(i)}\right|}\min\left(
\frac{\pi_{\theta}(o^{(i)}_{t}\mid q,o^{(i)}_{<t})}{\pi_{\theta_{\text{old}}}(o^{(i)}_{t}\mid q,o^{(i)}_{<t})}A^{(i)},
\operatorname{clip}\left(\frac{\pi_{\theta}(o^{(i)}_{t}\mid q,o^{(i)}_{<t})}{\pi_{\theta_{\text{old}}}(o^{(i)}_{t}\mid q,o^{(i)}_{<t})},1-\epsilon,1+\epsilon\right)A^{(i)}
\right)\right]
$$

**公式解析**：
- 外层期望：对问题q和G个输出o⁽ⁱ⁾采样
- 内层平均：对G个输出平均，对每个输出的所有token平均
- 核心是**min操作**：比较原始重要性采样比和裁剪后重要性采样比
- clip函数：将重要性采样比限制在[1-ε, 1+ε]范围内

#### 公式30：裁剪函数g(ε, A⁽ⁱ⁾)的定义
$$
g(\epsilon,A^{(i)})=\begin{cases}
(1+\epsilon)A^{(i)} & \text{if } A^{(i)}\geq 0 \\
(1-\epsilon)A^{(i)} & \text{if } A^{(i)}<0
\end{cases}
$$

#### 推导后的每个token目标函数
$$
\text{per-token objective} = \min\left(
\frac{\pi_{\theta}(o^{(i)}_{t}\mid q,o^{(i)}_{<t})}{\pi_{\theta_{\text{old}}}(o^{(i)}_{t}\mid q,o^{(i)}_{<t})}A^{(i)},
g(\epsilon,A^{(i)})
\right)
$$

#### 优势为正时的简化形式
$$
\text{per-token objective} = \min\left(
\frac{\pi_{\theta}(o^{(i)}_{t}\mid q,o^{(i)}_{<t})}{\pi_{\theta_{\text{old}}}(o^{(i)}_{t}\mid q,o^{(i)}_{<t})},
1+\epsilon
\right)A^{(i)}
\quad \text{当 } A^{(i)}>0
$$

#### 优势为负时的简化形式
$$
\text{per-token objective} = \min\left(
\frac{\pi_{\theta}(o^{(i)}_{t}\mid q,o^{(i)}_{<t})}{\pi_{\theta_{\text{old}}}(o^{(i)}_{t}\mid q,o^{(i)}_{<t})},
1-\epsilon
\right)A^{(i)}
\quad \text{当 } A^{(i)}<0
$$

### 🔬 技术细节深入

#### 1. **裁剪机制的数学行为**
```python
def clipping_behavior(ratio, advantage, epsilon=0.2):
    """裁剪机制的行为分析"""
    if advantage >= 0:
        # 优势为正：鼓励增加概率，但有上限
        clipped_ratio = min(ratio, 1 + epsilon)
        objective = clipped_ratio * advantage
    else:
        # 优势为负：鼓励减少概率，但有下限
        clipped_ratio = min(ratio, 1 - epsilon)  # 注意：ratio ≥ 0
        objective = clipped_ratio * advantage  # advantage为负，所以objective为负
    
    return objective

# 示例：观察不同ratio下的目标值
epsilon = 0.2
advantage = 1.0
for ratio in [0.5, 1.0, 1.5, 2.0]:
    obj = clipping_behavior(ratio, advantage, epsilon)
    print(f"ratio={ratio}: objective={obj}")
```

#### 2. **目标函数的实现示例**
```python
import torch
import torch.nn.functional as F

def grpo_clip_loss(new_log_probs, old_log_probs, advantages, epsilon=0.2):
    """
    计算GRPO-Clip损失
    """
    # 计算重要性采样比
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 根据优势符号计算裁剪边界
    upper_bound = torch.where(advantages >= 0, 1 + epsilon, float('inf'))
    lower_bound = torch.where(advantages < 0, 1 - epsilon, float('-inf'))
    
    # 裁剪的重要性采样比
    clipped_ratio = torch.clamp(ratio, lower_bound, upper_bound)
    
    # 计算原始目标和裁剪目标
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    
    # 取最小值（因为是最小化损失，但这里是最大化目标，所以加负号）
    loss = -torch.min(surrogate1, surrogate2)
    
    return loss.mean()

# 使用示例
def compute_grpo_clip_update(batch_data, policy_network, epsilon=0.2):
    """
    计算GRPO-Clip更新
    """
    prompts, old_outputs, old_log_probs, advantages = batch_data
    
    # 计算新策略的对数概率
    new_log_probs = policy_network.get_log_probs(prompts, old_outputs)
    
    # 计算GRPO-Clip损失
    loss = grpo_clip_loss(
        new_log_probs.flatten(),
        old_log_probs.flatten(),
        advantages.flatten(),
        epsilon
    )
    
    return loss
```

### 🎪 GRPO-Clip的优势

#### 与传统PPO的对比
```python
comparison_with_ppo = {
    "相似点": [
        "都使用裁剪机制限制策略更新幅度",
        "都基于重要性采样的离线策略学习",
        "目标函数结构类似（min操作）"
    ],
    "GRPO-Clip的特色": [
        "使用组归一化优势而非价值函数估计的优势",
        "每个输出的所有token共享同一个优势值",
        "特别为语言模型多输出采样场景设计"
    ]
}
```

#### 在语言模型训练中的价值
```python
value_in_lm_training = {
    "稳定性": "防止策略在单次更新中剧烈变化，避免文本质量崩溃",
    "效率": "允许在单批数据上多次更新，提高数据利用率",
    "适应性": "适用于各种文本生成任务和奖励函数设计",
    "简单性": "无需训练单独的价值函数网络"
}
```

### 📊 超参数ε的选择策略

```python
epsilon_selection_strategies = {
    "小值(0.1-0.2)": "保守更新，训练稳定但收敛慢，适合复杂任务",
    "中等值(0.2-0.3)": "平衡稳定性和收敛速度，通用选择",
    "大值(>0.3)": "激进更新，收敛快但不稳定，适合简单任务",
    "自适应调整": "训练初期用较大ε探索，后期减小ε精细调优"
}
```

## 7.2 Implementation

### 📊 内容概况

本页是**GRPO（Group Reward Proximal Policy Optimization）算法实现**的第一部分，重点讲解了**优势计算（组归一化奖励）**的具体实现。内容从GRPO训练循环的高层理解过渡到具体实现，特别讨论了两种计算组归一化奖励的方法，并引用了最新的研究成果。

### 🎯 要点总结

#### 1. **实现背景与连续性**
- 在理解了GRPO训练循环和目标的**高层概念**后，开始实现具体组件
- 许多组件在**SFT（监督微调）和EI（专家迭代）部分**已经实现，可以复用
- 体现了算法实现的**模块化设计和代码复用**思想

#### 2. **优势计算的核心任务**
- 实现计算**回放批次中每个示例的优势**（即组归一化奖励）的逻辑
- 这是GRPO训练循环的**第一个关键步骤**

#### 3. **两种组归一化方法对比**

##### 方法A：原始方法（Eq. 28）
```math
A^{(i)} = \frac{r^{(i)} - \text{mean}(r^{(1)}, r^{(2)}, \ldots, r^{(G)})}{\text{std}(r^{(1)}, r^{(2)}, \ldots, r^{(G)}) + \text{advantage\_eps}}
```
- 通过**标准差归一化**处理
- 问题：可能会**奖励批次内答案正确性变化较小的问题**，这可能不是理想的

##### 方法B：简化方法（Liu et al., 2025提出，Eq. 31）
```math
A^{(i)} = r^{(i)} - \text{mean}(r^{(1)}, r^{(2)}, \ldots, r^{(G)})
```
- **移除了归一化步骤**，只减去均值
- 避免了对低变化问题的偏好
- 计算更简单，减少了**除以接近零的标准差**可能带来的数值不稳定问题

#### 4. **文献引用与学术基础**
- 引用了**Liu等人2025年的工作**（Dr. GRPO），这是该领域的最新研究
- 绿色方框高亮显示了重要的**参考文献作者**
- 红色方框高亮了**公式编号(31)**，强调了这是本节的核心公式

### 🔬 技术细节深入

#### 原始方法的问题分析
```python
def analyze_std_normalization_issue():
    """分析标准差归一化可能存在的问题"""
    
    issues = [
        "数值不稳定：当组内奖励差异很小时，std接近零，可能导致除以零或极大值",
        "偏好偏差：会奖励那些组内所有答案都表现一致的问题（无论好坏）",
        "尺度敏感：对奖励的绝对尺度敏感，可能需要额外的奖励缩放"
    ]
    
    return issues
```

#### 简化方法的优势
```python
def simplified_method_advantages():
    """简化方法的优势分析"""
    
    advantages = [
        "数值稳定：避免了除以接近零的标准差",
        "计算高效：减少了一次标准差计算",
        "直观解释：优势就是相对于平均表现的偏离",
        "无偏偏好：不对组内变化大小施加偏好"
    ]
    
    return advantages
```

### 🔧 实现建议

#### 1. **两种方法的实现示例**
```python
import torch
import numpy as np

def compute_advantages_rewards(rewards, method="simplified", eps=1e-8):
    """
    计算组归一化优势
    
    Args:
        rewards: 形状为(G,)的张量，表示组内G个样本的奖励
        method: "original" 或 "simplified"
        eps: 防止除零的小常数
    
    Returns:
        优势值，形状同rewards
    """
    mean_reward = torch.mean(rewards)
    
    if method == "original":
        # 原始方法：除以标准差
        std_reward = torch.std(rewards)
        advantages = (rewards - mean_reward) / (std_reward + eps)
    elif method == "simplified":
        # 简化方法：只减去均值
        advantages = rewards - mean_reward
    else:
        raise ValueError(f"未知的方法: {method}")
    
    return advantages

# 使用示例
rewards = torch.tensor([0.9, 0.8, 0.85, 0.87, 0.82])
advantages_original = compute_advantages_rewards(rewards, method="original")
advantages_simplified = compute_advantages_rewards(rewards, method="simplified")

print(f"原始方法优势: {advantages_original}")
print(f"简化方法优势: {advantages_simplified}")
```

#### 2. **在训练循环中的集成**
```python
def grpo_training_step(self, batch, use_simplified=True):
    """GRPO训练步骤，包含优势计算"""
    # 采样多个输出
    outputs = self.sample_multiple_outputs(batch, num_samples=self.G)
    
    # 计算奖励
    rewards = self.compute_rewards(batch, outputs)
    
    # 计算优势
    batch_size = len(batch['prompt'])
    advantages = []
    
    for i in range(batch_size):
        group_rewards = rewards[i]  # 形状: (G,)
        
        if use_simplified:
            # 使用简化方法
            mean_reward = torch.mean(group_rewards)
            group_advantages = group_rewards - mean_reward
        else:
            # 使用原始方法
            mean_reward = torch.mean(group_rewards)
            std_reward = torch.std(group_rewards)
            group_advantages = (group_rewards - mean_reward) / (std_reward + 1e-8)
        
        advantages.append(group_advantages)
    
    advantages = torch.stack(advantages)
    
    # 后续使用这些优势计算GRPO-Clip损失
    loss = self.compute_grpo_clip_loss(outputs, advantages)
    
    return loss
```

### Problem (compute_group_normalized_rewards): Group normalization (2 points)

Deliverable: Implement a method compute_group_normalized_rewards that calculates raw
rewards for each rollout response, normalizes them within their groups, and returns both the
normalized and raw rewards along with any metadata you think is useful.
- 完成

### Problem (compute_naive_policy_gradient_loss): Naive policy gradient (1 point)

Deliverable: Implement a method compute_naive_policy_gradient_loss that computes the
per-token policy-gradient loss using raw rewards or pre-computed advantages.
- 完成

### Problem (compute_grpo_clip_loss): GRPO-Clip loss (2 points)
Deliverable: Implement a method compute_grpo_clip_loss that computes the per-token
GRPO-Clip loss.
- 完成

### Problem (compute_policy_gradient_loss): Policy-gradient wrapper (1 point)
Deliverable: Implement compute_policy_gradient_loss, a convenience wrapper that dispatches
to the correct loss routine (no_baseline, reinforce_with_baseline, or grpo_clip) and returns
both the per-token loss and any auxiliary statistics.
- 完成

### Problem (masked_mean): Masked mean (1 point)
Deliverable: Implement a method masked_mean that averages tensor elements while respecting a
boolean mask.
- 完成

### Problem (grpo_microbatch_train_step): Microbatch train step (3 points)
Deliverable: Implement a single micro-batch update for GRPO, including policy-gradient loss,
averaging with a mask, and gradient scaling.
- 完成

### Problem (grpo_train_loop): GRPO train loop (5 points)
Deliverable: Implement a complete train loop for GRPO. Begin training a policy on MATH and
confirm that you see validation rewards improving, along with sensible rollouts over time. Provide a
plot with the validation rewards with respect to steps, and a few example rollouts over time.

- 采用gsm8k体验流程
```python
Eval n_grpo_idx: 149 correct_num: 1069 error_num: 250
```