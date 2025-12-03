本文主要整理CS336 Lecture 16 RL章节的主要内容。

## 2.0 Why do we need yet another RL algorithm..?

### 要点总结

1. 为什么不使用 PPO？
PPO（特别是用于大语言模型微调时）存在两大主要挑战：
*   **实现复杂**：PPO在实际中的应用非常复杂，涉及到策略模型、价值模型、奖励模型等多个组件的协同训练，工程实现难度大，容易出错。
*   **资源消耗与调优难度高**：
    *   **内存消耗大**：价值模型需要额外的显存。
    *   **训练不稳定**：需要对价值模型进行额外的训练和精细的超参数调优，整个过程不够稳定和直接。

2. 为什么不使用 DPO？
DPO虽然比PPO更简单，但也有其固有的限制：
*   **数据要求高**：DPO依赖于**人工标注的偏好对比数据**（即对于同一个提示，需要有成对的“好回答”和“坏回答”）。这类数据并非天然存在，获取成本高昂。
*   **离线算法的局限性**：DPO本质上是一种**离线**强化学习算法。它无法利用训练过程中模型自己新生成的数据进行学习，学习效率受限。虽然可以通过迭代的方式（迭代DPO）模拟在线学习，但这并非其原生设计，增加了复杂性。

### 核心结论

综合来看，这张幻灯片论证了PPO和DPO各自存在明显的痛点：PPO过于**笨重和复杂**，而DPO受限于**数据形式和离线学习模式**。

## 2.1 New kid on the block: GRPO

### 内容理解

GRPO是一种旨在**降低强化学习训练成本**的新算法。它建立在PPO的框架之上，但进行了一项关键简化：**去除了需要单独训练的价值模型**。

#### 核心思想

1.  **从PPO出发**：GRPO保留了PPO的核心组件，特别是其**Clipped Surrogate Objective**，这确保了策略更新的稳定性。
2.  **关键创新：组相对优势估计**：GRPO最核心的改动是**不再使用一个神经网络（价值模型）来估计优势函数**。取而代之的是一种基于统计的方法：
    -   对于每个提示（q），从当前策略中采样生成**一组（G个）回答**。
    -   为这组回答中的每一个计算一个奖励分数（r_i），这个奖励可以来自一个奖励模型，也可以是其他形式的反馈。
    -   对于组内的每个回答，其**优势（A_i）** 被计算为该回答的奖励相对于整个组的奖励的**Z-score**（即减去组内平均奖励，再除以组内奖励的标准差）。
3.  **优势**：
    -   **大幅简化**：避免了训练一个与策略模型同等规模的价值模型，节省了巨大的计算成本和内存开销。
    -   **在线学习**：如图所示，在在线学习设置下（即生成数据后立即更新），GRPO本质上就变成了使用**组归一化奖励**的策略梯度方法。

#### 与PPO的对比

-   **PPO**：优势 `A` 依赖于一个需要被训练的价值模型 `Vφ(s)` 的预测（例如，通过GAE计算）。
-   **GRPO**：优势 `A_i` 直接从一个回答组内的奖励统计量中计算得出，无需训练任何价值模型。

---

### 打印公式

#### 公式 (1): GRPO 目标函数
这是GRPO要最大化的总体目标函数。它结合了裁剪后的策略梯度目标和KL散度惩罚。

$$ \mathcal{J}_{GRPO}(\theta)=\mathbb{E}_{q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{old}(O|q)}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_{\theta}(o_i|q)}{\pi_{old}(o_i|q)}A_{i},\ \text{clip}\left(\frac{\pi_{\theta}(o_i|q)}{\pi_{old}(o_i|q)},1-\epsilon,1+\epsilon\right)A_{i}\right)\right) - \beta D_{KL}\left(\pi_{\theta} \| \pi_{ref}\right)\right] $$

**参数解释：**
-   $\theta$: 待优化的策略模型参数。
-   $q$: 输入提示。
-   $o_i$: 对提示 $q$ 生成的第 $i$ 个回答。
-   $\pi_{old}$: 生成回答组时使用的旧策略（行为策略）。
-   $\pi_{\theta}$: 当前正在优化的新策略。
-   $\pi_{ref}$: 参考策略（例如SFT模型），用于防止模型偏离太远。
-   $A_i$: 第 $i$ 个回答的优势，由公式(3)定义。
-   $\epsilon$: PPO裁剪范围超参数。
-   $\beta$: KL散度惩罚项的权重超参数。
-   $D_{KL}$: KL散度，衡量新策略与参考策略的差异。

#### 公式 (2): KL散度的计算
这个公式给出了KL散度 $D_{KL}(\pi_{\theta} \| \pi_{ref})$ 的一种具体计算方式（基于对数概率）。

$$ D_{KL}\left(\pi_{\theta} \| \pi_{ref}\right) = \frac{\pi_{ref}(o|q)}{\pi_{\theta}(o|q)} - \log\frac{\pi_{ref}(o|q)}{\pi_{\theta}(o|q)} - 1 $$

#### 公式 (3): GRPO 优势计算
这是GRPO算法的灵魂，定义了如何在不使用价值模型的情况下计算优势。

$$ A_{i} = \frac{r_{i} - \text{mean}\left(\{r_{1}, r_{2}, \cdots, r_{G}\}\right)}{\text{std}\left(\{r_{1}, r_{2}, \cdots, r_{G}\}\right)} $$

**参数解释：**
-   $r_i$: 第 $i$ 个回答获得的原始奖励。
-   $\text{mean}(\{r_{1}, \cdots, r_{G}\})$: 同一提示下生成的整个回答组的平均奖励。
-   $\text{std}(\{r_{1}, \cdots, r_{G}\})$: 同一提示下生成的整个回答组的奖励标准差。
-   **本质**：$A_i$ 就是一个**Z-score**，它衡量了某个回答的奖励在它所在的组中是“高于平均水平”还是“低于平均水平”，以及超出平均水平的程度。一个正的优势（A_i > 0）意味着这个回答比组内其他回答更好。

#### （参考）PPO 策略损失核心
幻灯片底部提供了PPO的公式作为对比参考。

$$ \min\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{i}}(a|s)}A^{\pi_{\theta_{i}}}(s,a),\ \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{i}}(a|s)},1-\epsilon,1+\epsilon\right)A^{\pi_{\theta_{i}}}(s,a)\right) $$

**关键区别**：PPO中的优势 $A^{\pi_{\theta_{i}}}(s,a)$ 通常通过价值模型和广义优势估计（GAE）来计算，而GRPO中的 $A_i$ 则由公式(3)直接计算。

## 2.2 GRPO is very simple (thanks to lack of value function..)

### 内容理解

这张幻灯片的核心论点是：**“GRPO非常简单（这要归功于它没有价值函数…）”**，并通过展示一个可以编写的小型实现来证明这一点。

幻灯片分为三个部分：

1.  **文字描述**：强调了GRPO的简洁性，并概述了其实现的关键步骤。
2.  **伪代码/代码草图**：展示了一个名为 `compute_grpo_loss` 的函数框架，但图中的文本似乎有大量重复和错乱，可能是渲染或转录问题。
3.  **可运行的代码示例**：幻灯片底部指向了一个真实的GitHub仓库链接，表明存在一个可参考的简洁实现。

**GRPO实现的核心步骤（根据幻灯片左侧文字描述）：**

1.  **为每个轨迹计算奖励**：使用奖励模型为当前策略生成的每个回答打分。
2.  **对每组回答进行均值/方差归一化**：这就是GRPO的核心——在同一个提示生成的一组回答内部，计算每个回答奖励的Z-score作为其优势估计。
3.  **计算KL散度项**：计算当前策略与参考策略之间的KL散度作为正则化惩罚。
4.  **对损失进行梯度更新**：将策略梯度损失（使用归一化后的奖励）和KL损失结合起来，进行反向传播。

---

### 打印代码（整理和注释后的实现）

```python
import torch
import torch.nn.functional as F

def compute_grpo_loss(
    model,                    # 当前需要优化的策略模型
    reference_model,          # 参考模型（如SFT模型），用于计算KL散度
    prompts,                  # 输入的提示序列
    generated_sequences,      # 模型为每个提示生成的G个回答 [batch_size, G, seq_len]
    rewards,                  # 奖励模型为每个回答打出的分数 [batch_size, G]
    kl_coef=0.1,             # KL散度惩罚项的系数
    clip_epsilon=0.2          # PPO风格的裁剪范围，用于稳定训练
):
    """
    计算GRPO（组相对策略优化）损失。
    
    参数:
        model: 当前策略模型（torch.nn.Module）
        reference_model: 参考模型，参数被冻结（torch.nn.Module）
        prompts: 输入提示的token IDs [batch_size, prompt_len]
        generated_sequences: 生成的回答的token IDs [batch_size, G, response_len]
        rewards: 每个回答的奖励分数 [batch_size, G]
        kl_coef: KL散度惩罚的权重
        clip_epsilon: 概率比裁剪参数
        
    返回:
        total_loss: 总的GRPO损失（策略损失 + KL惩罚）
        stats: 包含各项损失的字典，用于日志记录
    """
    
    batch_size, group_size, seq_len = generated_sequences.shape
    # 重塑张量以便处理: [batch_size, G, seq_len] -> [batch_size * G, seq_len]
    generated_sequences_flat = generated_sequences.view(batch_size * group_size, seq_len)
    
    # 1. 获取当前策略模型和参考模型的对数概率
    with torch.no_grad():
        # 参考模型的对数概率（旧策略）
        ref_logprobs = compute_token_log_probs(reference_model, prompts, generated_sequences_flat)
        ref_logprobs = ref_logprobs.view(batch_size, group_size, -1)  # 恢复组维度
    
    # 当前策略模型的对数概率（新策略）
    current_logprobs = compute_token_log_probs(model, prompts, generated_sequences_flat)
    current_logprobs = current_logprobs.view(batch_size, group_size, -1)  # 恢复组维度
    
    # 2. 计算概率比 (probability ratio) r(θ)
    log_ratio = current_logprobs - ref_logprobs  # 对数空间计算更稳定
    ratio = torch.exp(log_ratio)  # r(θ) = π_θ(a|s) / π_ref(a|s)
    
    # 3. GRPO核心: 计算组内归一化的优势（Z-score）
    advantages = []
    for i in range(batch_size):
        group_rewards = rewards[i]  # 当前提示对应的G个奖励 [G]
        # 计算组内Z-score: (单个奖励 - 组平均奖励) / 组标准差
        mean_reward = group_rewards.mean()
        std_reward = group_rewards.std() + 1e-8  # 避免除零
        group_advantages = (group_rewards - mean_reward) / std_reward
        advantages.append(group_advantages)
    
    advantages = torch.stack(advantages)  # [batch_size, G]
    advantages = advantages.unsqueeze(-1)  # 扩展维度以匹配序列长度 [batch_size, G, 1]
    
    # 4. 计算PPO风格的裁剪策略损失
    # 未裁剪的损失
    pg_loss_unclipped = ratio * advantages
    # 裁剪的损失
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    pg_loss_clipped = clipped_ratio * advantages
    # 取两者中较小的（悲观更新）
    pg_loss = -torch.min(pg_loss_unclipped, pg_loss_clipped).mean()
    
    # 5. 计算KL散度惩罚（作为正则化项）
    # KL散度近似为： (π_θ - π_ref) 或更精确的计算
    kl_penalty = kl_coef * (current_logprobs - ref_logprobs).mean()
    
    # 6. 组合总损失
    total_loss = pg_loss + kl_penalty
    
    # 收集统计信息
    stats = {
        "loss/total": total_loss.item(),
        "loss/policy": pg_loss.item(),
        "loss/kl_penalty": kl_penalty.item(),
        "policy/ratio": ratio.mean().item(),
        "advantages/mean": advantages.mean().item(),
        "advantages/std": advantages.std().item(),
    }
    
    return total_loss, stats


def compute_token_log_probs(model, prompts, responses):
    """
    辅助函数：计算给定模型生成特定序列的对数概率。
    
    参数:
        model: 语言模型
        prompts: 提示序列 [batch_size, prompt_len]
        responses: 响应序列 [batch_size, response_len]
        
    返回:
        log_probs: 每个序列中每个token的对数概率 [batch_size, response_len]
    """
    # 将提示和响应拼接
    input_ids = torch.cat([prompts, responses], dim=1)
    # 获取模型的输出logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    
    # 计算对数概率
    log_probs = F.log_softmax(logits, dim=-1)
    # 只取响应部分，并获取对应token的概率
    response_log_probs = log_probs[:, -responses.size(1)-1:-1]  # 调整索引
    response_log_probs = response_log_probs.gather(-1, responses.unsqueeze(-1)).squeeze(-1)
    
    return response_log_probs
```

### 关键实现要点解析

1.  **组内归一化（核心创新）**：
    ```python
    group_advantages = (group_rewards - mean_reward) / std_reward
    ```
    这是GRPO与PPO最根本的区别。它不需要训练价值模型，而是通过**组内统计**来获得相对优势估计。

2.  **PPO裁剪机制（保持稳定）**：
    ```python
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    pg_loss = -torch.min(pg_loss_unclipped, pg_loss_clipped).mean()
    ```
    GRPO继承了PPO的裁剪机制，确保策略更新不会过于激进，保持了训练的稳定性。

3.  **KL散度惩罚（防止偏离）**：
    ```python
    kl_penalty = kl_coef * (current_logprobs - ref_logprobs).mean()
    ```
    这是一个重要的正则化项，防止当前策略过度偏离性能良好的参考模型（如SFT模型），避免模型“遗忘”已有的语言能力。

4.  **对数概率计算**：
    使用 `gather` 函数高效地计算生成序列中每个token在被选中时的对数概率，这是策略梯度计算的基础。

## 2.3 Thinking carefully about the GRPO objective..

### 内容理解

#### 核心问题：什么是“有效的基线”？

幻灯片开篇立论：**## GRPO不使用一个“有效的”基线**。

这里的“有效”或“合法”有严格的数学定义，源于我们之前讨论的**策略梯度定理**。该定理指出，一个基线 $ b(s) $ 要能保持梯度估计的**无偏性**，必须**只依赖于状态 $ s $**，而不能依赖于当前选择的动作 $ a $。

-   **原始GRPO的问题**：其优势估计公式 $ A_i = \frac{r_i - \mu}{\sigma} $ 中，除以标准差 $ \sigma $ 的操作**破坏了无偏性**。因为标准差 $ \sigma $ 是根据整个组（包含所有动作）的奖励计算出来的，它隐含地依赖于动作的分布，因此不是一个“合法”的基线。尽管除以标准差可以降低方差，但它可能在理论上引入偏差。

#### 解决方案：无偏梯度版本的GRPO

幻灯片随后提出了一个问题并给出了答案：**什么是GRPO的无偏梯度版本？**

这个版本被称为 **“正确的GRPO”** 或 **Dr. GRPO**。其修改非常简单直接：

-   **核心修改**：在优势估计中，**只减去均值（作为基线），而不再除以标准差**。
-   即，将优势估计从 $ \frac{r_i - \mu}{\sigma} $ 改为 $ r_i - \mu $。

这样，均值 $ \mu $ 就是一个完全合法的基线，因为它只依赖于状态（即提示），从而保证了梯度估计的无偏性。

#### 其他修改：长度归一化项

幻灯片还提到对公式左侧的**长度归一化项**进行了修改（图中未完全展示），这通常是为了解决模型倾向于生成长度不一的文本的问题，使训练更稳定。

#### 效果验证：令牌效率

右下角的图表从实证角度支持了理论修正。它比较了原始GRPO和Dr. GRPO的**“令牌效率”**，即模型消耗的计算资源（令牌数量）与获得的奖励之间的关系。

-   **结论**：Dr. GRPO（无偏版本）在相同的令牌消耗下，能获得比原始GRPO更高的奖励，**效率更优**。这证明移除标准差项不仅在理论上更正确，在实践中也可能带来更好的性能。

---

### 打印公式

#### 公式 1: 原始 GRPO 的优势估计（有偏）

$$
\tilde{A}_{ij} = \frac{Q_{ij} - \frac{1}{N} \sum_{j=1}^N Q_{ij}}{\sqrt{\frac{1}{N} \sum_{j=1}^N (Q_{ij} - \frac{1}{N} \sum_{j=1}^N Q_{ij})^2}}
$$

**公式解读：**
-   $ Q_{ij} $：在状态（提示）$ i $ 下，采取动作（生成回答）$ j $ 所获得的奖励。
-   $ \frac{1}{N} \sum_{j=1}^N Q_{ij} $：在状态 $ i $ 下，所有 $ N $ 个动作的奖励均值 $ \mu_i $。
-   分母是奖励的**样本标准差** $ \sigma_i $。
-   $ \tilde{A}_{ij} $ 就是奖励的 **Z-score**。

**理论缺陷：** 除以标准差 $ \sigma_i $ 的操作使得整个项不再是一个合法的基线，因为它依赖于当前批次中所有动作的奖励分布，可能引入偏差。

#### 公式 2: 修正版 GRPO - Dr. GRPO（无偏）

这是幻灯片提出的修正版本，确保了梯度估计的无偏性。

$$
\tilde{A}_{ij} = Q_{ij} - \frac{1}{N} \sum_{j=1}^N Q_{ij}
$$

**公式解读：**
-   移除了原始公式中的分母（标准差项）。
-   优势估计简化为：**个体奖励与组内平均奖励的差值**。
-   均值 $ \frac{1}{N} \sum_{j=1}^N Q_{ij} $ 作为一个**只依赖于状态 $ i $** 的基线，完全符合策略梯度定理的要求，保证了无偏性。

### 总结

这张幻灯片完成了一次从理论到实践的严谨推导：
1.  **发现问题**：从理论角度指出原始GRPO优势估计的数学瑕疵（使用无效基线）。
2.  **提出方案**：给出一个理论上更严谨的修正版本（Dr. GRPO），确保梯度无偏。
3.  **验证效果**：通过实验证明修正版本具有更高的“令牌效率”，实践性能更优。

## 2.4 Length biases of GRPO

这张幻灯片深入分析了GRPO算法中一个重要的非预期偏差问题——**长度偏差**，并展示了其修正方法的效果。

### 内容概况

幻灯片主要分为两个部分：
1.  **问题分析**：解释了GRPO算法中的两项操作（标准差归一化和响应长度归一化）各自的作用，并重点揭示了长度归一化项如何引入有害的“响应级长度偏差”。
2.  **效果验证**：通过五组并列的实验曲线图，对比了原始GRPO与修正后的算法在关键指标上的表现，直观展示了修正方法的有效性。

---

### 要点总结

#### 第一部分：GRPO长度偏差的成因与机制

1.  **两项操作的作用**：
    *   **`Stdev`（标准差归一化）**：主要影响**问题级别**的权重。它会放大对“过于简单”或“过于困难”的问题的关注。
    *   **长度归一化**：主要导致**响应级别**的长度偏差。这是本幻灯片的分析重点。

2.  **响应级长度偏差的核心机制**：
    当在损失函数中除以响应长度 $ |\mathbf{o}_t| $ 进行归一化时，会产生系统性偏差：
    *   **对于正确回答（优势值 $ \hat{A}_{i,t} > 0 $）**：梯度更新大小与 $ 1 / |\mathbf{o}_i| $ 成正比。因此，**更短的正确回答会获得更大的梯度更新**，导致模型在回答正确时倾向于生成简短内容。
    *   **对于错误回答（优势值 $ \hat{A}_{i,t} < 0 $）**：惩罚力度与 $ 1 / |\mathbf{o}_i| $ 成正比。因此，**更长的错误回答会因为其较大的分母而受到的惩罚更小**，导致模型在回答错误时反而倾向于生成长篇大论以减轻惩罚。

#### 第二部分：修正方法的效果验证

通过对比原始GRPO（灰色线）与修正后的Dr. GRPO（红色线）的学习曲线，可以得出以下结论：

1.  **奖励与性能**：
    *   **奖励**：修正后的算法获得了与原始方法相当甚至更高的奖励。
    *   **平均基准分数**：最关键的是，Dr. GRPO取得了**显著更高的基准测试分数**，证明修正偏差最终提升了模型的实际推理能力。

2.  **长度控制**：
    *   **总输出长度**：修正后的方法有效控制了输出长度。
    *   **正确/错误回答的长度**：图表清晰显示，Dr. GRPO**显著缩短了错误回答的长度**，同时保持了正确回答的适当长度。这直接证明修正方法成功遏制了模型在出错时“用长度弥补质量”的不良倾向。

### 核心结论

这张幻灯片阐明，GRPO中的长度归一化操作引入了一种非预期的、有害的优化信号，扭曲了模型的行为。对其进行修正（即**移除导致偏差的项**）后，不仅纠正了模型的长度偏好，更重要的是**提升了其真实性能**，这证明了算法理论严谨性的重要性。