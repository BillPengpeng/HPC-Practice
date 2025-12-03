本文主要整理CS336 Lecture 17 RL章节的主要内容。

## 1. rl_setup_for_language_models

### 要点总结

以下是图片中详细说明的强化学习应用于语言模型的核心要素：

1.  **状态 (State) `s`**
    *   **定义**：当前的状态由两部分组成：**初始提示 (prompt)** 加上**目前已生成的所有响应内容**。
    *   **解读**：随着模型不断生成新的词元（token），状态 `s` 也在不断演变和增长。

2.  **动作 (Action) `a`**
    *   **定义**：每一步的**动作**就是选择并生成**下一个词元 (token)**。
    *   **解读**：这相当于语言模型在每一步的预测任务。

3.  **奖励 (Rewards) `R`**
    *   **核心**：奖励用于衡量最终生成的**响应质量**。
    *   **侧重点**：
        *   **结果奖励**：依赖于整个完整响应的最终结果（例如，回答是否正确，内容是否优质）。
        *   **可验证的奖励**：其计算过程是确定性的（例如，生成的答案是否与标准答案匹配）。
    *   **特别说明**：在语言模型场景下，**折扣（discounting）** 和**自举（bootstrapping）** 这些传统强化学习概念的重要性相对较低。

4.  **转移概率 (Transition probabilities) `T(s'|s,a)`**
    *   **关键特性**：**具有确定性**。下一个状态 `s'` 完全由当前状态 `s` 和采取的动作 `a` 决定，即 `s' = s + a`。
    *   **优势**：这种确定性使得在生成响应时可以进行**规划**或**测试时计算**（例如，通过搜索或采样来寻找更优的序列），这与机器人学中环境的不确定性形成鲜明对比。
    *   **灵活性**：由于状态（文本）是人为构建的，操作上具有很大的灵活性。

5.  **策略 (Policy) `π(a|s)`**
    *   **定义**：策略就是一个**（经过微调的）语言模型本身**。它根据当前状态 `s`（即已有的文本），来决定下一个动作 `a`（即下一个词元）的概率分布。

6.  **回合/轨迹 (Rollout/Episode/Trajectory)**
    *   **过程**：描述了从初始状态到获得奖励的完整过程：`s → a → ... → a → a → R`。

7.  **目标 (Objective)**
    *   **最终目标**：**最大化期望奖励 `E[R]`**。
    *   **期望来源**：这里的期望是对**初始提示的分布**和**模型生成的所有词元**取平均。

总而言之，这张图片清晰地勾勒出了将语言模型视为强化学习智能体的数学模型，强调了其**状态和转移的确定性**，以及**奖励基于最终响应结果**的核心特点。

## 2. policy_gradient

### 内容概况

这两张图片系统地阐述了**策略梯度方法**，这是将强化学习应用于语言模型（如ChatGPT）的核心训练算法。内容从最基础的目标函数出发，逐步深入分析了该方法的直接实现（朴素策略梯度）所面临的**高方差挑战**，并引出了用于解决此问题的关键技术——**基线函数** 和**优势函数**。

*   **第一张图** 重点介绍了策略梯度的基本公式推导、直观的“朴素”实现方法及其存在的问题（高方差、稀疏奖励）。
*   **第二张图** 重点讲解了如何使用基线函数来降低方差，并进一步将其与强化学习中的核心概念“优势函数”联系起来，为后续更高级的优化方法做了铺垫。

---

### 要点总结

#### 1. 核心目标与公式推导
*   **目标**：最大化期望奖励 $ E[R] $ 对策略 $ π $ 的梯度。
*   **推导结果**：策略梯度的核心公式为 $ ∇ E[R] = E[ ∇ \log π(a|s) \cdot R(s, a) ] $。
    *   这意味着我们可以通过采样（一个提示 `s`，一个根据当前策略生成的响应 `a`）来估计梯度。
    *   **直观理解**：参数的更新方向是增加获得高奖励 `R(s,a)` 的动作 `a` 的概率，更新幅度与奖励大小成正比。如果奖励为0，则不做更新。

#### 2. 朴素策略梯度的挑战
*   **高方差/噪声**：由于奖励信号可能非常稀疏（例如，只有“正确”或“错误”两种奖励），梯度估计的随机性很大，导致训练不稳定、收敛慢。
*   **示例说明**：图片指出，在稀疏奖励设置下，朴素方法只对获得奖励（如奖励为1）的响应进行学习，大部分响应（奖励为0）被忽略。

#### 3. 解决方案：基线函数
*   **核心思想**：引入一个仅依赖于状态 `s` 的基线函数 $ b(s) $，将更新依据从 $ R(s, a) $ 改为 $ R(s, a) - b(s) $。
*   **数学性质**：减去基线不改变梯度估计的**无偏性**（期望值不变），但可以显著**降低方差**，从而加速训练。
*   **直观理解**：我们不再追求“高绝对奖励”，而是追求“高于平均水平的相对奖励”。如果动作 `a` 的奖励高于在状态 `s` 下的预期水平（基线），则增加其概率。

#### 4. 基线函数的选择
*   **最优基线**：理论上存在一个可以最小化方差的最优基线 $ b^*(s) $，但其表达式复杂，难以计算。
*   **实用启发式方法**：一个常用且有效的选择是使用**状态依赖的平均奖励**作为基线，即 $ b(s) = E[R|s] $（状态价值函数 $ V(s) $）。这仍然需要估计，但是一个可行的目标。

#### 5. 与优势函数的联系
*   **优势函数**：定义为 $ A(s, a) = Q(s, a) - V(s) $，衡量了在状态 `s` 下采取动作 `a` 相对于平均水平的“优势”有多大。
*   **关键结论**：当选择 $ b(s) = V(s) $ 时，基线化的奖励 $ R(s, a) - b(s) $ 恰好就等于优势函数 $ A(s, a) $。
*   **最终形式**：策略梯度可以统一表示为 $ E[ ∇ \log π(a|s) \cdot δ ] $，其中 $ δ $ 是一个“信号估计器”。使用优势函数 $ A(s, a) $ 作为 $ δ $ 是一种非常重要且有效的选择，为后续学习更先进的策略梯度算法（如PPO）奠定了基础。

总而言之，这两张图片清晰地勾勒出了从最基础的策略梯度公式，到认识其缺陷，再到引入基线/优势函数进行优化的完整逻辑链条，是理解现代基于强化学习的语言模型训练算法的关键。

## 3.0 compute_kl_penalty

```python
def compute_kl_penalty(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute an estimate of KL(model | ref_model), where the models are given by:
        log_probs [batch trial pos vocab]
        ref_log_probs [batch trial pos vocab]
    Use the estimate:
        KL(p || q) = E_p[q/p - log(q/p) - 1]
    """
    return (torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1).sum(dim=-1).mean()
```

从标准KL散度定义出发：
$$
KL(p || q) = E_p\left[\log\frac{p}{q}\right]
$$
现在，考虑您给出的表达式：
$$
E_p\left[\frac{q}{p} - \log\frac{q}{p} - 1\right]
$$
我们可以逐步简化：
$$
E_p\left[\frac{q}{p} - \log\frac{q}{p} - 1\right] = E_p\left[\frac{q}{p}\right] - E_p\left[\log\frac{q}{p}\right] - E_p[1]
$$
其中：
- $E_p[1] = 1$，因为常数的期望就是它本身。
- $E_p\left[\frac{q}{p}\right] = \int p \cdot \frac{q}{p} \, dx = \int q \, dx = 1$，因为q是概率分布，积分为1。
- $E_p\left[\log\frac{q}{p}\right] = E_p[\log q - \log p] = -E_p[\log p - \log q] = -KL(p || q)$。

代入上述结果：
$$
E_p\left[\frac{q}{p} - \log\frac{q}{p} - 1\right] = 1 - (-KL(p || q)) - 1 = KL(p || q)
$$
因此，公式得证。这个表达式是KL散度的另一种数学表示。

## 3.1 compute_deltas

```python
def compute_deltas(rewards: torch.Tensor, mode: str) -> torch.Tensor:  # @inspect rewards
    """
    Args:
        rewards (float[batch trial])
    Returns:
        deltas (float[batch trial]) which are advantage-like quantities for updating
    """
    if mode == "rewards":
        return rewards

    # Dr. GRPO
    if mode == "centered_rewards":
        # Compute mean over all the responses (trial) for each prompt (batch)
        mean_rewards = rewards.mean(dim=-1, keepdim=True)  # @inspect mean_rewards
        centered_rewards = rewards - mean_rewards  # @inspect centered_rewards
        return centered_rewards

    # GRPO
    if mode == "normalized_rewards":
        mean_rewards = rewards.mean(dim=-1, keepdim=True)  # @inspect mean_rewards
        std_rewards = rewards.std(dim=-1, keepdim=True)  # @inspect std_rewards
        centered_rewards = rewards - mean_rewards  # @inspect centered_rewards
        normalized_rewards = centered_rewards / (std_rewards + 1e-5)  # @inspect normalized_rewards
        return normalized_rewards

    if mode == "max_rewards":
        # Zero out any reward that isn't the maximum for each batch
        max_rewards = rewards.max(dim=-1, keepdim=True)[0]
        max_rewards = torch.where(rewards == max_rewards, rewards, torch.zeros_like(rewards))
        return max_rewards

    raise ValueError(f"Unknown mode: {mode}")
```

## 3.2 compute_loss

```python
def compute_loss(log_probs: torch.Tensor, deltas: torch.Tensor, mode: str, old_log_probs: torch.Tensor | None = None) -> torch.Tensor:
    if mode == "naive":
        return -einsum(log_probs, deltas, "batch trial pos, batch trial -> batch trial pos").mean()

    if mode == "unclipped":
        ratios = log_probs / old_log_probs  # [batch trial]
        return -einsum(ratios, deltas, "batch trial pos, batch trial -> batch trial pos").mean()

    # PPO / GRPO
    if mode == "clipped":
        epsilon = 0.01
        unclipped_ratios = log_probs / old_log_probs  # [batch trial]
        unclipped = einsum(unclipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")

        clipped_ratios = torch.clamp(unclipped_ratios, min=1 - epsilon, max=1 + epsilon)
        clipped = einsum(clipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")
        return -torch.minimum(unclipped, clipped).mean()

    raise ValueError(f"Unknown mode: {mode}")
```
