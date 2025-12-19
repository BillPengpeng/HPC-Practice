本文主要整理Assignment 5 (alignment)的主要内容。

## 5 Expert Iteration for MATH

### 内容概括

**专家迭代（Expert Iteration, EI）** 是一种基于自我改进的迭代训练范式，专门设计用于数学推理任务的模型优化。该方法的核心思想是：

> 让模型从**自身生成的更高质量推理**中学习，实现持续的性能改进。

#### 背景上下文：
- 在之前的监督微调（SFT）实验中，通过**过滤掉低质量样本**可以提升模型性能
- 专家迭代将这一思路进一步扩展：**对模型自身生成的推理轨迹进行筛选**
- 通过迭代过程，模型能够从自己的"正确回答"中学习，逐渐变得更好

#### 在MATH数据集上的应用：
文档明确指出，该方法将专门应用于MATH数学推理数据集，目标是提升模型解决复杂数学问题的能力。

### 要点总结

1. **核心目标**
- 通过迭代训练，让模型从自己的**成功解答**中学习
- 逐步提升模型解决数学问题的**准确性和推理能力**

2. **理论基础**
- 基于Anthony等人（2017）提出的专家迭代框架
- 在语言模型领域已有多个研究验证（Cobbe 2021, Zelikman 2022等）
- 适用于需要**逐步推理**的任务，特别是数学问题求解

3. **关键特性**
- **自我改进**：模型从自己的输出中学习
- **质量过滤**：只保留正确的推理轨迹进行训练
- **迭代优化**：多轮次逐步提升模型能力

4. **技术实现要点**
- 使用vLLM进行高效采样
- 设置`min_tokens=4`防止生成空字符串
- 每个问题生成多个答案（G个），选择正确的进行训练

### Problem (expert_iteration_experiment): Run expert iteration on the MATH dataset (2 points) (6 H100 hrs)

Run expert iteration on the MATH dataset (provided at /data/a5-alignment/MATH/train.jsonl)
using the Qwen 2.5 Math 1.5B Base model, varying the number of rollouts G per question and the
number of epochs used in the SFT step, and using n_ei_steps = 5. Vary the batch size for each
expert iteration step (i.e., the size of Db ) in {512, 1024, 2048}. (You do not need to try all possible
combinations of these hyperparameters. Just enough to draw conclusions about each is fine.) Log the
entropy of the model’s reponses over training. Make sure to have vLLM terminate generations at the
second answer tag </answer>, as done in the SFT section.

## 6 Primer on Policy Gradients

### 内容概况
本段文字是**策略梯度算法**的入门介绍，核心是解释其**在语言模型强化学习中的应用**。它指出，**基于已验证的奖励**对强大的基础语言模型进行强化学习训练，已被证明是**提升模型推理能力**的有效方法，当前**领先的开源推理模型**正是采用了此技术路径。

### 要点总结
1.  **核心发现**：针对“强基础模型”执行“基于已验证奖励”的强化学习，可**显著提升**其在“推理能力”和“整体性能”方面的表现。
2.  **核心算法**：**策略梯度**是实现这一目标的关键强化学习算法，因其能优化任意奖励函数。
3.  **成功案例**：当前最强的开源推理模型 **DeepSeek R1** 和 **Kimi k1.5** 即是通过策略梯度算法训练得到的，这为该方法提供了强有力的实证支持。
4.  **参考来源**：本介绍主要基于两个被广泛认可的深度学习资源：
    *   **《Spinning Up in Deep RL》** (OpenAI, Achiam, 2018a)
    *   **《Reinforcement Learning from Human Feedback (RLHF) Book》** (Nathan Lambert, 2024)

**简而言之，本段确立了“策略梯度+已验证奖励”作为提升语言模型推理能力的有效范式，并援引前沿模型作为其成功的证据。**

## 6.1 Language Models as Policies

### 内容概况

这段文字阐述了**将因果语言模型视为强化学习中的策略**的核心概念。它从强化学习（RL）的视角重新解读了语言模型的生成过程，为后续使用策略梯度等RL方法优化语言模型奠定了理论基础。

### 要点总结

1. **核心类比**：
   - **状态（State）**：当前的文本前缀（$s_t$），即已生成的序列。
   - **动作（Action）**：下一个待生成的词元（$a_t$），即语言模型要预测的token。

2. **策略定义**：
   - 参数为 $θ$ 的因果语言模型，本质上定义了一个**分类随机策略** $π_θ$。
   - 该策略在给定当前状态 $s_t$ 时，输出下一个动作 $a_t$ 的概率分布，其数学形式为softmax变换：$π_θ(a_t|s_t) = [softmax(f_θ(s_t))]_a_t$。

3. **两个基本操作**：
   - **从策略中采样**：根据概率分布 $π_θ(·|s_t)$ 生成下一个词元（动作）。
   - **评估动作的对数似然**：计算给定状态下，某个特定动作（词元）的log概率，即 $log π_θ(a_t|s_t)$。这是计算策略梯度所必需的。

4. **在LLM中进行RL的具体含义**：
   - 在解决任务（如数学推理）时，$s_t$ 代表截至当前生成的**部分解决方案**，$a_t$ 则是解决方案的**下一个词元**。
   - 一个**回合（episode）** 在生成特定的结束标记（例如 `</answer>` 或 `<|endoftext|>`）时终止。

**总结**：本段核心是将语言模型的**自回归生成过程**形式化地定义为强化学习中的**策略**，明确了状态、动作、策略及所需的基本操作，为应用策略梯度等RL算法优化语言模型提供了清晰的理论框架。

## 6.2 Trajectories

### 内容概况
本段明确定义了在**强化学习（RL）** 背景下，特别是**与大型语言模型（LLMs）结合时**的“轨迹”这一核心概念。

### 要点总结
1.  **轨迹的定义**：
    *   轨迹（Trajectory）是指智能体在**有限步数内**所经历的状态（$s_t$）与动作（$a_t$）的交错序列，表示为：
        $τ = (s_0, a_0, s_1, a_1, …, s_T, a_T)$。
    *   轨迹长度 $T$ 的终止条件：生成了**文本结束标记**（如EOS），或达到了预设的**最大生成长度（token数）**。

2.  **初始状态**：
    *   初始状态 $s_0$ 来源于一个**初始状态分布** $ρ_0(s_0)$。
    *   在LLM的RL设定中，这个分布特指**经过格式化的提示（prompt）的分布**，即每条轨迹始于一个给定的文本提示。

3.  **状态转移**：
    *   **通用RL环境**：下一个状态 $s_{t+1}$ 由环境动态 $P(·|s_t, a_t)$ 决定，具有随机性。
    *   **LLM的RL环境**：环境是**确定性的**。下一个状态是**将旧的状态（文本前缀）与新生成的动作（token）进行拼接**，即：$s_{t+1} = s_t || a_t$。

4.  **术语说明**：
    *   “轨迹”也常被称为**回合** 或**展开**，这些术语在本上下文中可互换使用。

## 6.3 Rewards and Return

### 内容概况
本段系统性地阐述了**强化学习（RL）框架中“奖励”与“回报”的核心概念及其数学形式化**。它特别聚焦于**“已验证领域”**（如数学问题求解）这一特定场景，定义了适用于此类任务的稀疏奖励机制，并引出了智能体（如语言模型）的最终优化目标。

### 要点总结
1.  **奖励的定义与特性**：
    *   **瞬时奖励** $r_t$ 是一个标量函数 $R(s_t, a_t)$，用于评估在状态 $s_t$ 下采取动作 $a_t$ 的即时质量。
    *   **验证奖励**：在数学推理等“已验证领域”的标准做法是，**只对最终结果给予奖励**。具体为：若完整轨迹（最终答案）与事实依据（奖励函数）相符，则给予奖励 $r_T = 1$；否则为 $0$。**中间步骤的奖励均为零**。这是一种典型的“稀疏奖励”设置。

2.  **回报的两种计算方式**：
    *   **有限视界无折扣回报**：适用于有明确终止点的任务（如生成一段文本）。回报是轨迹上所有瞬时奖励的简单求和。
    *   **无限视界折扣回报**：适用于持续或无限的任务。回报是未来所有奖励的折现和，折扣因子 $γ$ 体现了对近期奖励的偏好。

3.  **场景选择与智能体目标**：
    *   由于文本生成等任务具有自然终止点（如生成结束符或达到最大长度），因此**采用无折扣回报公式**更为合适。
    *   智能体的终极目标是**最大化其策略参数 θ 下的期望回报** $J(θ)$。这引出了寻找最优参数 $θ*$ 的优化问题。

### 公式打印
以下是图片中出现的核心公式：

1.  **已验证的终端奖励函数**：
    $$
    r_T = R(s_T, a_T) := \begin{cases} 1 & \text{if the trajectory } s_T\|a_T \text{ matches the ground-truth according to our reward function}\\ 0 & \text{otherwise.} \end{cases}
    $$

2.  **有限视界无折扣回报（finite-horizon undiscounted returns）**：
    $$
    R(\tau) := \sum_{t=0}^{T} r_t \qquad(5)
    $$

3.  **无限视界折扣回报（infinite-horizon discounted returns）**：
    $$
    R(\tau) := \sum_{t=0}^{\infty} \gamma^{t} r_t, \qquad 0 < \gamma < 1 \qquad(6)
    $$

4.  **期望回报（智能体目标）**：
    $$
    J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ R(\tau) \right] \qquad(7)
    $$

5.  **优化问题**：
    $$
    \theta^{*} = \arg \max_{\theta} J(\theta) \qquad(8)
    $$

## 6.4 Vanilla Policy Gradient

### 🔍 REINFORCE算法推导总览

推导的核心目标是：**从期望回报最大化问题出发，得到可计算的政策梯度表达式**。

```
推导路径：
期望回报 J(θ) → 轨迹求和形式 → 对数导数技巧 → 环境无关性假设 → REINFORCE梯度
```

### 📐 逐步推导详解

#### 步骤1：定义优化目标
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)] \qquad(7)
$$
**含义**：期望回报 $J(θ)$ 是策略 $π_θ$ 下轨迹回报 $R(τ)$ 的期望值。

#### 步骤2：将期望展开为求和形式
$$
\nabla_{\theta}J(\theta) = \nabla_{\theta}\mathbb{E}_{\tau\sim\pi_{\theta}}[R(\tau)] = \nabla_{\theta}\sum_{\tau}P(\tau|\theta)R(\tau)
$$
**关键转换**：期望 $E[·]$ 可以写为对所有可能轨迹的概率加权和。

#### 步骤3：梯度算子移入求和
$$
= \sum_{\tau}\nabla_{\theta}P(\tau|\theta)R(\tau)
$$
**合理性**：假设求和与梯度操作可交换（通常成立）。

#### 步骤4：应用**对数导数技巧**
这是推导的**核心数学技巧**：

$$
\nabla_{\theta}P(\tau|\theta) = P(\tau|\theta)\nabla_{\theta}\log P(\tau|\theta) \qquad(13)
$$

**证明这个技巧**：
```python
# 对数导数技巧的简单证明
def prove_log_derivative_trick():
    """
    证明：∇P = P * ∇logP
    """
    # 已知：logP = ln(P)，所以 ∇logP = (1/P) * ∇P
    # 因此：∇P = P * ∇logP
    return "证毕"
```

#### 步骤5：代回得到期望形式
$$
= \sum_{\tau}P(\tau|\theta)\nabla_{\theta}\log P(\tau|\theta)R(\tau) = \mathbb{E}_{\tau\sim\pi_{\theta}}[\nabla_{\theta}\log P(\tau|\theta)R(\tau)]
$$

现在问题转化为：**计算轨迹概率的对数梯度** $∇_θ log P(τ|θ)$。

#### 步骤6：分解轨迹概率
轨迹概率可以分解为三部分的乘积：

$$
P(\tau\mid\theta)=\rho_0(s_0)\prod_{t=0}^T P(s_{t+1}\mid s_t,a_t)\pi_\theta(a_t\mid s_t) \qquad(11)
$$

**三项含义**：
1. $ρ₀(s₀)$：初始状态分布
2. $P(sₜ₊₁|sₜ,aₜ)$：环境动态（状态转移概率）
3. $π_θ(aₜ|sₜ)$：策略概率

#### 步骤7：取对数简化计算
$$
\log P(\tau\mid\theta)=\log\rho_0(s_0)+\sum_{t=0}^T[\log P(s_{t+1}\mid s_t,a_t)+\log\pi_\theta(a_t\mid s_t)] \qquad(12)
$$

**优势**：乘积变求和，梯度计算更简单。

#### 步骤8：应用**环境无关性假设**
**关键假设**：只有策略项依赖于参数θ：

$$
\nabla_{\theta}\rho_{0} = \nabla_{\theta}P = \nabla_{\theta}R(\tau) = 0 \qquad(14)
$$

**物理意义**：环境动态（$P$）、初始状态（$ρ₀$）、奖励函数（$R$）都是**环境固有的属性**，不随我们优化的策略参数$θ$改变。

因此：
$$
\nabla_{\theta}\log P(\tau|\theta) = \sum_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})
$$

#### 步骤9：得到最终REINFORCE梯度
将步骤8的结果代回步骤5：

$$
\nabla_{\theta}J(\pi_{\theta}) = \mathbb{E}_{\tau\sim\pi_{\theta}}\left[\left(\sum_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})\right)R(\tau)\right] \qquad(10)/(20)
$$

### 🎯 直观理解

#### 梯度表达式的物理意义
$$
\nabla_{\theta}J(\theta) = \mathbb{E}\left[\underbrace{\sum_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})}_{\text{策略变化方向}} \times \underbrace{R(\tau)}_{\text{回报权重}}\right]
$$

**解读**：
- **∑∇logπ**：轨迹中每个动作的**策略梯度方向**（增加该动作概率的方向）
- **R(τ)**：整个轨迹的**回报**，作为权重因子
- **期望E[·]**：对所有可能轨迹取平均

#### 工作机理
```python
def reinforce_intuition():
    """REINFORCE直观工作机制"""
    
    mechanism = {
        "高回报轨迹": "梯度更新会增加该轨迹中所有动作的概率",
        "低回报轨迹": "梯度更新会减少该轨迹中所有动作的概率", 
        "中性效果": "所有动作的梯度方向会相互抵消",
        "收敛方向": "策略逐渐偏向产生高回报的动作序列"
    }
    
    return mechanism
```

### 🔧 实际计算：从理论到实践

#### 理论梯度 vs 实际估计
理论公式是期望形式，实际中我们用**蒙特卡洛估计**：

$$
\nabla_{\theta}J(\pi_{\theta}) \approx \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}(a_{t}^{(i)}|s_{t}^{(i)})R(\tau^{(i)}) \qquad(21)
$$

#### 实现伪代码
```python
def reinforce_update(policy_network, trajectories, optimizer, learning_rate):
    """REINFORCE参数更新实现"""
    
    policy_gradients = []
    
    for trajectory in trajectories:
        states, actions, returns = trajectory  # returns = R(τ)
        
        log_probs = []
        for state, action in zip(states, actions):
            # 计算每个动作的对数概率
            action_dist = policy_network(state)
            log_prob = torch.log(action_dist[action])
            log_probs.append(log_prob)
        
        # 计算轨迹的梯度贡献
        policy_gradient = -torch.sum(torch.stack(log_probs)) * returns
        policy_gradients.append(policy_gradient)
    
    # 平均梯度
    total_gradient = torch.mean(torch.stack(policy_gradients))
    
    # 梯度上升（注意：PyTorch是最小化，所以取负号）
    optimizer.zero_grad()
    (-total_gradient).backward()  # 最大化 = 最小化负值
    optimizer.step()
```

## 6.5 Policy Gradient Baselines

### 📊 内容概况

1. **问题提出**：普通策略梯度（Vanilla Policy Gradient/REINFORCE）存在的**高方差问题**
2. **解决方案**：引入**仅依赖于状态的基线函数**来减少方差而不引入偏差
3. **数学证明**：详细证明了带基线策略梯度的**无偏性**
4. **实用建议**：在PyTorch等框架中实现策略梯度时的注意事项和评估指标

### 🎯 要点总结

#### 第一部分：基线技术的引入与原理

1. **核心问题**：
   - 普通策略梯度的梯度估计**方差很高**
   - 高方差导致训练不稳定、收敛缓慢

2. **解决方案**：
   - 引入**基线函数** $b(s_t)$，该函数**仅依赖于状态**（不依赖于动作）
   - 从回报 $R(τ)$ 中减去基线，得到**带基线的策略梯度**

3. **数学形式**：
   $$
   B = E_{\tau\sim\pi_{\theta}}\left[\sum_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})(R(\tau)-b(s_{t}))\right] \qquad(22)
   $$

4. **基线选择示例**：
   - **在策略价值函数**：$V^π(s) = E_{τ∼π_θ}[R(τ) | s_t = s]$
   - 直观意义：$R(τ) - V^π(s_t)$ 表示**实际轨迹比期望好多少**

5. **技术原理**：
   - 这是一种**控制变量法**（Control Variate）
   - 通过减去与估计量相关的项来减少方差，同时**不引入偏差**

#### 第二部分：无偏性证明与实现指导

1. **无偏性证明**：
   - 只要基线**仅依赖于状态**，带基线的策略梯度就是**无偏的**
   - 关键数学性质：得分函数（Score Function）的期望为零
     $$
     E_{x\sim P_{\theta}}[\nabla_{\theta}\log P_{\theta}(x)] = 0
     $$

2. **证明过程**：
   - 将基线梯度分解为两部分（公式23）
   - 证明基线项的期望为零（公式24-25）
   - 得出结论：$B = ∇_θJ(π_θ)$，即**基线梯度等于原始梯度**

3. **实用实现**：
   - 在PyTorch中，定义**策略梯度损失**（pg_loss）：
     $$
     pg\_loss = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T}\log\pi_{\theta}(a_{t}^{(i)}|s_{t}^{(i)})(R(\tau^{(i)})-b(s_{t}^{(i)})) \qquad(26)
     $$
   - **重要说明**：`pg_loss` 不是传统意义的损失函数
     - 调用 `pg_loss.backward()` 是为了计算近似策略梯度 $ĝ$
     - 不应在训练/验证集上报告 `pg_loss` 作为评估指标
     - 好的验证 `pg_loss` 不表示模型泛化能力强


### 🎪 在您项目中的应用建议

对于您的MATH推理任务，基线技术特别重要：

```python
# 在您的REINFORCE实现中考虑基线
def compute_pg_loss_with_baseline(trajectories, baseline_function):
    """
    计算带基线的策略梯度损失
    """
    losses = []
    
    for trajectory in trajectories:
        states, actions, returns = trajectory
        baseline_values = baseline_function(states)  # 计算基线
        
        # 计算每个时间步的损失
        for t, (state, action) in enumerate(zip(states, actions)):
            log_prob = policy_network.get_log_prob(state, action)
            advantage = returns[t] - baseline_values[t]  # 优势函数
            losses.append(-log_prob * advantage)  # 负号因为PyTorch最小化
    
    return torch.mean(torch.stack(losses))
```

### ✅ 总结

**策略梯度基线的核心价值**：

1. **解决实际问题**：显著降低梯度估计的方差，提高训练稳定性
2. **理论保证**：在仅依赖于状态的条件下保持无偏性
3. **实用灵活**：可与多种基线函数结合，如价值函数、移动平均等
4. **实现简单**：在现有REINFORCE框架上只需简单修改

**关键启示**：
- 基线是改进策略梯度算法的**基础且有效**的技术
- 理解其数学原理有助于正确实现和调试
- 在实践中应**始终关注真正的优化目标**（奖励），而非中间损失值

## 6.6 Off-Policy Policy Gradient

| 特性 | 在线策略 (REINFORCE) | 离线策略 (PPO/GRPO) |
|------|-------------------|-------------------|
| **数据收集** | 必须用最新策略 | 可用旧策略数据 |
| **数据重用** | 不能重用 | 可以重用 |
| **更新次数** | 一批数据一次更新 | 一批数据多次更新 |
| **计算效率** | 低 | 高 |


### 第一部分：在线策略梯度（REINFORCE）的问题

1. **REINFORCE的在线策略本质**：
   - 训练数据必须从**正在优化的当前策略**$π_θ$中采样
   - 算法需要不断用**最新策略**收集新的轨迹数据

2. **算法三步骤**：
   1. 从π_θ采样一批轨迹：$\{\tau^{(i)}\}_{i=1}^{N}$
   2. 近似策略梯度：$$\widehat{g}=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T}\nabla_\theta\log\pi_\theta(a_t^{(i)}|s_t^{(i)})R(\tau^{(i)})$$
   3. 更新参数：$$\theta\leftarrow\theta+\alpha\widehat{g}$$

3. **效率问题**：
   - 每收集一批新轨迹**只能进行一次梯度更新**
   - 语言模型（LM）的**行为在单步中变化不大**，导致采样效率低下
   - 需要大量推理计算来收集新数据

### 第二部分：离线策略梯度

1. **核心思想**：
   - 使用**旧策略**$π_{θ_{old}}$采样的轨迹来优化**当前策略**$π_θ$
   - 打破了数据收集与策略优化的耦合
   - 允许**数据重用**，提高采样效率

2. **离线策略梯度公式**：
   $$
   \widehat{g}_{\text{off-policy}}=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T}\frac{\pi_{\theta}(a_{t}^{(i)}|s_{t}^{(i)})}{\pi_{\theta_{\text{old}}}(a_{t}^{(i)}|s_{t}^{(i)})}\nabla_{\theta}\log\pi_{\theta}(a_{t}^{(i)}|s_{t}^{(i)})R(\tau^{(i)})
   $$

3. **关键技术**：
   - **重要性采样**：通过权重比$π_θ/π_{θ_{old}}$纠正分布偏差
   - 这是**普通策略梯度的重要性采样版本**
   - 要求新旧策略**不能差异太大**，否则重要性权重方差会很大

4. **实际应用**：
   - 现代算法如**PPO、GRPO**都采用离线策略思想
   - 允许使用**历史数据**进行多次更新
   - 显著提高**数据利用效率**，适合语言模型等计算密集型场景



