本文主要整理CS336 Mixture of expert章节的主要内容。

## 1 - MoE结构、非MoE结构区别

![MoE](https://pic1.zhimg.com/v2-d356729ca9fb57095bd86250cd772c18_r.jpg)


| 特征           | 非MoE（稠密模型）                         | MoE（混合专家）                                      |
| -------------- | ------------------------------------------ | ---------------------------------------------------- |
| **核心思想**   | 所有参数处理所有Token                      | 每个Token由少数几个专家处理                      |
| **激活**       | 稠密（所有参数用于每个Token）               | **稀疏（仅少量专家用于每个Token）**                    |
| **模型容量（总参数量）** | 有限增加会导致计算成本急剧上升               | **可极大幅度提升（数千亿至万亿），代价较低**            |
| **单个Token计算成本（相对）** | 高（与模型总容量成正比）                 | **低（仅与少量专家容量相关，约为稠密模型的1.5-3倍）** |
| **推理效率（吞吐/延迟）** | 较低（无法利用稀疏性）                     | **理论上高（仅计算少量专家），但受通信/显存等限制**    |
| **显存占用**   | 相对低（与模型参数规模相当）               | **非常高（需加载所有专家参数，远超计算成本所需）**      |
| **训练复杂度** | 相对简单、成熟                           | **复杂（路由难训练，负载平衡，通信开销大）**         |
| **主要优势**   | 简单、稳健、易部署                         | **在可控计算成本下达到极高模型容量和潜力性能**        |
| **核心挑战**   | 提升容量导致计算成本剧增                   | **显存需求、通信开销、训练稳定性（路由、负载平衡）** |

**简单来说：**

*   **非MoE模型：** 就像一个大型委员会，每个人（参数）每次会议（处理每个Token）都必须工作发言（参与计算），效率不高。
*   **MoE模型：** 像一个拥有大量专家（可能几百个）的公司，每封邮件（每个Token）进来时，一个智能路由器（门控）会快速决定这封邮件最适合交给哪1-2个专家处理。这样绝大部分专家在绝大部分时间都在休息（不激活），公司整体效率高（计算少），拥有的知识库（总参数量）却可以非常庞大。**核心是“能力按需分配”。**

MoE模型的出现是为了突破模型容量（参数量）与计算成本/效率之间的瓶颈，使得构建万亿参数级别的模型在可接受的计算开销下成为可能。然而，这也带来了显著的显存和系统工程挑战。

## 2 - Top-K Routing / Hash Routing

![Top-K Routing / Hash Routing](https://pica.zhimg.com/v2-2481439c1ce4719846c37e4ade67b466_r.jpg)

![Top-K Routing](https://picx.zhimg.com/v2-663ce5d61316a5e242740a2cbf7c94b5_1440w.jpg)

| 特性              | Top-K Routing (软路由)                       | Hash Routing (硬路由)                         |
|------------------|---------------------------------------------|----------------------------------------------|
| **核心原理**      | • 可学习门控网络计算专家权重(Softmax) <br>• 选择 Top-K <br>• 专家加权输出 | • 固定哈希函数处理 `token_id`/`position_id`<br>• 哈希值取模 `% N` 分配专家 <br>• 固定分配单一专家输出 |
| **设计哲学**      | **智能匹配、专业化优先、可控稀疏** <br>• 学习 Token 语义与专家能力匹配 <br>• `K` 控制稀疏度与鲁棒性 <br>• 追求最大化整体模型质量与表达能力 | **极致简单、均衡优先、零开销路由** <br>• 路由规则预定义且零学习成本 <br>• 依赖哈希数学特性天然均衡 <br>• 追求最小路由开销与最大计算利用率 |
| **关键优势**      | • 智能路由，专家可专业化 <br>• 表达潜力大 <br>• 鲁棒性好 (`K>1`)    | • 路由速度极快，零学习开销 <br>• 天然负载均衡（理想分布下）<br>• 实现异常简单 <br>• 负载绝对均衡 (理想情况) |
| **核心缺点**      | • 路由网络需学习，难优化 <br>• 负载均衡是重大挑战 (需额外约束) <br>• 门控计算有小开销 | • **无法实现专家专业化（最大缺点！）**<br>• 路由僵化，与语义无关 <br>• 模型潜力上限低 <br>• 性能依赖输入分布假设 |
| **专家角色**      | 专业分化，各有所长                           | 基本等同（或随机差异），仅为并行计算单元           |
| **负载均衡**      | 严重挑战，需设计保证机制 (辅助损失/Capacity)     | 天然较均衡（依赖哈希均匀性），无需额外机制          |
| **路由类型**      | **软路由** (含权重融合)                      | **硬路由** (单一专家输出)                     |
| **计算开销**      | 有（门控网络计算，但通常较小）                 | 几乎为零                                     |
| **系统复杂度**    | 较高 (动态路由，需通信优化)                   | 较低 (规则简单，易于预分配)                   |
| **代表场景**      | 现代主流 LLM MoE (如 Mixtral 8x7B、DeepSeek-MoE、GPT-MoE) | 早期探索、极度追求速度/极简实现而牺牲性能的场景、特征工程为主的系统 |

---

### 🎯 总结一句话精髓

*   **`Top-K Routing`:** 💡 **“为每个 Token 动态、智能地选出最合适的一两个专家（K个），目标是最大程度提升模型质量和专业分工潜力。”**
*   **`Hash Routing`:** ⚡ **“以最快速度、零开销、完全公平地将 Token 随机分配（K=1）给专家运行，目标是最大化系统吞吐和计算利用率。”**

> 选择哪条路，取决于你的 **核心目标**：
> * 目标是打造 *最有潜力突破性能极限* 的大模型？选 **`Top-K` 路由**（尽管训练挑战大）。
> * 目标是 *在特定系统限制下跑得最快*？选 **`Hash 路由`**（但需接受性能天花板）。

## 3 - Shared experts

![Shared experts](https://pic2.zhimg.com/v2-e42c56107573a1e06d9f1742a4a97f15_1440w.jpg)

![Shared experts](https://pic2.zhimg.com/v2-9ceae9a6039edfc9f972d55a8050c6eb_1440w.jpg)

## 4 - Noisy Top-K Gating

在 MoE 发展历程中，**Shazeer et al. 2017 提出的「Noisy Top-K Gating」** 是奠基性工作之一。其核心思想是：**在路由分数中添加高斯噪声，以提升模型探索能力并缓解负载不均衡问题**，成为现代 MoE 路由标准设计的起点。下面从**原理本质、训练目标、实现细节**进行深入剖析：

---

### 🧠 一、核心设计原理

#### 1. **基本目标：解决 MoE 训练两大痛点**
   - **探索不足（Exploration）**：路由网络易陷入局部最优（总是选某几个专家），其他专家未被训练（“死专家”问题）。
   - **负载不均衡（Load Imbalance）**：少数专家被过度激活，多数专家闲置。

#### 2. **核心创新：带噪分数 = 路由信号 + 高斯噪声**
   - 路由决策仍依赖 **Top-K 选择**，但计算分数时引入扰动：
     $$ \tilde{s}_i = \frac{s_i + \epsilon_i \cdot W_{\text{noise}}}{\text{temperature}} $$
     - $s_i$：原始路由分数（`router_network(x)`)
     - $\epsilon_i \sim \mathcal{N}(0, 1)$：独立高斯噪声
     - $W_{\text{noise}}$：**可学习的噪声缩放因子**（关键！）
     - $\text{temperature}$：软化参数（≈1.0）

#### 3. **噪声的作用机制：**
   - **训练初期**：$W_{\text{noise}}$ 较大 → 噪声显著 → 路由选择有**强随机性**（增加探索）。
   - **训练后期**：模型学习降低 $W_{\text{noise}}$ → 噪声减弱 → 路由收敛至**确定性策略**（更好利用专家）。
   - **负载均衡引导**：噪声使「过度热门专家」的分数发生波动，偶尔被抑制，让冷门专家有机会被选中。

#### 4. **数学目标：自动学习探索强度**
   $$ \min_{\theta, W_{\text{noise}}} \mathbb{E} \left[ \mathcal{L}_{\text{task}} + \lambda \cdot \text{Load\_Loss} \right] $$
   - 通过梯度下降学习 $W_{\text{noise}}$，**让模型自行决定何时需噪声、用多大噪声**。

---

### ⚙️ 二、具体实现步骤（代码级拆解）

以下是 **Shazeer et al. 2017** 的 Noisy Top-K Routing 的完整训练流程实现（以 PyTorch 风格为例）：

#### 步骤 1: 定义路由器及噪声参数
```python
class NoisyTopKRouter(nn.Module):
    def __init__(self, input_dim, num_experts, k=2, init_noise=1.0):
        super().__init__()
        self.k = k
        self.w_gate = nn.Linear(input_dim, num_experts, bias=False) # 路由权重
        self.w_noise = nn.Parameter(torch.ones(1) * init_noise)   # 可学噪声缩放因子

    def forward(self, x, train_mode=True):
        # x: [batch_size, seq_len, hidden_dim]
        s = self.w_gate(x)  # [batch, seq_len, num_experts]
        
        if not train_mode:
            # 推理时：无噪声，直接 Top-K
            probs = torch.softmax(s, dim=-1)
            topk_probs, topk_idx = probs.topk(self.k, dim=-1)
            return topk_probs, topk_idx
        
        # ============= 训练时：注入高斯噪声 =============
        # 生成与 s 同形的标准高斯噪声 (独立同分布)
        eps = torch.randn_like(s)  # ϵ ~ N(0,1)
        
        # 噪声缩放：s̃_i = s_i + w_noise * ϵ_i
        s_noisy = s + self.w_noise * eps
        
        # 计算带噪声的 softmax 概率
        probs_noisy = torch.softmax(s_noisy, dim=-1)
        
        # 选 Top-K 个专家及其原始分数（非噪声分数！）
        topk_probs, topk_idx = probs_noisy.topk(self.k, dim=-1)
        
        return topk_probs, topk_idx  # 返回噪声概率 & 专家索引
```

#### 步骤 2：前向传播时调用路由器 → 计算 MoE 层输出
```python
def moe_layer(x, router, experts):
    # 训练模式：调用带噪声的路由
    topk_probs, topk_idx = router(x, train_mode=True)  # probs: [B, S, K], idx: [B, S, K]
    batch_size, seq_len, _ = x.shape
    
    # 构造扁平化输入（方便并行计算）
    x_flat = x.view(-1, x.shape[-1])                 # [B*S, D]
    topk_idx_flat = topk_idx.view(-1, topk_idx.shape[-1]) # [B*S, K]
    
    # 为每个Token计算 K 个专家的输出（并行所有专家）
    expert_inputs = x_flat.unsqueeze(1).repeat(1, router.k, 1) # [B*S, K, D]
    
    # 收集选中的专家索引 -> 映射为专家模块计算
    expert_outputs = []
    for k in range(router.k):
        expert_k = topk_idx_flat[:, k]  # 第k个专家索引 [B*S]
        # 将输入路由给该专家处理（需高效实现，如 group_by 或 scatter）
        out_k = expertsexpert_inputs[:, k, :]
        expert_outputs.append(out_k)
    # → 输出 [B*S, K, D_out]

    # 加权融合：y = ∑(weight_k * output_k)
    weights = topk_probs.view(-1, router.k).unsqueeze(-1) # [B*S, K, 1]
    y_flat = torch.sum(weights * expert_outputs, dim=1)    # [B*S, D_out]
    
    # 恢复序列结构
    y = y_flat.view(batch_size, seq_len, -1)
    return y
```

#### 步骤 3：添加负载均衡损失（Load Balancing Loss）协同优化
```python
def load_balancing_loss(router_probs, expert_idx):
    # router_probs: [batch, seq, K] 选各专家的概率
    # expert_idx:   [batch, seq, K] 选的专家ID
    
    # 计算每个专家的「被选概率」（batch内平均）
    batch, seq, k = router_probs.shape
    num_experts = router.num_experts
    
    # 构造专家负载指示矩阵（稀疏 → 稠密）
    expert_mask = torch.zeros(batch * seq, num_experts, device=x.device) # [B*S, E]
    expert_mask.scatter_(1, expert_idx.view(-1, k), 1.0)  # [B*S, E] → 1表示被选中
    
    # 每个Token的K个专家权重之和 = 1，因此按Token平均即负载期望
    load_per_expert = expert_mask.mean(dim=0)  # [E]
    
    # 损失：L_balance = (负载期望)的平方和 → 鼓励均匀分配
    balancing_loss = torch.sum(load_per_expert ** 2) * num_experts
    return balancing_loss

# 总损失 = 任务损失 + λ * 均衡损失
total_loss = loss_task + 0.01 * load_balancing_loss(topk_probs, topk_idx)
```

#### 关键训练动态示意：
```text
          训练早期                                    训练后期
s_i    :  [1.0, 0.5, -0.1]                     [2.1, 0.2, 0.1]
W_noise:  1.0 (初始)                            0.01 (学习收敛)
ϵ_i    :  [1.3, -0.7, 0.5] (随机采样)            [0.01, -0.02, 0.001] (微小噪声)
s̃_i    :  [2.3, -0.2, 0.4]                     [2.11, 0.18, 0.101] 
→ 专家选择：波动大（Explore）                     → 稳定选择第一个专家（Exploit）
```

---

### 📌 三、工程优化要点（现代演进）

1. **门控简化**：后续工作（如 Switch Transformer）将门控网络简化为 → `s = x @ W_gate`（无偏置项）。
2. **噪声参数共享**：一个 `W_noise` 控制所有专家的噪声强度。
3. **分专家噪声**：更精细做法是每个专家独立学 `w_noise_i`（参数量增但更灵活）。
4. **结合 Gumbel**：后期将高斯噪声改为 **Gumbel噪声**（支持精准 Top-K 采样），形成当前主流方法。
5. **温度退火**：与 Gumbel-softmax 类似，设置降温策略 `temperature = max(0.1, 1 - step/10000)`。

---

### 💎 四、总结：Noise in Routing 的价值

> 🔥 **Shazeer 的「噪声路由」本质是将「探索-利用困境」建模为可学习过程：  
>   —— 训练初期以高斯噪声注入强探索，打破专家冷启动；  
>   —— 训练后期噪声衰减，路由收敛到高效确定性策略；  
>   —— 可学的 `W_noise` 让模型自己掌握探索节奏，  
>   —— 负载损失迫使专家负载分布更均衡。**

这种设计思想在 **GShard、Switch Transformer、GLaM、T5-MoE** 中均被继承与提升，奠定了万亿级稀疏大模型的基础。如果你正复现经典MoE或设计新型路由器，Shazeer 2017 仍是必经之路！🚀

## 5 - Stochastic Jitter

Fedus 等人在 2022 年 Switch Transformer 工作中提出的 **Stochastic Jitter（随机抖动）** 是一种新颖的路由扰动策略，旨在缓解专家脆弱性并提升模型鲁棒性。与 Shazeer 的加性高斯噪声不同，Stochastic Jitter 采用**乘法式均匀扰动**实现更可控的探索机制。以下从**设计原理、数学本质到代码实现**深入解析其工作流程：

---

### 🔍 一、核心设计原理与目标

#### 🧠 设计背景：
- **问题：** MoE 路由易收敛至**少数固定专家组合**（Brittle Experts），导致：
  1. **专家利用不足**（Underutilization）
  2. **负载严重失衡**
  3. **模型易受输入扰动影响**
- **目标：** 在**不显著增加计算开销**前提下，向路由决策注入**可控随机性**以提升鲁棒性。

#### ⚡ 创新点：**均匀分布乘法扰动**
- **扰动公式**：
  $$
  \tilde{s}_i = s_i \times (1 + \epsilon_i)
  $$
  - 原始路由分数 $s_i$（路由层输出）
  - $\epsilon_i \sim \text{Uniform}(-c, +c)$：**均匀分布噪声**，$c$ 为扰动幅度（如 0.5）
- **本质**：对路由分数进行**比例缩放**而非加减式偏移

#### 💡 关键特性：
| **维度**        | **Stochastic Jitter**                     | **Shazeer Gaussian Noise**          |
|-----------------|-------------------------------------------|-------------------------------------|
| **噪声类型**     | ❌ 均匀分布（Uniform）乘法扰动              | ✅ 高斯分布（Gaussian）加法扰动       |
| **扰动方向**     | ➕➖ 分数按比例缩放（Scaling）                | ➕➖ 分数线性偏移（Offset）           |
| **参数性质**     | ⚠️ 固定幅度 $c$（超参）                     | ✅ 可学习噪声权重 $W_{\text{noise}}$ |
| **影响范围**     | 🔄 分数越高，扰动绝对幅度越大（相对稳定）      | 🔄 所有分数同等幅度震荡              |

> 🔥 **设计哲学：**  
> **“通过比例扰动保留分数间相对关系，避免低分专家被随机噪声过度激活，使探索过程更具方向性”**

---

### ⚙️ 二、工作流程与具体实现

#### 📜 伪代码流程：
```
输入：token 向量 x, 抖动幅度 c, 专家数 N
1. s = router_network(x)                   // 计算原始路由分数 [s1, s2, ..., sN]
2. 对每个 s_i 生成扰动噪声：
      ε_i = Uniform(-c, +c)               // 均匀采样噪声
      s̃_i = s_i * (1 + ε_i)                // 乘法扰动
3. p_i = Softmax(s̃_i)                      // 计算带扰动概率
4. 选取 Top-K 专家索引（基于 p_i）
5. 加权计算专家输出：y = ∑(w_i * Expert_i(x))
```

#### 🐍 Python 实现（PyTorch）
```python
class StochasticJitterRouter(nn.Module):
    def __init__(self, input_dim, num_experts, k=1, jitter_ratio=0.5):
        super().__init__()
        self.k = k
        self.jitter_ratio = jitter_ratio  # 扰动比例 c (e.g. 0.5)
        self.router = nn.Linear(input_dim, num_experts)  # 路由层

    def forward(self, x, training=True):
        # 原始路由分数 [batch, seq_len, num_experts]
        s = self.router(x)  
        
        if not training:
            # 推理模式：无扰动
            probs = torch.softmax(s, dim=-1)
            return probs.topk(self.k, dim=-1)  # topk_probs, topk_indices
        
        # ========== 训练模式：注入Stochastic Jitter扰动 ==========
        # 生成均匀噪声：范围 [-c, +c], 形状与 s 相同
        eps = torch.empty_like(s).uniform_(-self.jitter_ratio, self.jitter_ratio)
        
        # 乘法扰动：s̃_i = s_i * (1 + ε_i)
        s_tilde = s * (1 + eps)
        
        # 计算扰动后概率分布
        probs = torch.softmax(s_tilde, dim=-1)
        
        # 选择 Top-K 专家（概率与索引）
        topk_probs, topk_idx = probs.topk(self.k, dim=-1)
        return topk_probs, topk_idx
```

#### 🧩 MoE 层整合调用：
```python
def switch_moe_layer(x, router, experts):
    batch, seq_len, d_model = x.shape
    x_flat = x.reshape(-1, d_model)  # [batch*seq_len, d_model]
    
    # 获取路由决策（含扰动）
    probs, expert_idx = router(x_flat)  # probs: [B*S, K], expert_idx: [B*S, K]
    
    outputs = torch.zeros_like(x_flat)  # 准备输出
    
    # 为每个 token 处理所选专家
    for k in range(router.k):
        # 创建专家选择掩码
        mask = (torch.arange(experts.num_experts, device=x.device)[None,:] 
                == expert_idx[:, k].unsqueeze(1))  # [B*S, E]
        
        # 计算每个专家处理的数据子集
        for exp_i in range(experts.num_experts):
            token_idx = mask[:, exp_i].nonzero(as_tuple=True)[0]  # 选该专家的token
            if len(token_idx) > 0:
                # 调用专家处理其分配到的 token
                expert_in = x_flat[token_idx]  # 专家输入
                expert_out = expertsexpert_in  # 专家输出
                
                # 加权输出 (乘以对应路由权重)
                kth_weight = probs[token_idx, k].unsqueeze(1)  # [tokens, 1]
                outputs[token_idx] += kth_weight * expert_out
    
    # 恢复原始形状
    return outputs.view(batch, seq_len, d_model)
```

---

### 📊 三、动态效果分析与参数设定

#### ⚖️ 扰动幅度 $c$ 的影响：
| **幅度 c** | 模型行为                                                                 | 适用场景                 |
|------------|--------------------------------------------------------------------------|--------------------------|
| **0.0**    | ❌ 无扰动，退化至原始路由                                                 | 基准测试                 |
| **0.1**    | ⚠️ 弱扰动，专家选择稳定性高                                              | 高精度敏感任务           |
| **0.3~0.5**| ✅ 推荐范围：平衡探索与利用                                               | Switch Transformer 默认  |
| **>0.7**   | 🧪 强扰动，路由严重随机化（可能损害性能）                                   | 需配合正则项探索实验     |

#### 🔧 工程建议：
1. **无需学习参数**：$c$ 固定为超参（简化实现）
2. **激活位置**：在 softmax **前**施加扰动
3. **分布选择**：均匀分布（Uniform）比高斯分布更不易产生极端值
4. **组合技巧**：与 **Load Balancing Loss** 配合使用效果更佳（详见下文）

---

### 🔗 四、与负载均衡的协同优化

Stochastic Jitter 通常需配合负载均衡损失使用：
```python
def load_balancing_loss(router_logits, expert_indices):
    num_experts = router_logits.shape[-1]
    
    # 计算每个token对各专家的总权重贡献
    router_probs = torch.softmax(router_logits, dim=-1)  # [B*S, E]
    selection_mask = torch.zeros_like(router_probs)      # [B*S, E]
    
    # 标记被选中的专家位置
    selection_mask.scatter_(1, expert_indices, 1.0)      # [B*S, E]
    
    # 专家被选中概率的期望值（沿batch维平均）
    load_per_expert = selection_mask.mean(dim=0)         # [E]
    
    # 专家权重期望值（重要性加权）
    importance_per_expert = router_probs.mean(dim=0)      # [E]
    
    # 关键公式：负载均衡损失 = 方差(期望负载) * 专家数
    loss_balance = torch.var(load_per_expert) * num_experts
    
    # 可选：增加重要性方差项（减轻高分专家被频繁选择）
    loss_importance = torch.var(importance_per_expert) * num_experts
    
    return 0.5 * (loss_balance + loss_importance)

# 调用 (在训练循环中)
total_loss = task_loss + 0.01 * load_balancing_loss(s, expert_idx)
```

---

### 💎 五、总结：价值与局限

#### ✅ 核心贡献：
1. **突破专家脆弱性**：通过比例扰动使路由更鲁棒
2. **计算零开销**：仅增加均匀采样操作，不影响并行
3. **无需学习噪声参数**：超参化简化训练流程
4. **保留分数分布特性**：乘法扰动维持分数关系稳定

#### ⚠️ 潜在局限：
- 扰动幅度 $c$ 需人工调优（无自适应能力）
- 缺乏类似 $W_{noise}$ 的自衰减机制
- 极端场景下可能抑制高分专家（$1+\epsilon_i$可能使 $s_i$ 为负）

> 🔥 **终极哲学：**  
> **Stochastic Jitter 将路由鲁棒性转化为「分数空间的比例抖动」，通过均匀分布的乘法扰动在高效性（零参/计算轻量）与有用性（打破专家固化）间取得创新平衡，成为现代大规模稀疏模型训练的基石组件之一。**

该方法在 **Google Switch Transformer、T5-XL-MoE** 等千亿级模型中实证有效。如需在自定义 MoE 中实现，推荐先尝试 $c=0.4$ 配合负载损失使用（完整代码见上文）。

## 6 - Heuristic balancing losses (switch transformer)

![switch transformer](https://pic3.zhimg.com/v2-45360579ff1c6ae63596a06855a699e6_1440w.jpg)

```python
def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
return overall_loss * num_experts
```

这段代码实现了 Switch Transformer 中提出的负载均衡损失函数。让我们逐行解析其工作原理和实现细节：

### 🧠 核心原理
负载均衡损失的目标是惩罚路由决策不均衡的情况，确保所有专家都能被充分利用。公式为：
```
loss = num_experts * Σ(专家i的令牌比例 * 专家i的平均路由概率)
```

### 📝 代码解析

#### 1. 输入处理
```python
if gate_logits is None or not isinstance(gate_logits, tuple):
    return 0
```
- 如果没有门控 logits 或不是元组（多层），直接返回 0

#### 2. 数据准备
```python
if isinstance(gate_logits, tuple):
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) 
                                         for layer_gate in gate_logits], dim=0)
```
- 将所有层的门控 logits 拼接成一个大矩阵
- 确保所有张量都在同一设备上

#### 3. 路由计算
```python
routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
_, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
```
1. 计算路由权重（softmax 概率）
2. 选择每个 token 的 top_k 专家
3. 创建专家掩码（one-hot 编码）

#### 4. 无注意力掩码情况
```python
if attention_mask is None:
    # 计算每个专家被选中的令牌比例
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    
    # 计算每个专家的平均路由概率
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
```
- `tokens_per_expert`: 每个专家被选中的平均比例
- `router_prob_per_expert`: 每个专家的平均路由概率

#### 5. 有注意力掩码情况
```python
else:
    batch_size, sequence_length = attention_mask.shape
    num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)
    
    # 扩展注意力掩码以匹配专家掩码形状
    expert_attention_mask = (...).reshape(-1, top_k, num_experts)
    
    # 计算加权令牌比例
    tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / 
                        torch.sum(expert_attention_mask, dim=0)
    
    # 扩展注意力掩码以匹配路由权重形状
    router_per_expert_attention_mask = (...).reshape(-1, num_experts)
    
    # 计算加权路由概率
    router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / 
                             torch.sum(router_per_expert_attention_mask, dim=0)
```
- 考虑注意力掩码，忽略填充 token
- 计算加权平均值而非简单平均

#### 6. 损失计算
```python
overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
return overall_loss * num_experts
```
- 计算乘积和：Σ(令牌比例 × 路由概率)
- 乘以专家数量进行缩放

### 🧩 关键设计点

1. **多层聚合**：
   - 将所有层的路由决策合并计算
   - 确保全局负载均衡

2. **注意力掩码处理**：
   - 精细处理填充 token
   - 避免无效 token 影响负载计算

3. **双指标乘积**：
   - 同时考虑实际分配比例(tokens_per_expert)
   - 和路由偏好强度(router_prob_per_expert)

4. **归一化处理**：
   - 使用均值而非总和
   - 确保指标与 batch 大小无关

### ⚖️ 数学原理

损失函数最小化以下表达式：
$$
L_{balance} = N \times \sum_{i=1}^{N} \left( \frac{\text{count}_i}{T} \right) \times \left( \frac{1}{T} \sum_{t=1}^{T} p_i^{(t)} \right)
$$
其中：
- $N$：专家数量
- $\text{count}_i$：分配给专家 $i$ 的 token 数量
- $T$：总 token 数量（考虑掩码）
- $p_i^{(t)}$：token $t$ 分配给专家 $i$ 的概率

### 🚀 工程优化建议

1. **数值稳定性**：
   ```python
   # 添加小值防止除零
   tokens_per_expert = ... / (torch.sum(...) + 1e-8)
   ```

2. **分布式计算**：
   ```python
   # 跨设备聚合统计数据
   if torch.distributed.is_initialized():
       # 聚合 tokens_per_expert 和 router_prob_per_expert
       ...
   ```

3. **稀疏优化**：
   ```python
   # 使用稀疏操作减少内存
   expert_mask = expert_mask.to_sparse()
   ```

4. **混合精度支持**：
   ```python
   with torch.cuda.amp.autocast():
       # 在自动混合精度上下文中计算
       ...
   ```

这段代码实现了 Switch Transformer 论文中描述的负载均衡机制，通过惩罚不均衡的路由决策，鼓励模型更均匀地使用所有专家资源。

## 7 - Heuristic balancing losses (Deepseek V1-V2)

![Deepseek V1-V2](https://pic4.zhimg.com/v2-e097120cb312e1e3692e74a0a2d78069_1440w.jpg)

## 8 - per-expert biases (Deepseek V3)

![Deepseek V3](https://pic2.zhimg.com/v2-08f82f7e7f3ab378b3a27b639679a9ff_1440w.jpg)

![Deepseek V3](https://pic4.zhimg.com/v2-03cb1dd4c8265b5ffbb16afad7282e15_1440w.jpg)