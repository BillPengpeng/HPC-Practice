本文主要整理CS336 Lecture 10 Inference章节的主要内容。

## 1. landscape

### 1. **推理的广泛应用场景**
- **直接应用**：聊天机器人、代码补全、批量数据处理。
- **模型评估**：例如评估模型遵循指令的能力。
- **测试阶段计算**：“思考”过程需要更多的推理计算。
- **强化学习训练**：涉及样本生成和评分。

### 2. **效率是核心关键（Why Efficiency Matters）**
- **核心论点**：训练是一次性成本，而推理是海量重复发生的成本。因此，推理效率直接关系到产品的用户体验和运营成本。
- **数据佐证**：材料中引用了两张图表（关于OpenAI的token消耗和Cursor代码行的生成）来直观展示推理规模的巨大。

### 3. **衡量推理效率的关键指标**
- **Time-to-first-token (TTFT)**：用户开始等待到收到第一个回复的时间，影响交互体验。
- **Latency (seconds/token)**：每个令牌出现的速度，影响交互流畅度。
- **Throughput (tokens/second)**：单位时间内处理的令牌总数，对批量处理应用至关重要。

### 4. **训练与推理的根本技术差异**
- **训练**：可以并行处理整个序列，能充分利用计算资源。
- **推理**：必须按顺序生成令牌，无法并行化，因此**更难充分利用计算能力**，优化挑战更大。

### 5. **行业生态与主要技术栈**
- **服务提供商**：
    - **闭源模型**：OpenAI, Anthropic, Google 等。
    - **开源权重模型**：Together, Fireworks, DeepInfra 等。
- **主流开源优化包**：
    - **vLLM**（来自伯克利）：以其高效的内存管理和吞吐量著称。
    - **TensorRT-LLM**（来自NVIDIA）：深度集成于NVIDIA硬件的推理优化器。
    - **TGI**（来自Hugging Face）：Hugging Face官方推出的文本生成推理工具。

## 2. review_transformer

![transformer](https://jax-ml.github.io/scaling-book/assets/img/transformer-diagram.png)
- Simplifications (following conventions): `F = 4*D, D = N*H, N = K*G, S = T`
- FLOPs for a feedforward pass: 6 * (B*T) * (num_params + O(T))

## 2. transformer示例图解释

本部分进行 **Grouped Multi-Query Attention (GQMA)** 的公式推导。我们将严格按照图中定义（如 `B, T, S, D, H, K, G`）来阐述，并直接使用分组数 `G`，避免使用 `D/H`。

图片中的关键关系是 **`G = H / K`**，即：
*   `H`： 查询头总数。
*   `K`： 键头数/值头总数（`K < H`）。
*   `G`： 分组数，表示有 `G` 组键值对，每组被 `H/G` 个查询头共享。

---

### 公式推导

#### 第1步：定义输入与线性投影

*   **输入**： `X`，形状为 `[B, T, D]`。
*   **查询投影**： 
    *   `Q = X @ W_Q`
    *   `W_Q` 形状为 `[D, H * (D/H)]`。为保持清晰，我们关注逻辑维度。投影后，`Q` 的形状为 `[B, T, H * (D/H)]`。为了后续分组，我们将其**重塑**为 `[B, T, H, D/H]`。**注意**： 虽然出现了 `D/H`，但它只是一个中间维度，我们接下来的操作将围绕 `G` 和 `H` 进行，不再深入处理它。
*   **键投影**：
    *   `K = X @ W_K`
    *   `W_K` 形状为 `[D, K * (D/H)]`。投影后，`K` 的形状为 `[B, S, K * (D/H)]`，然后重塑为 `[B, S, K, D/H]`。
*   **值投影**：
    *   `V = X @ W_V`
    *   同理，`V` 的形状为 `[B, S, K, D/H]`。

#### 第2步：核心步骤 - 引入分组维度 `G`

这是GQA与MHA最核心的区别。我们需要根据关系 `G = H / K` 对 `Q`、`K`、`V` 进行重塑。

1.  **重塑查询张量 `Q`**：
    *   当前 `Q` 形状： `[B, T, H, D/H]`。
    *   目标：将 `H` 个查询头分成 `G` 组，每组有 `H/G` 个头。因为 `G = H / K`，所以 `H/G = K`。
    *   **重塑操作**： 将 `H` 维度拆分成 `[G, K]` 两个维度。
    *   `Q_reshaped = Q.reshape(B, T, G, K, D/H)`
    *   为了方便后续计算，通常调整维度顺序为： `[B, G, K, T, D/H]`。
    *   **物理意义**： 现在，`Q` 张量被明确组织为 `G` 组，每组内有 `K` 个查询头。

2.  **重塑键张量 `K` 和值张量 `V`**：
    *   当前 `K` 形状： `[B, S, K, D/H]`。
    *   目标：`K` 和 `V` 本来就有 `K` 个头。我们需要为其显式添加一个“组”维度，以匹配 `Q` 的分组结构。因为总共有 `G` 组，所以 `K` 个键头实际上就是 `G` 组键头（因为 `K = G`？ 这里需要澄清：更准确地说，`K` 是键值头的总数，它等于分组数 `G`，即 `K = G`）。
    *   **重塑操作**： 直接在 `K` 和 `V` 的 `K` 维度前插入一个大小为1的维度，然后利用**广播机制**。
    *   `K_reshaped = K.reshape(B, S, 1, K, D/H)` -> 调整顺序为 `[B, 1, K, S, D/H]`。
    *   `V_reshaped = V.reshape(B, S, 1, K, D/H)` -> 调整顺序为 `[B, 1, K, S, D/H]`。
    *   **物理意义**： 现在，`K` 和 `V` 张量都被组织为“1个组，包含 `K` 个头”。这个“1”的维度在计算时会被广播到 `Q` 的 `G` 个组。

**至此，我们得到了分好组的张量：**
*   `Q`： `[B, G, K, T, D/H]` （含义：批次， 分组， 组内查询头数， 目标序列长， 头维度）
*   `K`： `[B, 1, K, S, D/H]` （含义：批次， (伪)分组， 键头数， 源序列长， 头维度）
*   `V`： `[B, 1, K, S, D/H]` （含义：批次， (伪)分组， 值头数， 源序列长， 头维度）

#### 第3步：注意力计算 - 广播机制

现在进行注意力计算。`K` 和 `V` 在维度1（分组维度）上的大小为1，而 `Q` 的大小为 `G`。根据广播规则，`K` 和 `V` 会在该维度上被复制 `G` 次，从而在计算中变为 `[B, G, K, S, D/H]`。

1.  **计算注意力分数**：
    *   **操作**： `Scores = Q @ K.transpose(-1, -2)` （即对最后一个维度 `D/H` 进行点积）
    *   **维度变化**：
        *   `Q`: `[B, G, K, T, D/H]`
        *   `K^T`: `[B, G, K, D/H, S]` （经过广播和转置）
        *   **`Scores`**: `[B, G, K, T, S]`
    *   **物理意义**： 对于 `G` 个组中的每一个，计算组内 `K` 个查询头与（广播后的）`K` 个键头的注意力分数。`T` 和 `S` 的不同体现了编解码器或自回归推理的特性。

2.  **应用Softmax**：
    *   `P = Softmax(Scores / sqrt(d_k), dim=-1)`， `d_k` 是 `D/H`。
    *   维度不变： `[B, G, K, T, S]`。

3.  **加权求和**：
    *   **操作**： `O = P @ V`
    *   **维度变化**：
        *   `P`: `[B, G, K, T, S]`
        *   `V`: `[B, G, K, S, D/H]` （经过广播）
        *   **`O`**: `[B, G, K, T, D/H]`
    *   **物理意义**： 使用注意力权重 `P` 对（广播后的）值向量 `V` 进行加权求和，得到每个位置的输出。

#### 第4步：合并输出

1.  **合并分组维度**： 现在需要将分组结构还原。
    *   `O_merged = O.reshape(B, T, G * K, D/H)`。 因为 `G * K = H`，所以此操作的结果是 `[B, T, H, D/H]`。
2.  **合并最后两个维度**： 将头维度 `H` 和头维度 `D/H` 合并，还原为模型维度 `D`。
    *   `O_final = O_merged.reshape(B, T, D)`。
3.  **输出投影**： 最后，通过输出投影矩阵 `W_O` 得到该注意力层的最终结果。

---

### 核心总结（直接使用 `G`）

**Grouped Multi-Query Attention 的公式推导核心可总结为三步变形：**

1.  **投影与重塑（引入G）**：
    *   将 `Q` 的 `H` 个查询头**重塑**为 `[G, K]` 两个维度（`G` 组，每组 `K` 个头）。
    *   将 `K` 和 `V` 的 `K` 个键值头**重塑**为 `[1, K]`，即人为添加一个组维度以便广播。

2.  **分组广播计算**：
    *   在注意力计算中，`K` 和 `V` 的“组”维度（大小为1）会**广播**到 `Q` 的“组”维度（大小为 `G`）。
    *   这样，**每组（共 `G` 组）查询头都与同一组 `K` 和 `V` 进行计算**，实现了键值头的共享。

3.  **合并输出（消除G）**：
    *   计算完成后，将 `G` 和 `K` 维度**合并**回 `H`，得到标准的多头输出格式。

**最终，整个过程的维度变换清晰地围绕 `G` 展开：引入 `G` -> 在 `G` 维度上广播 -> 消除 `G`。** 这个 `G` 维度是实现参数共享和效率提升的关键。

## 2. 6 * num parameters * num tokens 解释

### 第1步：前向传播的 FLOPs ≈ 2 × 模型参数量 × 训练 tokens 数量

对于一个包含 `N` 个参数的模型，处理一个 token 的前向传播过程，其计算量主要来自于矩阵乘法。

*   **核心操作**: 对于线性层 `Y = XW + B`，其中 `W` 是权重矩阵。计算 `XW` 的 FLOPs 大约是 `2 × 输入维度 × 输出维度`。这里的 `2` 来源于：每个权重参数需要进行一次乘法和一次加法运算。
*   **简化估算**: 如果我们忽略偏置项 `B` 和激活函数等次要操作，可以近似认为，**模型每处理一个 token，前向传播的 FLOPs 约为 `2N`**，其中 `N` 是模型总参数量。
*   **扩展到整个训练集**: 如果我们在整个训练过程中处理了 `T` 个 tokens（即 `T` 是训练步数 × 批次大小 × 序列长度），那么总的前向传播 FLOPs 就是：
    `前向 FLOPs ≈ 2 × N × T`

### 第2步：反向传播的 FLOPs ≈ 前向传播的 2 倍

反向传播是根据损失函数的梯度来更新模型权重的过程。根据自动微分原理，计算梯度的 FLOPs 通常比前向传播要多。

*   **经验法则**: 在计算上，**一次反向传播的 FLOPs 大约是一次前向传播 FLOPs 的 2 倍**。
*   **原因**: 反向传播需要计算损失对每个参数的梯度，这涉及到重新计算前向传播的中间结果以及链式法则的广泛应用，其计算复杂度与前向传播同阶，但常数因子更大。`2 倍` 是一个被广泛验证和接受的经验值。
*   **因此，总的反向传播 FLOPs 约为**：
    `反向 FLOPs ≈ 2 × (前向 FLOPs) ≈ 2 × (2 × N × T) ≈ 4 × N × T`

### 第3步：总 FLOPs = 前向 + 反向

将前向传播和反向传播的 FLOPs 相加，就得到了训练模型所需的总计算量：

`总训练 FLOPs ≈ 前向 FLOPs + 反向 FLOPs`
`总训练 FLOPs ≈ (2 × N × T) + (4 × N × T)`
`总训练 FLOPs ≈ 6 × N × T`

**总结：`6 * N * T` 公式是一个源于 OpenAI GPT-3 论文的、用于快速估算 LLM 训练总计算量的强大经验法则，它将训练过程简化为前向传播（2N）和反向传播（4N）两个主要部分。**

## 3. review_of_arithmetic_intensity

### 要点总结

#### 1. **核心概念：算术强度**
- **定义**：算术强度 = 总浮点运算次数 / 总内存读写字节数。它衡量了“计算密度”。
- **意义**：算术强度越高，意味着每从内存中读取1字节数据，就能进行更多的计算，硬件计算单元就越不容易“闲置”，效率越高。

#### 2. **矩阵乘法的“记账”分析**
代码对 `(B x D) @ (D x F)` 的矩阵乘法进行了详细的FLOPs和内存访问分析：
- **FLOPs**：`2 * B * D * F`。（因为每个输出元素需要D次乘加运算，乘加算2次操作）
- **内存访问**：
    - 读取输入矩阵 `X`：`2 * B * D` 字节（假设FP16，占2字节）。
    - 读取权重矩阵 `W`：`2 * D * F` 字节。
    - 写入输出矩阵 `Y`：`2 * B * F` 字节。
- **总内存访问字节数**：`2*B*D + 2*D*F + 2*B*F`。

#### 3. **关键结论：计算受限的判断条件**
- 代码推导出，当批量大小 `B` 远小于隐藏层维度 `D` 和 `F` 时，该运算的算术强度近似等于 `B`。
- **H100加速器的强度**：其峰值算力（989 TFLOPS）除以峰值内存带宽（3.35 TB/s），得到其强度约为 **295 FLOPs/byte**。这是硬件能力的上限。
- **判断标准**：
    - 如果 **计算强度 > 硬件强度**，则是 **计算受限**。计算单元满负荷运转，这是高效状态。
    - 如果 **计算强度 < 硬件强度**，则是 **内存受限**。计算单元等待数据从内存中读取，这是低效状态。
- **因此，对于此运算，要达到计算受限（高效），需要满足**：`B > 295`。

#### 4. **最重要洞见：LLM推理（生成）的瓶颈**
- **极端情况（B=1）**：当批量大小为1时，这正对应了LLM逐个生成token的**推理场景**。
- 此时，**算术强度 ≈ 1**，远小于H100的硬件强度（295）。
- **结论**：LLM的自回归生成过程本质上是**严重内存受限**的。系统大部分时间不是在计算，而是在从内存中读取巨大的权重矩阵 `W`（对于大模型，`D` 和 `F` 都很大）。
- 这就解释了为什么诸如 **KV Cache、量化、模型压缩** 等技术对于提升推理速度至关重要——因为它们的目标都是**减少需要访问的内存量**，从而缓解内存带宽瓶颈。

### 总结

这张图片精辟地指出，**LLM的训练（通常B很大）可能是计算受限的，但LLM的推理/生成（B=1或很小）几乎总是内存受限的。** 提升推理性能的关键不在于追求更高的峰值算力，而在于优化内存访问模式、减少数据搬运量。这是理解现代LLM推理优化技术的基石。

## 3. intensity = intensity.subs(D, c*B).subs(F, c*B).limit(c, oo).simplify()

### 逐步解释

#### 第1步：初始的算术强度公式
首先，我们有一个精确的算术强度公式：
`强度 = (2 * B * D * F) / (2*B*D + 2*D*F + 2*B*F)`

这个公式看起来很复杂，我们想知道在什么情况下它可以被简化。

#### 第2步：引入假设并进行代换（Substitution）
代码中的假设是：**`D` 和 `F` 的维度远大于 `B`**（即 `D >> B` 且 `F >> B`）。这是LLM中非常常见的场景，因为隐藏维度（`D`, `F`）通常是数千，而批大小（`B`）可能是1或几十。

为了数学上严谨地表达“远大于”，代码引入了**一个趋向于无穷大的常数 `c`**。
*   `.subs(D, c*B)` ： 将公式中的所有 `D` 替换为 `c * B`。
*   `.subs(F, c*B)` ： 将公式中的所有 `F` 替换为 `c * B`。

这里做了一个更强的假设，即 `D` 和 `F` 不仅远大于 `B`，而且它们本身是**同阶无穷大**（都与同一个 `c` 成正比）。这简化了推导，但并不影响最终结论的本质。

**代入后的公式变为：**
```
强度 = (2 * B * (cB) * (cB)) / (2*B*(cB) + 2*(cB)*(cB) + 2*B*(cB))
     = (2 * B * cB * cB) / (2*B*cB + 2*cB*cB + 2*B*cB)
     = (2 * c² * B³) / (2cB² + 2c²B² + 2cB²)
     = (2 c² B³) / (2c²B² + 4cB²) // 合并了分母中的前两项和后一项
```

#### 第3步：取极限（Limit）
现在，我们执行 `.limit(c, oo)`，即让 `c` 趋向于无穷大（`oo`）。这一步的目的是**找出当 `D` 和 `F` 无限大于 `B` 时，强度的极限值**。

对于极限 `c -> oo`，我们关注分子和分母中 `c` 的最高次幂：
*   **分子**： `2 c² B³` （`c` 的阶数是 2）
*   **分母**： `2c²B² + 4cB²` （`c` 的阶数分别是 2 和 1）

当 `c` 非常大时，分母中阶数最高的项 `2c²B²` 会占据绝对主导地位，阶数较低的 `4cB²` 项可以忽略不计。

因此，强度的极限为：
```
强度 ≈ (2 c² B³) / (2c² B²) = B
```

`.simplify()` 操作会执行这个极限计算，并得到最终简化结果 `B`。

### 结论与直观解释

这行代码的最终结论是：
**当一个全连接层/矩阵乘法的权重维度（`D`, `F`）远大于批处理大小（`B`）时，其算术强度近似等于批处理大小 `B`。**

**这意味着什么？**
这对于理解LLM推理瓶颈至关重要：
1.  **推理时（`B = 1`）**： 算术强度 ≈ 1。这是一个非常低的值，远低于现代GPU的硬件强度（例如H100约为295）。这证明了**LLM的单令牌生成是极度内存受限的**，瓶颈在于从显存中读取权重数据，而非计算本身。
2.  **训练时（`B` 很大）**： 通过增大批大小，可以显著提高算术强度，使计算过程转变为**计算受限**，从而能够充分利用GPU的强大算力。


## 4. arithmetic_intensity_of_inference

![Naive inference](https://jax-ml.github.io/scaling-book/assets/img/naive-inference-1400.webp)

![KV cache](https://jax-ml.github.io/scaling-book/assets/img/cached-inference-1400.webp)

```python
text("### MLP layers (only looking at the matrix multiplications)")
flops = 0
bytes_transferred = 0
text("Steps:")
text("1. Read X (B x T x D) from HBM")
bytes_transferred += 2*B*T*D
text("2. Read Wup (D x F), Wgate (D x F), Wdown (F x D) from HBM")
bytes_transferred += 3 * 2*D*F
text("3. Compute U = X (B x T x D) @ Wup (D x F)")
flops += 2*B*T*D*F
text("4. Write U (B x T x F) to HBM")
bytes_transferred += 2*B*T*F
text("5. Compute G = X (B x T x F) @ Wgate (D x F)")
flops += 2*B*T*D*F
text("6. Write G (B x T x F) to HBM")
bytes_transferred += 2*B*T*F
text("7. Compute Y = GeLU(G)*U (B x T x F) @ Wdown (F x D)")
flops += 2*B*T*D*F
text("8. Write Y (B x T x D) to HBM")
bytes_transferred += 2*B*T*D

text("Let's take stock of the accounting results.")
assert flops == 6*B*T*D*F
assert bytes_transferred == 4*B*T*D + 4*B*T*F + 6*D*F

text("### Attention layers (focusing on the matrix multiplications with FlashAttention)")
flops = 0
bytes_transferred = 0
text("Steps:")
text("1. Read Q (B x T x D), K (B x S x D), V (B x S x D) from HBM")
bytes_transferred += 2*B*T*D + 2*B*S*D + 2*B*S*D
text("2. Compute A = Q (B x T x D) @ K (B x S x D)")
flops += 2*B*S*T*D
text("3. Compute Y = softmax(A) (B x S x T x K x G) @ V (B x S x K x H)")
flops += 2*B*S*T*D
text("4. Write Y (B x T x D) to HBM")
bytes_transferred += 2*B*T*D

assert flops == 4*B*S*T*D
assert bytes_transferred == 4*B*S*D + 4*B*T*D
```

### 要点总结

#### 1. **LLM推理的两个核心阶段**
*   **预填充**：处理整个输入提示。所有令牌可以并行计算，类似于训练过程。**特点：计算密集，可优化。**
*   **生成**：基于提示和已生成的内容，逐个生成新的令牌。**特点：串行操作，内存访问密集。**

#### 2. **核心分析指标：算术强度**
*   **定义**：`算术强度 = 总FLOPs / 总内存访问字节数`。
*   **意义**：衡量每次内存访问能完成多少计算。强度越高，说明计算单元越忙，效率越高。将该强度与硬件上限（如H100的强度为295 FLOPs/byte）对比，可判断任务是**计算受限**（高效）还是**内存受限**（低效）。

#### 3. **MLP层与注意力层的分析结果对比**

| 层面 | MLP层 | 注意力层 |
| :--- | :--- | :--- |
| **算术强度** | **`B * T`** （B: 批大小, T: 生成令牌数） | **`S * T / (S + T)`** （S: 上下文长度, T: 生成令牌数） |
| **预填充阶段 (T=S)** | 强度 = `B * S`。通过增大批处理大小`B`，可轻松达到计算受限。 | 强度 ≈ `S/2`。由于上下文长度`S`通常很大（如4096），强度很高，属于**计算受限**。 |
| **生成阶段 (T=1)** | 强度 = `B`。需要足够多的**并发请求**（大`B`）才能提高效率，但实践中难以保证。 | 强度 = `S/(S+1)` ≈ **1**。极低，远低于硬件能力，是严重的**内存受限**。 |
| **瓶颈根源** | 权重`W`被所有序列共享，可被高效复用。 | 每个序列有自己独立的KV Cache，必须从内存中读取，无法通过批处理优化。 |

#### 4. **核心结论与工程启示**

1.  **预填充阶段是计算受限的**：可以利用GPU的强大算力进行高效并行计算。
2.  **生成阶段是内存受限的**：主要瓶颈在于从内存中读取模型权重和KV Cache，而非计算本身。这是LLM推理延迟高、吞吐量难提升的根本原因。
3.  **优化方向**：提升推理性能（尤其是生成阶段）的关键，不在于追求更高的峰值算力，而在于**优化内存访问**。这解释了为什么以下技术至关重要：
    *   **KV Cache优化**：量化、压缩KV Cache。
    *   **模型量化**：将权重从FP16降至INT8/INT4，直接减少内存占用和带宽压力。
    *   **先进的注意力算法**：如FlashAttention，通过算子融合减少HBM访问次数。
    *   **连续批处理**：动态调度请求，尽可能提高硬件利用率，对抗生成阶段的低算术强度。

**总结**：这套分析精辟地指出，**LLM推理，特别是令牌生成阶段，本质是一场与内存带宽的赛跑，而非与计算能力的赛跑。** 理解算术强度是理解所有现代LLM推理优化技术为何有效的基石。

## 5. throughput_and_latency

```python
def compute_transformer_stats(config):  # @inspect config
    """Return symbols corresponding to various statistics of a Transformer."""
    text("The memory, throughput, and latency depends on the shape of the Transformer. "), text(" "), link("")

    text("Compute the number of parameters in the Transformer:")
    num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L
    text("To store parameters, just use bf16 (training requires fp32)")
    parameter_size = num_params * 2  # 2 for bf16
    
    text("We also don't need gradients and optimizer states since we're not training.")
    text("But we do have to store the KV cache (which are some of the activations) for each sequence (of length S):")
    text("How much we have to store per sequence:")
    kv_cache_size = S * (K*H) * L * 2 * 2  # 2 for key + value, 2 for bf16

    text("Total memory usage:")
    memory = B * kv_cache_size + parameter_size
    text("Latency is determined by memory IO (read all parameters and KV cache for each step)")
    latency = memory / memory_bandwidth
    text("Throughput is the inverse of latency, but we're generating B tokens in parallel")
    throughput = B / latency

    # Substitute
    num_params = num_params.subs(config).simplify()  # @inspect num_params
    memory = memory.subs(config).simplify()  # @inspect memory
    latency = latency.subs(config).simplify()  # @inspect latency
    throughput = throughput.subs(config).simplify()  # @inspect throughput

    return num_params, memory, latency, throughput
```

### 要点总结

#### 1. **核心权衡：延迟 vs. 吞吐量**
*   **小批次大小**：
    *   **优点**：**延迟低**。单个请求的响应速度快，用户体验好。
    *   **缺点**：**吞吐量差**。GPU计算资源利用率低，单位时间内能处理的请求总数少。
*   **大批次大小**：
    *   **优点**：**吞吐量高**。通过并行处理多个请求，显著提升GPU利用率，总处理能力高。
    *   **缺点**：**延迟高**。因为要等待凑够一个批次才能开始计算，单个请求的等待时间变长。

#### 2. **批次大小的具体影响**
*   **`B=1`**： 内存占用最小，延迟最低，但吞吐量也最低。适用于对实时性要求极高的场景。
*   **`B=64`**：  latency增加，但throughput显著提升。这是在延迟和吞吐量之间取得的一个常见平衡点。
*   **`B=256`**： 内存占用可能超出单卡容量，且**吞吐量的提升效益递减**，但延迟会进一步恶化。这体现了盲目增大批次规模的局限性。

#### 3. **关键的工程优化建议**
*   **分阶段优化**：
    *   **预填充阶段**：**使用小批次**。首个令牌的生成时间主要取决于此阶段，小批次可以最大化速度，优化**首令牌时间**。
    *   **生成阶段**：**使用大批次**。后续令牌的生成是内存瓶颈，通过大批次处理可以摊薄内存访问开销，极大提升**吞吐量**。现代推理引擎（如vLLM）通过**连续批处理** 技术动态实现这一点。
*   **并行化策略**：
    *   **简单并行**：启动多个模型副本。**延迟不变，吞吐量线性增长**。这是提升吞吐量最直接的方式。
    *   **模型分片**：将模型和KV Cache分布到多个设备上。这是更复杂但能突破单机算力与显存限制的高级技术。

### 总结

这张图片精辟地总结道：**LLM推理服务的优化，本质上是在延迟和吞吐量之间进行精心权衡的艺术。** 成功的优化策略不是寻找一个“万能”的批次大小，而是根据推理管道的不同阶段（预填充快，生成吞吐高）和业务目标（重实时还是重成本），动态调整资源分配。这为理解和设计高性能LLM推理服务提供了核心框架。