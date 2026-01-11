本文主要整理triton-flash-attention的主要内容。

## 7 - Gradient of MatMul

### 内容概括

这两张图共同完成了对神经网络基本操作 **\( Y = XW \)** 的梯度推导。这是一个从**具体数值示例**到**抽象公式推导**的完整过程：
*   **第一张图（图1）**：从**具体维度**和**一个行向量（N=1）的特例**出发，定义了问题。它展示了当 \( X \) 是一个 \( 1 \times 3 \) 的行向量，\( W \) 是一个 \( 3 \times 4 \) 的矩阵时，输出 \( Y \) 如何计算，并引入了损失函数 \( \phi \) 对 \( X \) 和 \( W \) 的梯度目标。
*   **第二张图（图2）**：进行了**通用化的数学推导**。它从标量链式法则出发，将其推广到矩阵形式，最终得出了适用于任意维度 \( X \)（形状 `[N, D]`）和 \( W \)（形状 `[D, M]`）的梯度计算公式。

整个过程严格遵循了**反向传播**的核心思想：已知上游梯度 \( \frac{\partial \phi}{\partial Y} \)，利用局部导数（雅可比矩阵）来求解对参数 \( X \) 和 \( W \) 的梯度。

### 详细要点总结

#### **图1：问题定义与具体展开**

1.  **操作定义**：
    *   前向传播公式：\( Y = XW \)
    *   输入：\( X \)，维度为 `[N, D]`，图中示例 `N=1, D=3`（一个行向量）。
    *   参数：\( W \)，维度为 `[D, M]`，图中示例 `M=4`。
    *   输出：\( Y \)，维度为 `[N, M]`，图中为 `[1, 4]` 的行向量。

2.  **前向计算展开**：
    *   详细写出了 \( Y \) 中每个元素的计算过程，即行向量 \( X \) 与矩阵 \( W \) 的乘法的结果：
    \[
    Y = \begin{bmatrix}
    x_{11}w_{11}+x_{12}w_{21}+x_{13}w_{31}, &
    x_{11}w_{12}+x_{12}w_{22}+x_{13}w_{32}, &
    \dots
    \end{bmatrix}
    \]
    *   **作用**：通过展开，可以清晰地看到每个输出 \( y_{1j} \) 是如何依赖于所有输入 \( x_{1k} \) 和权重 \( w_{kj} \) 的。

3.  **梯度目标**：
    *   明确了反向传播的目标是计算损失函数 \( \phi \) 对输入 \( X \) 的梯度 \( \frac{\partial \phi}{\partial X} \) 和对参数 \( W \) 的梯度 \( \frac{\partial \phi}{\partial W} \)。
    *   引入了关键的**上游梯度**：\( \frac{\partial \phi}{\partial Y} \)，其维度与 \( Y \) 相同 (`[N, M]`)。

4.  **应用链式法则**：
    *   指出了计算路径：\( \frac{\partial \phi}{\partial X} = \frac{\partial \phi}{\partial Y} \cdot \frac{\partial Y}{\partial X} \)。这里 \( \frac{\partial Y}{\partial X} \) 是一个四阶张量（雅可比矩阵），直接计算非常复杂。这张图停在这里，引出了核心问题。

#### **图2：数学推导与通用公式**

1.  **从标量到矩阵的链式法则**：
    *   对于一个标量损失 \( \phi \) 和矩阵运算 \( Y = XW \)，链式法则的**正确矩阵形式**为（根据分子布局）：
    \[
    \frac{\partial \phi}{\partial X_{ij}} = \sum_{k,l} \frac{\partial \phi}{\partial Y_{kl}} \frac{\partial Y_{kl}}{\partial X_{ij}}
    \]

2.  **推导 \( \frac{\partial \phi}{\partial X} \)**：
    *   通过代入 \( Y_{kl} = \sum_p X_{kp}W_{pl} \)，可以求出 \( \frac{\partial Y_{kl}}{\partial X_{ij}} = \delta_{ki} W_{jl} \)（当 \( p=j \) 时）。
    *   将上述结果代入链式法则求和公式，经过化简（求和号的消除），**得到简洁的矩阵乘法形式**：
    \[
    \boxed{\frac{\partial \phi}{\partial X} = \frac{\partial \phi}{\partial Y} \cdot W^T}
    \]
    *   **维度验证**：`[N, M]` 的 \( \frac{\partial \phi}{\partial Y} \) 乘以 `[M, D]` 的 \( W^T \)，结果维度为 `[N, D]`，与 \( X \) 的维度一致。

3.  **推导 \( \frac{\partial \phi}{\partial W} \)**：
    *   运用相同的逻辑，计算 \( \frac{\partial Y_{kl}}{\partial W_{ij}} = X_{ki} \delta_{jl} \)。
    *   代入链式法则公式并化简后，得到：
    \[
    \boxed{\frac{\partial \phi}{\partial W} = X^T \cdot \frac{\partial \phi}{\partial Y}}
    \]
    *   **维度验证**：`[D, N]` 的 \( X^T \) 乘以 `[N, M]` 的 \( \frac{\partial \phi}{\partial Y} \)，结果维度为 `[D, M]`，与 \( W \) 的维度一致。

## 8 - Gradient of Softmax

### 内容概括

这两张笔记构成了一个连续的数学推导过程：
*   **第一张图（图1）**：**场景设定与基础推导**。将问题置于注意力机制的框架下（`S=QK^T`, `P=Softmax(S)`），聚焦于一行 `S_i` 及其对应的 `P_i`，严格使用**商法则**推导出Softmax梯度中**对角线元素**和**非对角线元素**的两种不同情况。
*   **第二张图（图2）**：**总结与模式探索**。在基础推导上，试图用更紧凑的符号（如 `P_k`, `P_{-k}`）总结梯度公式，并探索其**雅可比矩阵**的对称性等性质，为进一步分析做铺垫。

整个过程的目标是透彻理解标准Softmax梯度 `∂P/∂S` 的计算方式，这为后续理解FlashAttention为何要优化这一步骤提供了坚实基础。

### 详细要点总结

#### **图1：Softmax梯度推导（标准方法）**

1.  **问题定义于注意力机制**：
    *   定义了核心变量：`S = QK^T` (注意力分数)，`P = softmax(S)` (注意力权重)，`O = PV` (输出)。
    *   聚焦于**第i行**：将问题简化为研究一个向量 `S_i`（`S`的第i行）经过softmax得到 `P_i` 的梯度 `∂P_i/∂S_i`。这是一个 `N x N` 的雅可比矩阵（`N`为序列长度）。

2.  **应用商法则推导**：
    *   从softmax的标量形式出发：`P_ik = e^{S_ik} / ∑_j e^{S_ij}`。
    *   使用商法则 `(f/g)’ = (f’g - fg’)/g²` 进行求导，其中 `f = e^{S_ik}`, `g = ∑_j e^{S_ij}`。

3.  **得出两种情况的梯度公式**：
    *   **情况一：当对 `S_i` 的第k个元素求导，且该元素是 `P_ik` 的自变量时（即 `j = k`）**。
        *   推导结果：`∂P_ik/∂S_ik = P_ik * (1 - P_ik)`。
        *   **意义**：这是对角线上的元素，表示**某个分数对其自身对应概率的影响**。
    *   **情况二：当对 `S_i` 的第k个元素求导，但计算的是对其他概率 `P_ij` (j ≠ k) 的偏导时**。
        *   推导结果：`∂P_ij/∂S_ik = -P_ij * P_ik`。
        *   **意义**：这是非对角线上的元素，表示**提高某个分数的权重，会以线性比例降低其他所有分数的权重**（概率之和为1的约束所致）。

#### **图2：梯度公式的总结与性质探索**

1.  **尝试统一表示**：
    *   使用 `P_k` 代表 `P_ik`，并可能用 `P_{-k}` 代表其他概率的和，试图将两种情况总结成更紧凑的形式。
    *   笔记中反复书写 `P_k (1-P_k)` 和 `-P_k P_j`，意在强化记忆这两种核心模式。

2.  **探索雅可比矩阵的性质**：
    *   提到了 **“symmetric”**（对称）。这里需要澄清：标准Softmax的雅可比矩阵 `∂P_i/∂S_i` 是**对称的**吗？**通常不是对称矩阵**，因为 `∂P_ij/∂S_ik = -P_ij P_ik`，而 `∂P_ik/∂S_ij = -P_ik P_ij`，两者相等，这说明该雅可比矩阵是**对称矩阵**（`J_{jk} = J_{kj}`）。这是一个重要的数学性质。
    *   提到了 **“d log(p)”**。这指向了另一个重要结论：**Softmax的梯度可以通过其输出 `P` 巧妙地表示**。综合图1的结论，整个 `N x N` 的雅可比矩阵可以写成：
      \[
      \frac{\partial P_i}{\partial S_i} = \text{diag}(P_i) - P_i P_i^T
      \]
      *   `diag(P_i)` 是一个对角线为 `P_i` 元素的对角矩阵，对应 `j=k` 的情况（`P_ik*(1-P_ik)` 展开后的一部分）。
      *   `- P_i P_i^T` 是向量 `P_i` 与其自身的外积，取负，对应所有 `j≠k` 的情况。

### 核心联系：为何这是FlashAttention优化的起点

这两页笔记推导出的 **`∂P/∂S = diag(P) - P P^T`** 公式，正是理解FlashAttention核心优化动机的钥匙：

1.  **标准方法的计算代价**：
    *   在反向传播中，为了计算损失 `L` 对 `S` 的梯度 `dS`，我们需要计算 `dS = (diag(P) - P P^T) dO`。
    *   这个公式涉及一个 **`N x N` 的矩阵 `P P^T`（即P的自身外积）**。当序列长度 `N` 很大时（如2048， 4096），这个矩阵是**极其庞大**的（数百万甚至上亿个元素），计算和存储它的代价非常高。

2.  **FlashAttention的优化精髓**：
    *   您之前提到的FlashAttention关键公式 **`dS_ij = P_ij ⊙ (dP_ij - D_i)`**，正是对上述标准公式 `dS = (diag(P) - P P^T) dO` 的**数学等价变换和工程优化**。
    *   **`D_i = rowsum(P_i ⊙ dP_i)`** 这个预计算的标量，巧妙地捕获了 `P P^T` 中所需的行汇总信息，从而避免了显式构造和计算整个 `N x N` 的外积矩阵，将计算复杂度从矩阵运算降为逐元素运算。

## 9.0 - _attn_bwd_preprocess源码分析

这是一个 **FlashAttention 反向传播的预处理步骤内核**，它负责计算一个关键的中间变量 `D`，为后续计算 `dQ`, `dK`, `dV` 做准备。下面是对这段代码的详细分析。

### 函数功能概览

这个 `_attn_bwd_preprocess` 函数的核心任务是：**计算并存储向量 `D`，其每个元素 `D_i` 是注意力输出 `O` 的对应行与上游梯度 `dO` 的对应行的点积结果** 。

用公式表示就是：
\[
D_i = \sum_{j} dO_{ij} \cdot O_{ij}
\]

这里的 `D` 是 FlashAttention 反向传播公式 `dS_ij = P_ij ⊙ (dP_ij - D_i)` 中的关键组成部分，用于高效计算注意力分数 `S` 的梯度 。

### 代码逐行解析

#### 1. 参数定义
```python
def _attn_bwd_preprocess(
    O,      # 注意力层的输出张量 [batch*heads, seq_len, head_dim]
    dO,     # 损失函数对 O 的梯度，维度与 O 相同
    D,      # 输出的中间变量 [batch*heads, seq_len]
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,  # 编译时常量，Q 的分块大小
    HEAD_DIM: tl.constexpr,       # 编译时常量，每个头的维度
):
```
- **`O` 和 `dO`** 是三维张量，但在这里被展平为二维视角 `[batch*heads, seq_len, head_dim]`
- **`D`** 是二维输出 `[batch*heads, seq_len]`

#### 2. 并行化策略
```python
block_index_q = tl.program_id(0)      # 在序列长度维度并行
index_batch_head = tl.program_id(1)   # 在batch和head维度并行
```
这是 FlashAttention-V2 的核心优化之一：**在序列长度维度增加并行化**。当处理长序列（batch size 较小时），这种并行策略能显著提高 GPU 利用率 。

#### 3. 计算偏移量
```python
offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
offs_dim = tl.arange(0, HEAD_DIM)
```
- `offs_q`：当前块在序列维度上的偏移量
- `offs_dim`：在特征维度上的偏移量

#### 4. 分块加载数据
```python
O_block = tl.load(
    O + index_batch_head * HEAD_DIM * SEQ_LEN
    + offs_q[:, None] * HEAD_DIM
    + offs_dim[None, :]
)

dO_block = tl.load(
    dO + index_batch_head * HEAD_DIM * SEQ_LEN  
    + offs_q[:, None] * HEAD_DIM
    + offs_dim[None, :]
).to(tl.float32)  # 转换为float32保证计算精度
```
这里使用了 **Triton 的块指针运算**，高效地从全局内存加载数据块到共享内存 。

#### 5. 核心计算
```python
D_block = tl.sum(dO_block * O_block, axis=1)  # Shape: (BLOCK_SIZE_Q,)
```
这是最关键的一步：
- **逐元素相乘**：`dO_block * O_block`
- **沿特征维度求和**：`tl.sum(..., axis=1)`
- 结果 `D_block` 包含 `BLOCK_SIZE_Q` 个标量值，每个对应一个查询位置的 `D_i`

#### 6. 结果存储
```python
D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
tl.store(D_block_ptrs, D_block)
```
将计算好的 `D` 块写回全局内存，供后续反向传播使用。

## 9.1 - _attn_bwd_dq源码分析

### 函数功能概览

这个 `_attn_bwd_dq` 函数负责计算**损失函数对查询矩阵 Q 的梯度**。它采用分块策略，通过重新计算注意力权重并应用FlashAttention的高效梯度公式，避免了存储庞大的中间矩阵 `S` 和 `P`。

### 代码逐行解析

#### 1. 并行化与内存布局
```python
index_batch_head = tl.program_id(2)
index_batch = index_batch_head // NUM_HEADS
index_head = index_batch_head % NUM_HEADS
```
- **并行策略**：在**批次（batch）和注意力头（head）维度**并行，每个程序实例处理一个特定的`(batch, head)`组合
- **内存访问**：通过`offset_batch_head`计算精确的内存偏移，确保每个实例访问正确的数据区域

#### 2. 张量指针初始化
```python
Q += offset_batch_head
K += offset_batch_head
# ... 类似的指针调整
```
- **指针定位**：将所有输入/输出张量指针调整到当前处理的`(batch, head)`对应的内存位置
- **内存连续性**：假设数据在内存中是连续存储的，这是高效指针运算的前提

#### 3. 分块加载数据
```python
start_q = index_block_kv * BLOCK_Q
offs_q = start_q + tl.arange(0, BLOCK_Q)

Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
```
- **Q块加载**：加载当前要处理的Q块到SRAM
- **梯度加载**：同步加载对应的上游梯度dO块
- **分块思想**：将大的矩阵运算分解为小块，适合在高速缓存中计算

#### 4. 核心计算循环
```python
for blk_idx in range(num_steps):
    K_T_block = tl.load(kT_ptrs)
    V_T_block = tl.load(vT_ptrs)
    # ... 计算过程
```
循环处理整个K/V序列，每次处理一个K/V块：

**a. 重计算注意力权重**
```python
QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
P_block = tl.math.exp(QK_block - M_block)
```
- **在线重计算**：重新计算注意力分数S和权重P，避免存储中间矩阵
- **数值稳定性**：使用`M_block`（运行最大值）确保指数计算不会溢出

**b. 因果掩码处理**
```python
if STAGE == 3:
    mask_block = offs_q[:, None] >= offs_kv[None, :]
    P_block = tl.where(mask_block, P_block, 0.0)
```
- **自回归注意力**：确保每个位置只能关注当前位置及之前的信息
- **掩码应用**：将"未来"位置的注意力权重强制设为0

**c. FlashAttention梯度计算**
```python
dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
dS_block = P_block * (dP_block - Di[:, None])
```
这是**最关键的优化公式**：
- `dP_block`：标准注意力梯度的一部分
- `Di = tl.load(D + offs_q)`：来自预处理步骤的校正项，等于`rowsum(P ⊙ dP)`
- `dS_block`：应用FlashAttention的高效梯度公式，避免了显式计算外积矩阵

**d. dQ累加更新**
```python
dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
```
- **梯度累加**：将每个K/V块计算的梯度贡献累加到dQ块中
- **反缩放**：乘以`softmax_scale`来抵消前向传播中的缩放效应

#### 5. 结果写回
```python
dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
tl.store(dQ_block_ptrs, dQ_block)
```
- **存储优化**：只有当所有K/V块处理完成后，才将最终的dQ块写回全局内存
- **内存效率**：最大限度减少全局内存访问次数

## 9.2 - _attn_bwd_dk_dv源码分析

这是一个 **FlashAttention 反向传播中计算 dK（键梯度）和 dV（值梯度）的高效 Triton 内核实现**。它展示了如何通过分块计算、在线重计算和特定的并行化策略，在不存储庞大中间矩阵的情况下，正确且高效地计算出梯度。

以下是详细分析，我将通过一个表格总结其核心工作流程，然后解析关键代码段落：

| 步骤 | 操作 | 目的 | 实现要点 |
| :--- | :--- | :--- | :--- |
| **1. 初始化与指针设置** | 定位到当前处理的批次和注意力头对应的数据块。 | 确保每个并行实例处理正确的数据分区。 | 通过 `index_batch_head` 计算偏移量，调整 Q, K, V, dO, dQ, dK, dV, M, D 等所有输入输出张量的指针。 |
| **2. 加载当前K/V块** | 将全局内存中的 K 和 V 的一个块加载到共享内存。 | 当前K/V块将在后续内循环中保持驻留，被重复使用。 | 使用 `tl.load` 加载 `K_block` 和 `V_block`。这些数据在内循环中保持不变。 |
| **3. 循环处理Q序列** | 遍历Q序列的每个块。 | 累积当前K/V块对所有Q块的梯度贡献。 | 循环次数为 `num_steps = SEQ_LEN // BLOCK_Q`。在每次迭代中加载一个新的 `qT_block`（Q块的转置）和对应的 `dO_block`。 |
| **4. 在线重计算P^T** | 使用当前K块和Q块重新计算注意力权重矩阵的转置 P^T。 | 避免存储庞大的中间注意力矩阵 P (N×N)。 | 计算 `QK_T_block = softmax_scale * tl.dot(K_block, qT_block)`，然后应用指数和最大值修正 (`tl.math.exp(QK_T_block - m[None, :])`)。这是FlashAttention的核心思想之一。 |
| **5. 应用因果掩码（如需要）** | 在自回归（STAGE==3）时，将“未来”位置的注意力权重置零。 | 确保解码时每个位置只能关注自身及之前的信息。 | 使用 `tl.where(mask_block, P_T_block, 0.0)` 实现，其中 `mask_block` 由 `offs_q` 和 `offs_kv` 的位置关系生成。 |
| **6. 计算dV梯度** | 累加当前块对dV的贡献：`dV_block += tl.dot(P_T_block, dO_block)`。 | 这是标准注意力梯度公式 `dV = P^T @ dO` 的分块实现。 | 由于K/V块是固定的，内循环遍历Q块，通过矩阵乘法累加得到当前K/V块对应的dV部分。 |
| **7. 计算dK梯度** | 1. 计算中间梯度 `dP^T = V_block @ dO_block^T`。<br>2. 计算注意力分数梯度 `dS^T = P_T_block * (dP^T - Di)`。<br>3. 累加对dK的贡献：`dK_block += softmax_scale * tl.dot(dS_T_block, qT_block)`。 | 这是FlashAttention最关键的优化公式`dS_ij = P_ij ⊙ (dP_ij - D_i)`的矩阵转置形式，避免了显式计算外积。 | 公式中的 `Di` 是预处理阶段计算好的标量校正项 `D_i = rowsum(O_i ⊙ dO_i)`，它使得计算简化为逐元素操作。 |
| **8. 写回结果** | 将计算好的 dK_block 和 dV_block 写回全局内存。 | 完成当前K/V块的梯度计算。 | 循环结束后，每个K/V块对应的 dK 和 dV 的梯度部分已计算完成，写入对应的全局内存位置。 |

### 关键代码段落解析

1.  **并行化策略**
    ```python
    index_batch_head = tl.program_id(2)
    index_block_kv = tl.program_id(0)
    ```
    *   内核在三个维度上并行：`program_id(0)` 在**K/V的序列维度**上进行分块并行，这是FlashAttention-V2的关键优化，增加了长序列下的并行度。`program_id(2)` 在**批次大小（batch）和注意力头（head）** 维度并行。这种策略确保了即使batch size很小，也能有足够的并行块来充分利用GPU。

2.  **在线重计算与内存优化**
    ```python
    # 在内循环中重新计算P^T
    QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
    P_T_block = tl.math.exp(QK_T_block - m[None, :])
    ```
    *   内核**没有存储**前向传播中计算的庞大 `(SEQ_LEN, SEQ_LEN)` 注意力矩阵 `P`。而是在反向传播需要时，利用当前驻留在SRAM中的K块和Q块**重新计算**所需的 `P_T_block`（一个 `BLOCK_KV x BLOCK_Q` 的小块）。这是FlashAttention实现**O(N)内存复杂度**（而非标准注意力的O(N²)）的基石。

3.  **核心梯度计算公式**
    ```python
    # FlashAttention的精髓公式
    dS_T_block = P_T_block * (dpT_block - Di[None, :])
    ```
    *   这行代码是优化后的Softmax梯度计算。它等价于标准公式 `dS = (diag(P) - PP^T) dP`，但通过引入预计算的标量 `D_i`（来自预处理内核），将复杂的矩阵运算简化为**高效的逐元素操作**（`dS_ij = P_ij * (dP_ij - D_i)`）。这非常适合GPU的并行架构，并且避免了显式构造和存储 `PP^T` 这个巨大的中间矩阵。

## 10 - triton.autotune

这段代码是 **Triton 自动调优系统的核心配置**，它通过预定义一系列参数组合，让 FlashAttention 内核能在不同的硬件和输入尺寸下自动选择最优配置，从而最大化计算效率。

### 🔧 自动调优机制详解

`@triton.autotune` 装饰器的工作原理是：当内核函数被调用时，它会根据 `key` 中指定的参数（这里是 `SEQ_LEN` 和 `HEAD_DIM`），在提供的配置列表 (`configs`) 中进行测试和比较，最终选择并缓存性能最佳的那个配置供后续使用。

你所提供的代码使用列表推导式生成了一个包含多种参数的配置组合：

```python
[
    triton.Config(
        {
            "BLOCK_SIZE_Q": BLOCK_SIZE_Q,   # Q的块大小
            "BLOCK_SIZE_KV": BLOCK_SIZE_KV  # K/V的块大小
        },
        num_stages=num_stages,  # 计算流水线阶段数
        num_warps=num_warps,    # 每个线程块的warp数量
    )
    # 以下是生成所有组合的循环
    for BLOCK_SIZE_Q in [64, 128]      # 2种选择
    for BLOCK_SIZE_KV in [32, 64]     # 2种选择 → 2x2=4种块组合
    for num_stages in [3, 4, 7]       # 3种选择 → 4x3=12种
    for num_warps in [2, 4]           # 2种选择 → 12x2=24种配置
]
```

这个推导式最终会生成 **2 × 2 × 3 × 2 = 24 种** 不同的配置供自动调优系统测试和筛选。

### ⚙️ 核心参数的作用与影响

下表详细解释了这些参数如何影响内核的性能表现：

| 参数 | 定义与作用 | 取值影响分析 |
| :--- | :--- | :--- |
| **`BLOCK_SIZE_Q`** | **查询（Query）矩阵的分块大小**<br>决定每次处理多少行Q向量 | `64`：更适合短序列或共享内存受限场景<br>`128`：更适合长序列，提升计算吞吐量 |
| **`BLOCK_SIZE_KV`** | **键值（Key/Value）矩阵的分块大小**<br>控制K/V矩阵的加载粒度 | `32`：内存访问更精细，适合小规模数据<br>`64`：减少内存事务数，提升带宽利用率 |
| **`num_stages`** | **流水线阶段深度**<br>控制指令级并行和内存操作的重叠程度 | `3`/`4`：平衡派发延迟和资源占用<br>`7`：更深的流水线，可能提升峰值性能但增加寄存器压力 |
| **`num_warps`** | **每个线程块的warp数量**<br>决定线程级并行规模 | `2`（64线程）：适合内存密集型任务<br>`4`（128线程）：提升计算并行度，适合计算密集型任务 |
