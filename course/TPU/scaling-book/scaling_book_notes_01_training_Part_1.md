本文主要整理Part 5 Traning (How to Parallelize a Transformer for Training) 的主要内容。

Here we discuss four main parallelism schemes used during LLM training: **data parallelism, fully-sharded data parallelism (FSDP), tensor parallelism, and pipeline parallelism**. For each, we calculate at what point we become bottlenecked by communication.

## 1. What Do We Mean By Scaling?

### 要点总结

**四种并行方案**：
    *   **数据并行**：最基础方案。每个设备都有完整的模型副本，但处理不同的数据批次。通信仅发生在反向传播时同步梯度。
    *   **全分片数据并行**：数据并行的增强版。将模型参数、梯度、优化器状态都进行分片，极大节省了单个设备的内存。但需要在计算前即时收集参数，引入了额外的通信。
    *   **张量并行**：将模型内部的巨大权重矩阵进行拆分，分布到不同设备上计算。这要求在前向和反向传播中进行特定的通信操作（如AllGather, ReduceScatter）来聚合结果。
    *   **流水线并行**：将模型按层分割成多个阶段，每个阶段在不同的设备上。数据以流水线方式依次通过各个阶段。主要瓶颈是“流水线气泡”（设备等待时间）。

---

### 公式解释

![A Transformer Layer](https://jax-ml.github.io/scaling-book/assets/img/transformer-layer.png)

#### 符号说明（结合第2张图）

*   **基本维度**：
    *   $B$: 总批次大小（token数量）
    *   $D$: 模型隐藏层维度
    *   $F$: 前馈网络层维度
    *   $L$: 模型层数
*   **网格轴**：$X$, $Y$, $Z$ 代表设备网格的不同维度。下标 $X$, $Y$, $Z$ 表示张量沿该维度被分片。例如：
    *   $B_X$ 表示批次大小 $B$ 在 $X$ 轴方向的设备上被分片。
    *   $D_Y$ 表示隐藏维度 $D$ 在 $Y$ 轴方向的设备上被分片。
*   **标记法格式**：$张量名称 ⌊ 维度1, 维度2, ... ⌋$
    *   例如：$Win ⌊ D, F ⌋$ 表示一个名为 $Win$ 的权重矩阵，其形状为 $[D, F]$，并且**没有被分片**（因为下标没有网格轴）。
    *   例如：$In ⌊ B_X, D ⌋$ 表示输入张量 $In$，形状为 $[B, D]$，并且沿批次维度 $B$ 在 $X$ 轴设备上被分片。

#### 四种并行方案的公式解读

**1. 数据并行**
*   **公式**：$In ⌊B_X, D⌋ ·_D Win ⌊D, F⌋ ·_F Wout ⌊F, D⌋ → Out ⌊B_X, D⌋$
*   **解读**：
    *   **输入** 沿批次维度 $B$ 在 $X$ 轴设备上分片。每个设备只处理一部分数据。
    *   **权重** 完全没有分片。每个设备上都保存有完整的 $Win$ 和 $Wout$ 矩阵副本。
    *   **输出** 同样在批次维度 $B$ 上分片。
    *   **核心思想**：**分片数据，复制模型**。通信只在反向传播后同步各设备上的梯度。

**2. 全分片数据并行**
*   **公式**：$In ⌊B_X, D⌋ ·_D Win ⌊D_X, F⌋ ·_F Wout ⌊F, D_X⌋ → Out ⌊B_X, D⌋$
*   **解读**：
    *   **输入** 同样在批次维度 $B$ 上分片。
    *   **权重** 被分片了！$Win$ 沿 $D$ 维度在 $X$ 轴设备上分片，$Wout$ 沿 $F$ 维度在 $X$ 轴设备上分片。
    *   **核心思想**：**数据和模型都分片**。计算前需要通过通信操作将所有设备上属于同一层的权重分片收集起来。这大大减少了单个设备的内存占用，但增加了通信量。

**3. 张量并行**
*   **公式**：$In ⌊B, D_Y⌋ ·_D Win ⌊D, F_Y⌋ ·_F Wout ⌊F_Y, D⌋ → Out ⌊B, D_Y⌋$
*   **解读**：
    *   **输入** 沿隐藏维度 $D$ 在 $Y$ 轴设备上分片。这与数据并行完全不同。
    *   **权重** 也相应地被分片。$Win$ 沿 $F$ 维度在 $Y$ 轴设备上分片。
    *   **核心思想**：**将模型内部的单个大矩阵运算拆分开**。这要求在前向传播时通过AllGather通信整合输入，在反向传播后通过ReduceScatter通信聚合梯度。通常在一个机架内高速互联的设备上使用。

**4. 流水线并行**
*   **公式**：$In ⌊L_Z, B, D⌋ ·_D Win ⌊L_Z, D, F⌋ ·_F Wout ⌊L_Z, F, D⌋ ⌊i⌋ → Out ⌊L_Z, B, D⌋ ⌊i⌋$
*   **解读**：
    *   这个标记法稍有不同，引入了层维度 $L$ 和设备轴 $Z$。
    *   **权重** 沿层维度 $L$ 在 $Z$ 轴设备上分片。即，设备1持有第1-4层的权重，设备2持有第5-8层的权重，以此类推。
    *   **输入/输出** 的标记 $⌊i⌋$ 表示数据以“微批次”的形式在流水线的各个阶段（层）间流动。
    *   **核心思想**：**将模型按层切分到不同设备上**。通信只发生在相邻设备之间，传递的是中间激活值。

### 总结

- FSDP​ 是 “数据并行的终极形态”。它继承了数据并行的灵魂（分片数据批次），然后通过“分时复用”参数的思想，解决了数据并行的阿喀琉斯之踵——内存浪费。
- 张量并行​ 是 “模型并行的典型代表”。它通过“分工合作”的方式，将一个巨大的计算任务拆解开，解决了“单个任务太大，一块卡干不了”的问题。

通过这套标记法，我们可以一目了然地看到不同并行策略最根本的区别：**张量的哪个维度在设备网格的哪个方向上被分片**。数据并行分片批次维度$B$，张量并行分片模型维度$D$或$F$，流水线并行分片层维度$L$。这种分片策略直接决定了内存的节省程度和通信开销的模式，是理解和优化分布式训练的核心。

## 1.1 Data Parallelism

![Data Parallelism](https://jax-ml.github.io/scaling-book/assets/img/data-parallelism.png)

```Algorithm
Pure Data Parallelism Algorithm:

Forward pass: need to compute Loss[BX]

    Tmp[BX, F] = In[BX, D] *D Win[D, F]
    Out[BX, D] = Tmp[BX, F] *F Wout[F, D]
    Loss[BX] = …

Backward pass: need to compute dWout[F, D], dWin[D, F]

    dOut[BX, D] = …
    dWout[F, D] {UX} = Tmp[BX, F] *B dOut[BX, D]
    dWout[F, D] = AllReduce(dWout[F, D] {UX}) (not on critical path, can be done async)
    dTmp[BX, F] = dOut[BX, D] *D Wout[F, D]
    dWin[D, F] {UX} = In[BX, D] *B dTmp[BX, F]
    dWin[D, F] = AllReduce(dWin[D, F] {UX}) (not on critical path, can be done async)
    dIn[BX, D] = dTmp[BX, F] *F Win[D, F] (needed for previous layers)
```


### 1. 核心思想与工作原理
*   **目标**：通过增加芯片数量来增大有效的批次大小（Batch Size），从而加速训练。
*   **方法**：将整个批次数据（激活值）沿批次维度（B）分片到多个设备上（如TPU），**每个设备都持有模型的完整副本**。
*   **通信**：
    *   **前向传播**：**无通信**。每个设备用自己的数据和自己持有的完整模型权重独立计算。
    *   **反向传播**：每个设备计算得到本地梯度后，需要**同步梯度**。这是通过$AllReduce$集体通信操作完成的，确保所有设备上的模型参数更新一致。

### 2. 优势与劣势
*   **优势**：
    *   **实现友好**：反向传播中的$AllReduce$操作**不阻塞关键路径**，可以异步执行，实现和优化相对简单。
    *   **节省激活值内存**：通过分片批次，减少了每个设备需要存储的中间激活值（Intermediate Activations）的大小。
*   **劣势（致命限制）**：
    *   **无法节省模型参数内存**：每个设备都必须存储一份完整的模型参数、梯度和优化器状态的副本。这是纯数据并行最主要的**内存瓶颈**。

### 3. 关键结论：内存限制的经验法则
*   一个极其重要的实践结论是：当使用Adam优化器（参数为bf16，优化器状态为fp32）时，纯数据并行下，单个设备能训练的最大模型参数量约为该设备高带宽内存（HBM）的十分之一。**这个“十分之一”的规律，本质上是这样一个快速心算技巧：当你把内存容量数值（以GB为单位）除以10，结果的数字部分就直接代表了模型参数量（以十亿个为单位）的近似值**。
*   **举例**：对于拥有96GB HBM的TPU v5p芯片，最多能训练约**90亿（9B）参数**的模型。如果模型更大，则必须采用其他并行策略（如FSDP、张量并行）来分片模型本身。

### 4. 通信瓶颈的量化分析
*   分析表明，数据并行是否受通信瓶颈限制取决于**每设备批次大小**（$B/X$，即总批次大小除以设备数量）与一个硬件相关常数（$C / W_ici$，即计算能力与芯片间互联带宽的比值）之间的关系。
*   **计算受限的条件**：$B/X > C / W_ici$
*   **分析意义**：代入TPUv5p的参数（C=4.6e14 FLOPs, W_ici=2*9e10 B/s）后，得出每TPU的批次大小只需大于2550即可避免通信瓶颈。由于现代硬件算力强大且支持多维度并行，**在实践中，纯数据并行很难成为通信瓶颈**。它的主要瓶颈在于上述的**模型参数内存**，而非通信。 **(FLOPs/second) / (Bytes/second) = FLOPs/second * second/Bytes = FLOPs/Bytes**，这个结果 FLOPs/Bytes是一个非常重要的概念，称为 计算强度​ 或 计算通信比。它表示 “处理每字节的通信数据，需要执行多少计算量的浮点运算”。
*  Let’s put in some real numbers to get a sense of scale. For TPUv5p, **C=4.6e14 and W=2*9e10** for 1D data parallelism over ICI, so our batch size per chip must be **at least 2,550** to avoid being communication-bound. Since we can do data parallelism over multiple axes, if we dedicate all three axes of a TPUv5p pod to pure data parallelism, we 3x our bandwidth WiciWici​ and can scale down to only **BS=850** per TPU or 7.6M tokens per batch per pod (of 8960 chips)! This tells us that it’s fairly hard to become bottlenecked by pure data parallelism!

### 5. 重要概念补充
*   **上下文并行**：可以将数据并行不仅应用于“批次”维度，还可应用于“序列”维度。因为对于MLP层而言，所有token都是独立的。这种在序列长度维度上的并行也称为序列并行。
*   **多网格轴优势**：当一种并行策略（如数据并行）跨越硬件的多个网格轴（如X轴和Y轴）时，可用的聚合通信带宽会近似成倍增加，从而进一步降低通信时间。

### 总结

这组图片清晰地表明，**纯数据并行是一种强大但适用场景有限的基础并行策略**。它非常适合**模型本身能完全放入单个设备内存**的场景，可以高效地通过增加设备来扩大批次大小。然而，对于训练当今的大语言模型（LLMs），其模型参数动辄百亿、千亿，纯数据并行由于无法分片模型参数，其用处非常有限。要训练这些大模型，必须结合**全分片数据并行（FSDP）**、**张量并行（TP）** 等模型并行技术。

## 1.1 Data Parallelism 公式解释

### 公式 (1): 通信时间

**公式：**
$$ T_{\text{comms}}=\frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_{ici}} $$

**解释：**
这个公式计算的是**每层在反向传播后，同步梯度所需的时间**。

*   **$D · F$**：这是权重矩阵 $W_in$ 或 $W_out$ 的大小（参数数量）。$D$是隐藏层维度，$F$是前馈网络层维度。
*   **第一个系数 $2$**：因为参数通常以16位精度（如bf16）存储，每个参数占**2个字节**。所以，一个矩阵的字节大小是 $2 · D · F$。
*   **第二个系数 $2$**：因为一层有**两个**需要同步梯度的矩阵（$W_in$ 和 $W_out$）。
*   **第三个系数 $2$**：这是$AllReduce$操作本身的特性。在一个一维网格中，$AllReduce$ 可以看作先进行一次$Reduce-Scatter$（将各设备的梯度分片求和），再进行一次$All-Gather$（收集全部分片）。每次操作的数据量大约是整个梯度张量的大小。因此，总通信量约为 **2倍** 的单个梯度张量大小。
*   **$W_ici$**：这是**芯片间互联带宽**，单位是**字节/秒**。它是硬件性能的关键指标。

**结论：** 所以，分子 $8 · D · F$ 代表每层需要通信的**总字节数**。整个公式 $T_comms = 总字节数 / 带宽$，就是标准的计算通信时间的方法。

**关键点：** 通信时间 $T_comms$ **与批次大小 $B$ 和设备数量 $X$ 无关**。无论你用一个样本还是一百万个样本训练，需要同步的梯度大小都是固定的（只由模型结构 $D$ 和 $F$ 决定）。

---

### 公式 (2): 计算时间

**公式：**
$$ T_{\text{math}}=\frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} $$

**解释：**
这个公式计算的是**每个设备上，完成一层计算（主要是矩阵乘法）所需的时间**。

*   **$B · D · F$**：这是一个矩阵乘法（$[B/X, D] · [D, F]$）的**浮点运算次数**。一个大小为 $[M, K]$ 和 $[K, N]$ 的矩阵乘法，需要 $2 · M · K · N$ 次浮点运算（FLOPs）。在这里，$M = B/X$，$K = D$，$N = F$，所以一次矩阵乘法的FLOPs是 $2 · (B/X) · D · F$。
*   **第一个系数 $2$**：一层包含**两个**矩阵乘法（$输入 -> W_in -> W_out$）。
*   **第二个系数 $2$**：反向传播的计算量大约是前向传播的**2倍**（因为需要为输入和权重都计算梯度）。所以，一层完整的后向传播计算量约等于 **4次** 矩阵乘法。$2 * 2 = 4$，但公式中写作了 $2 · 2 · ...$ 的形式，最终分子是 $8 · B · D · F$，代表了总FLOPs。
*   **$X$**：设备数量。因为数据被分片，每个设备只处理 $B/X$ 的数据量，所以每个设备上的计算量也减少了X倍。
*   **$C$**：单个设备的**计算吞吐量**，单位是**FLOPs/秒**（每秒浮点运算次数）。

**结论：** 所以，分子 $8 · B · D · F$ 是**所有设备上的总计算量**，分母 $X · C$ 是**系统的总计算能力**。整个公式 $T_math = 总计算量 / 总算力$，就是标准的计算时间的方法。

**关键点：** 计算时间 $T_math$ **与批次大小 $B$ 成正比，与设备数量 $X$ 成反比**。增大批次大小或增加设备数量都会增加总计算量，但通过并行化，每个设备分担的计算量更少，从而可能减少计算时间。

---

### 公式 (3): 计算受限的条件

**公式：**
$$ \frac{B}{X}>\frac{C}{W_{ici}} $$

**解释：**
这个公式给出了系统从“通信受限”转变为“计算受限”的**临界条件**。

*   **推导过程**：系统是计算受限的，意味着计算时间比通信时间长（$T_math > T_comms$）。我们让公式(2) > 公式(1)：
    $$ \frac{8 \cdot B \cdot D \cdot F}{X \cdot C} > \frac{8 \cdot D \cdot F}{W_{ici}} $$
    等式两边同时除以 $8 · D · F$（它们大于0，不等号方向不变），就得到了这个简洁的条件。

*   **物理意义**：
    *   **$B/X$**：**每设备批次大小**。这是每个设备独立处理的样本数量。它代表了计算的“粒度”。
    *   **$C / W_ici$**：这是一个**硬件相关的常数**。它是设备计算能力 ($C$) 和通信带宽 ($W_ici$) 的比值。可以理解为“为了隐藏掉一次单位字节通信所需付出的计算量”。

*   **如何理解这个条件**：
    当每个设备处理的批量足够大，以至于它做这些计算所花的时间，超过了同步梯度所需的时间时，通信操作就可以被计算完全隐藏掉。此时，系统的效率由计算速度决定，我们称之为“计算受限”。反之，如果每个设备的批量很小，很快就计算完了，然后必须停下来等待所有设备同步梯度，那么系统就是“通信受限”的。

## 1.2 Fully-Sharded Data Parallelism (FSDP)

![FSDP](https://jax-ml.github.io/scaling-book/assets/img/fsdp.png)

```Algorithm
Fully-Sharded Data Parallelism (FSDP):

Forward pass: need to compute Loss[BX]

    Win[D, F] = AllGather(Win[DX, F]) (not on critical path, can do it during previous layer)
    Tmp[BX, F] = In[BX, D] *D Win[D, F] (can throw away Win[D, F] now)
    Wout[F, D] = AllGather(Wout[F, DX]) (not on critical path, can do it during previous layer)
    Out[BX, D] = Tmp[BX, F] *F Wout[F, D]
    Loss[BX] = …

Backward pass: need to compute dWout[F, DX], dWin[DX, F]

    dOut[BX, D] = …
    dWout[F, D] {UX} = Tmp[BX, F] *B dOut[BX, D]
    dWout[F, DX] = ReduceScatter(dWout[F, D] {UX}) (not on critical path, can be done async)
    Wout[F, D] = AllGather(Wout[F, DX]) (can be done ahead of time)
    dTmp[BX, F] = dOut[BX, D] *D Wout[F, D] (can throw away Wout[F, D] here)
    dWin[D,F] {UX} = dTmp[BX, F] *B In[BX, D]
    dWin[DX, F] = ReduceScatter(dWin[D, F] {UX}) (not on critical path, can be done async)
    Win[D, F] = AllGather(Win[DX, F]) (can be done ahead of time)
    dIn[BX, D] = dTmp[BX, F] *F Win[D, F] (needed for previous layers) (can throw away Win[D, F] here)
```

### 1. 核心思想与优势
*   **目标**：在纯数据并行的基础上，进一步**极致地节省内存**。
*   **方法**：不仅将**数据（激活值）** 沿批次维度（$B_X$）分片，还将**模型参数（W）、梯度（dW）和优化器状态**也进行分片（如 $W_in[D_X, F]$）到所有设备上。仅在需要时通过$AllGather$临时收集完整参数进行计算，计算完毕后立即丢弃。
*   **优势**：相比纯数据并行，FSDP将每个设备的内存占用从$O(模型总大小)$降低到$O(模型总大小 / 设备数)$，从而能够训练参数量远大于单设备内存的模型。

### 2. 算法流程与通信操作
*   **前向传播**：在计算每一层之前，通过$AllGather$从所有设备收集该层权重的完整副本（如 $W_in[D_X, F] -> W_in[D, F]$）。计算完成后，丢弃完整副本，仅保留分片。
*   **反向传播**：
    1.  计算得到完整权重的梯度（如 $dW_{out}[F, D]$）。
    2.  通过$ReduceScatter$操作，将完整梯度分散求和到各个设备，每个设备最终只保留自己分片对应的梯度（如 $dW_{out}[F, D] -> dW_{out}[F, D_X]$）。
    3.  同样，在计算需要时，会临时$AllGather$参数。
*   **通信重叠**：算法强调$AllGather$和$ReduceScatter$操作**不在关键路径上**或**可以异步执行**，这意味着这些通信操作可以与计算过程重叠，从而隐藏大部分通信开销。

### 3. 通信成本分析与瓶颈
*   **关键结论**：FSDP的**通信量与纯数据并行相同**。因为一个$AllReduce$操作在逻辑上等价于一个$AllGather$加上一个$ReduceScatter$。
*   **性能模型**：与纯数据并行类似，系统是否受通信瓶颈制约取决于**每设备批次大小（B/X）**。
    *   **计算时间**：$T_{math} ∝ (B * D * F) / (X * C)$ （与批次大小$B$成正比）
    *   **通信时间**：$T_{comms} ∝ (D * F) / W_ici$ （与批次大小$B$无关，是固定值）
*   **计算受限条件**：当 $B/X > C / W_ici$ 时，系统是**计算受限**的。对于TPU v5p，这个临界值约为**2550个token/设备**。只要每设备处理的token数大于此值，通信开销就能被计算有效隐藏。

### 4. 实践意义与重要启示
*   **无缝升级**：如果一个训练任务在纯数据并行下已经是计算受限的，那么可以**直接升级到FSDP**来节省大量内存，而无需担心会陷入通信瓶颈。
*   **规模与批次的矛盾**：一个反直觉的结论是，**总批次大小（B）越小，越容易受通信瓶颈限制**。因为$B$固定时，设备数（$X$）越多，每设备批次大小（$B/X$）就越小，越容易低于临界值。
*   **现代训练的挑战**：根据缩放定律，最优的总批次大小通常是确定的。当我们需要用极多的芯片（巨大的$X$）来训练一个批次大小相对较小（固定的$B$）的模型时，FSDP和数据并行可能会因为$B/X$过小而遇到通信瓶颈。这就引出了需要其他并行策略（如张量并行、流水线并行）来协同解决这一问题。

### 总结

FSDP是一种极其有效的内存优化技术，它通过分片模型状态，允许我们训练规模远超单设备内存的模型。其核心洞见在于通信成本与纯数据并行相当，但内存效益巨大。然而，其有效性依赖于足够的每设备计算量（即每设备批次大小），这在当今追求极大规模分布式训练的时代，成为一个重要的设计和权衡因素。

## 1.3 Tensor Parallelism

![Tensor Parallelism](https://jax-ml.github.io/scaling-book/assets/img/model-parallelism.png)

```Algorithm
Tensor Parallelism:

Forward pass: need to compute Loss[B]

    In[B, D] = AllGather(In[B, DY]) (on critical path)
    Tmp[B, FY] = In[B, D] *D Win[D, FY] (not sharded along contracting, so no comms)
    Out[B, D] {UY} = Tmp[B, FY] *F Wout[FY, D]
    Out[B, DY] = ReduceScatter(Out[B, D] {UY}) (on critical path)
    Loss[B] = …

Backward pass: need to compute dWout[FY, D], dWin[D, FY]

    dOut[B, DY] = …
    dOut[B, D] = AllGather(dOut[B, DY]) (on critical path)
    dWout[FY, D] = Tmp[B, FY] *B dOut[B, D]
    dTmp[B, FY] = dOut[B, D] *D Wout[FY, D] (can throw away dOut[B, D] here)
    In[B, D] = AllGather(In[B, DY]) (this can be skipped by sharing with (1) from the forward pass)
    dWin[D, FY] = dTmp[B, FY] *B In[B, D]
    dIn[B, D] {U.Y} = dTmp[B, FY] *F Win[D, FY] (needed for previous layers)
    dIn[B, DY] = ReduceScatter(dIn[B, D] {U.Y}) (on critical path)

```

### 1. 核心思想与工作原理
*   **目标**：将**单个过于庞大的模型层**（如前馈网络中的大矩阵）拆分到多个设备上计算，以解决该层无法放入单个设备内存的问题。
*   **方法**：将模型的**内在维度**（如隐藏维度$D$或前馈网络维度$F$）进行分片。如图中所示，输入$In$沿$D$维度分片（$[B, D_Y]$），权重矩阵$W_in$和$W_out$也相应地被分片（$[D, F_Y]$和$$[F_Y, D]$$）。
*   **与FSDP的区别**：FSDP是**数据并行**的增强，沿**批次维度（B）** 分片；张量并行是真正的**模型并行**，沿**模型维度（D/F）** 分片。

### 2. 算法流程与通信操作
*   **通信密集**：张量并行的通信操作**位于关键路径上**，无法像FSDP那样轻松异步化。
*   **前向传播**：
    1.  **AllGather**：在计算第一个矩阵乘法前，收集完整的输入激活值（$In[B, D_Y] -> In[B, D]$）。
    2.  **局部计算**：每个设备用本地分片的权重进行计算。
    3.  **ReduceScatter**：在第二个矩阵乘法后，对输出进行分散求和，得到最终分片的输出（$Out[B, D] -> Out[B, D_Y]$）。
*   **通信优化**：聪明的实现（如Megatron-LM）将两个矩阵乘法的计算串联起来，只需在开始和结束时进行一次$AllGather$和一次$ReduceScatter$，避免了在中间进行昂贵的$AllReduce$操作。

### 3. 通信成本分析与瓶颈
*   **性能模型**：
    *   **计算时间**：$T_math ∝ (B * D * F) / (Y * C)$ （与总计算量成正比，但被设备数$Y$分担）
    *   **通信时间**：$T_comms ∝ (B * D) / W_ici$ （通信量与**批次大小$B$** 成正比，这是与FSDP的关键区别）
*   **计算受限条件**：系统不受通信瓶颈限制的条件是 **$F > Y * (C / W_ici)$**。
    *   **物理意义**：只有当模型层的内部维度$F$足够大时，拆分到$Y$个设备上产生的计算量，才能掩盖住因收集和分散激活值（数据量为$B*D$）所产生的通信开销。
    *   **对于TPU v5p**，$C / W_ici ≈ 2550$。因此条件简化为 $F > Y * 2550$。

### 4. 关键结论与实践意义
*   **核心结论（Takeaway）**：张量并行在 **$Y > M_Y * F / 2550$** 时会变为通信受限。其中$M_Y$是通信网格的轴数，能提升有效带宽。
*   **并行度限制**：对于大多数模型，**张量并行的可行度通常被限制在8路或16路以内**。超过这个范围，通信开销将主导训练时间，导致效率下降。
*   **实例分析**：
    *   **LLaMA 3 70B**（$F ≈ 30,000$）：可轻松进行8路并行，但16路并行可能会遇到通信瓶颈。
    *   **Gemma 7B**（$F ≈ 50,000$）：可支持更高的并行度（约19路），因此16路并行仍能保持良好性能。
*   **组合使用**：常与FSDP（ZeRO分片）结合使用。FSDP减少了权重大小，从而也减小了张量并行中需要$AllGather$的激活值大小，使得张量并行更经济。

### 总结

张量并行是一种强大的模型并行技术，用于训练那些单个层就过于庞大的模型。然而，它的使用有严格的限制：**其效率高度依赖于模型层的大小（F）和采用的并行度（Y）**。由于通信量与批次大小（B）成正比，它在处理大模型时有效，但在并行度设置过高时会迅速遇到通信瓶颈。因此，在实践中，它通常被限制在较小的并行度内，并与FSDP等技术协同工作，以实现极大规模模型的高效训练。

## 1.3 Tensor Parallelism公式解释

### **公式 (4): 计算时间**

$$ T_{\text{math}} = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} $$

*   **物理意义**：完成一层前向传播所需的**计算时间**。
*   **分子 $4 · B · D · F$**：这是**总的浮点运算量**。
    *   $B · D · F$ 是单个矩阵乘法（如 $[B, D] · [D, F]$）的运算量级。
    *   一个MLP层有**两个**连续的矩阵乘法，所以乘以 $2$ -> $2 · B · D · F$。
    *   这里只建模了前向传播，其计算量约为反向传播的一半。图片脚注说明为了简化，只分析前向传播，反向传播是类似的转置操作。因此，完整的前后向传播计算量约是这里的2倍，即 $4 · B · D · F$。这个系数不影响最终比较的临界条件。
*   **分母 $Y · C$**：这是**系统的总计算能力**。
    *   $Y$ 是张量并行的设备数量。
    *   $C$ 是单个设备的计算吞吐量（FLOPs/秒）。
    *   设备数量 $Y$ 越多，总算力越强，计算时间越短。

**关键点**：计算时间与**总计算量 $B·D·F$** 成正比，与**设备数 $Y$** 成反比。

---

### **公式 (5): 通信时间**

$$ T_{\text{comms}} = \frac{2 \cdot 2 \cdot (B \cdot D)}{W_{\text{ici}}} $$

*   **物理意义**：完成一层前向传播所需的**通信时间**。
*   **分子 $2 · 2 · (B · D)$**：这是**总的通信数据量（字节数）**。
    *   $B · D$：是激活值（输入）张量的大小（元素个数）。
    *   第一个系数 $2$：因为激活值通常以 $bfloat16$ 精度存储，每个元素占 **2个字节**。所以字节数为 $2 · (B · D)$。
    *   第二个系数 $2$：因为一次前向传播需要进行**两次**集体通信：一次 $AllGather$（在第一个矩阵乘之前收集完整的输入）和一次 $ReduceScatter$（在第二个矩阵乘之后对输出进行分散求和）。每次通信的数据量约为 $B·D$ 个元素。
*   **分母 $W_ici$**：这是**芯片间互联带宽**（字节/秒）。

**关键点（与FSDP的根本区别）**：张量并行的通信量与**批次大小 $B$** 成正比。因为通信的内容是**激活值**，而激活值的大小取决于 $B$。这与FSDP不同（FSDP通信的是模型梯度，与 $B$ 无关）。

---

### **公式 (6), (7), (8), (9): 推导通信瓶颈的条件**

这些公式是推导的核心，目的是找出“计算时间大于通信时间”的条件，从而避免通信瓶颈。

1.  **公式 (6): 总时间估算**
    $$ T \approx \max(T_{\text{math}}, T_{\text{comms}}) $$
    *   在理想的重叠下，总时间由较慢的那个操作（计算或通信）决定。

2.  **公式 (7): 建立不等式**
    $$ \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{4 \cdot B \cdot D}{W_{\text{ici}}} $$
    *   我们希望系统是**计算受限**的，即 $T_math > T_comms$。
    *   注意，公式(5)中的 $2*2$ 就是 $4$，所以右边是 $(4 · B · D) / W_ici$。

3.  **公式 (8) & (9): 简化不等式**
    *   不等式两边同时除以 $4 · B · D$（这些项都大于0，可以约掉）：
    $$ \frac{F}{Y \cdot C} > \frac{1}{W_{\text{ici}}} $$
    *   然后，两边同时乘以 $Y · C · W_ici$，得到最终的临界条件：
    $$ F > Y \cdot \frac{C}{W_{\text{ici}}} $$

---

### 核心结论与解读

**最终条件 $F > Y · (C / W_ici)$ 的物理意义是：**

**只有当模型层的内部维度 $F$ 足够大时，将其拆分到 $Y$ 个设备上所产生的计算量，才能掩盖住因收集和分散激活值（数据量为 $B*D$）所产生的通信开销。**

*   **$C / W_ici$** 是一个**硬件常数**，代表“计算能力与通信带宽的比值”。对于TPUv5p，这个值约为 $2550$。
*   **$Y$** 是**并行度**。你希望用越多的设备来并行计算一个层（$Y$ 越大），这个层本身就必须越大（$F$ 越大），否则通信开销就会成为瓶颈。

**举例说明**：
对于TPUv5p，条件变为 $F > Y · 2550$。
*   如果一个层的 $F = 8192$，那么最大并行度 $Y$ 最好小于 $8192 / 2550 ≈ 3.2$。这意味着3路张量并行可能是高效的，但4路或以上就会开始出现明显的通信瓶颈。
*   对于 $F$ 非常大的层（如几万），则可以支持更高的并行度（如8路或16路）。

**图片最后蓝色框中的总结（Takeaway）** 正是基于此：
> 张量并行在 **$Y > M_Y * F / 2550$** 时会变为通信受限。对于大多数模型，这（临界点）在8到16路张量并行之间。

*   **$M_Y$** 是一个补充因素，指通信可以使用的硬件轴数量。如果通信能利用多个轴（如X轴和Y轴），有效带宽 $W_ici$ 会成倍增加，从而提升 $M_Y$ 倍，允许更高的并行度 $Y$。

### 总结

这张图片的公式清晰地表明，**张量并行是一种受通信严重制约的并行策略**。其效率不取决于批次大小 $B$，而取决于**模型层本身的规模（F）和所采用的并行度（Y）**。这解释了为什么在实践中，张量并行通常只用于模型中最庞大的那些层，并且并行度被限制在一个相对较小的范围内。

## 1.4 Combining FSDP and Tensor Parallelism

![Combining FSDP and Tensor Parallelism](https://jax-ml.github.io/scaling-book/assets/img/mixed-fsdp-model-parallelism.png)

```Algorithm
Forward pass: need to compute Loss[B]

    In[BX, D] = AllGatherY(In[BX, DY]) (on critical path)
    Win[D, FY] = AllGatherX(Win[DX, FY]) (can be done ahead of time)
    Tmp[BX, FY] = In[BX, D] *D Win[D, FY]
    Wout[FY, D] = AllGatherX(Wout[FY, DX]) (can be done ahead of time)
    Out[BX, D] {U.Y} = Tmp[BX, FY] *F Wout[FY, D]
    Out[BX, DY] = ReduceScatterY(Out[BX, D] {U.Y}) (on critical path)
    Loss[BX] = …

Backward pass: need to compute dWout[FY, DX], dWin[DX, FY]

    dOut[BX, DY] = …
    dOut[BX, D] = AllGatherY(dOut[BX, DY]) (on critical path)
    dWout[FY, D] {U.X} = Tmp[BX, FY] *B dOut[BX, D]
    dWout[FY, DX] = ReduceScatterX(dWout[FY, D] {U.X})
    Wout[FY, D] = AllGatherX(Wout[FY, DX]) (can be done ahead of time)
    dTmp[BX, FY] = dOut[BX, D] *D Wout[FY, D] (can throw away dOut[B, D] here)
    In[BX, D] = AllGatherY(In[BX, DY]) (not on critical path + this can be shared with (2) from the previous layer)
    dWin[D, FY] {U.X} = dTmp[BX, FY] *B In[BX, D]
    dWin[DX, FY] = ReduceScatterX(dWin[D, FY] {U.X})
    Win[D, FY] = AllGatherX(Win[DX, FY]) (can be done ahead of time)
    dIn[BX, D] {U.Y} = dTmp[BX, FY] *F Win[D, FY] (needed for previous layers)
    dIn[BX, DY] = ReduceScatterY(dIn[BX, D] {U.Y}) (on critical path)
```

### 核心目标与基本设定

**目标**：在总芯片数 $N$ 固定的情况下，如何将芯片分配给FSDP（数量记为 $X$）和TP（数量记为 $Y$），使得总通信时间最短，从而让系统更容易进入“计算受限”状态（即计算时间大于通信时间，硬件计算单元被充分利用）。

**基本关系**：$N = X * Y$。一旦确定了 $X$，$Y$ 也就确定了（$Y = N / X$）。

**关键变量**：
*   $B$：总批次大小（Token数量）
*   $F$：模型前馈层的维度（如 32,768）
*   $D$：模型隐藏层维度
*   $C$：单个芯片的计算能力（FLOPs/秒）
*   $W_ici$：芯片间互联带宽（Bytes/秒）
*   $α$：$α = C / W_ici$，是一个重要的硬件常数，代表“计算能力与通信带宽的比值”（对于TPUv5p，$α ≈ 2550 FLOPs/Byte$）。
*   $M_X$, $M_Y$：FSDP和TP通信所能利用的硬件网格轴数量。对于一个3D网格，通常有 $M_X * M_Y ≈ 3$，文中为简化取 $M_X * M_Y = 2$。

---

### 公式1：通信时间模型

图片首先建立了两种并行策略的通信时间模型。

**1. FSDP通信时间**
$T_{FSDP} = (4 * D * F) / (Y * W_ici * M_X)$

*   **物理意义**：FSDP需要通信的是**模型梯度**（大小正比于 $D * F$）。
*   **为何除以 $Y$**：当使用TP（$Y$ 路）时，模型权重在TP维度上已经被分片。因此，FSDP需要通信的梯度大小也减小为原来的 $1/Y$。
*   **$M_X$**：如果FSDP通信能利用多个硬件轴，有效带宽会提升 $M_X$ 倍。

**2. TP通信时间**
$T_{TP} = (4 * B * D) / (X * W_ici * M_Y)$

*   **物理意义**：TP需要通信的是**激活值**（大小正比于 $B * D$）。
*   **为何除以 $X$**：当使用FSDP（$X$ 路）时，批次大小 $B$ 在FSDP维度上被分片。因此，TP需要通信的激活值大小也减小为原来的 $1/X$。
*   **$M_Y$**：如果TP通信能利用多个硬件轴，有效带宽会提升 $M_Y$ 倍。

**3. 总通信时间**
$T_{comms} = max( T_{FSDP}, T_{TP} )$

*   由于FSDP和TP的通信可以重叠进行，总通信时间由两者中较慢的那个决定。

---

### 公式2：寻找最优芯片分配 (X_opt)

这是最关键的推导。我们希望找到一个 $X$，使得 $T_{comms}$ 最小。

**方法**：观察 $T_{comms}(X) = max( A*X, B/X )$（其中A和B是常数），这个函数在 $A*X = B/X$ 时取得最小值。因为当 $X$ 较小时，$T_{TP}$ 很大；当 $X$ 较大时，$T_{FSDP}$ 很大。最优解在两者平衡点。

**推导过程**：
令 $T_{FSDP} = T_{TP}$：
$$(F * X) / (N * M_X) = B / (X * M_Y)$$

解这个方程求 $X$：
$F * X * M_Y = B * N * M_X / X$ （两边同时乘以 $N * M_X * X$ 并整理）
$X² = (B / F) * (M_X / M_Y) * N$
$$X_{opt} = √( (B / F) * (M_X / M_Y) * N )$$

**结论**：最优的FSDP芯片数 $X_opt$ 与 **总批次大小 $B$ 的平方根** 和 **总芯片数 $N$ 的平方根** 成正比，与 **模型大小 $F$ 的平方根** 成反比。

**示例**：代入 $N=64$, $B=48,000$, $F=32,768$, $M_X/M_Y=1$，得到 $X_opt ≈ 13.9$。因此，一个接近最优的配置是 $X=16$（FSDP），$Y=N/X=4$（TP）。

---

### 公式3：成为“计算受限”的条件

最终目标是让系统“计算受限”，即计算时间 $T_{math}$ 大于通信时间 $T_{comms}$。

**1. 计算时间**
$T_{math} = (4 * B * D * F) / (N * C)$
*   总计算量 ($4 * B * D * F$) 由所有芯片 ($N$) 共同分担。

**2. 计算受限条件**
在最优配置 $X_opt$ 下，$T_{FSDP} = T_{TP}$。将 $X_opt$ 代入任意一个通信时间公式，并与 $T_math$ 比较，经过一系列代数推导（图片中略去了一些中间步骤），可以得到一个简洁的条件：

$$B/N > α² / (M_X * M_Y * F)$$

*   **$B/N$**：**每芯片批次大小**，这是衡量计算“粒度”的关键指标。
*   **物理意义**：要避免通信瓶颈，每个芯片必须处理足够多的数据（足够大的 $B/N$），使得计算时间能够掩盖通信时间。这个“足够大”的阈值，与硬件常数 $α$ 的平方成正比，与模型大小 $F$ 成反比。

**3. 巨大优势**
代入具体数值 $α=2550$, $F=32,768$, $M_X*M_Y=2$：
*   **纯FSDP所需条件**：$B/N > 850$
*   **FSDP+TP所需条件**：$B/N > 2550² / (2 * 32768) ≈ 99$

**最终结论**：**通过组合使用FSDP和TP，可以将训练时所需的最小每芯片批次大小降低约8倍**（从850降至约100）。这意味着在相同的总批次大小下，我们可以使用更多的芯片进行高效训练，或者在芯片数固定时，能够用更小的批次大小进行训练而不陷入通信瓶颈。这是实现极大规模模型训练的关键洞见。
