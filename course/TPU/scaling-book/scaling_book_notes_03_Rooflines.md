本文主要整理Part 1 Roofllines (All About Rooflines) 的主要内容。

## 1. 前言

1.  **三个关键硬件性能瓶颈**
    - **计算速度：** 硬件执行数学运算的峰值速度，单位为**OPS（操作次数）/秒**。这可以看作是芯片的“算力”天花板。
    - **数据带宽：** 在内存层级（如从显存/内存到缓存）之间移动数据的速率，单位为**字节/秒**。这决定了数据供给的速度。
    - **存储容量：** 可用于存储数据的总内存大小，单位为**字节**。

2.  **Roofline模型的核心作用**
    - 这些限制（特别是计算速度和带宽）构成了所谓的“roofline”约束。
    - 该模型的作用是，根据特定算法的计算特性和硬件的能力，**从理论上界定（上限和下限）完成一次计算所需的最短和最长时间**。这有助于开发者识别性能瓶颈是出现在“算得不够快”还是“数据来得不够快”，从而进行针对性优化。

## 2. Where Does the Time Go?

### Computation

1.  **核心论点**：深度学习模型的“计算”阶段耗时主要取决于其计算量和硬件的计算能力。
2.  **计算本质**：深度学习模型的计算主要由浮点数的乘法和加法（Floating-Point Operations, FLOPs）构成，可视为为矩阵乘法。
3.  **关键公式**：计算时间 $T_{math}$ 等于模型的总计算量（FLOPs）除以硬件的计算速度（FLOPs/s）。这一定义直观地表明，更快的硬件或更小的模型都能缩短计算时间。
4.  **实例说明**：通过具体数据对比了两种主流加速器：
    *   **NVIDIA H100**：算力约为 $9.89 x 10^14$ FLOPs/s (bfloat16精度)。
    *   **Google TPU v6e**：算力约为 $9.1 x 10^14$ FLOPs/s。
    *   **计算示例**：执行 $1 x 10^12$ (1e12) 次浮点运算，在H100上约需 **1.01毫秒**，在TPU v6e上约需 **1.1毫秒**。

5.  **核心公式**：

$$
T_{math} = (Computation FLOPs) / (Accelerator FLOPs/s)
$$

### Communication within a chip

1.  **核心概念：芯片内通信**
    *   图片聚焦于加速器芯片**内部**的数据流动问题，即计算所需的数据（张量）需要从存储单元（HBM）高效地传输到执行计算的单元（计算核心）。

2.  **关键性能指标：HBM带宽**
    *   **HBM带宽** 是衡量这种芯片内部数据传输速率的决定性指标，单位通常是TB/s（太字节每秒）。它直接决定了计算核心获取数据的速度，是影响整体计算效率的关键因素之一。

3.  **实例对比：主流芯片的HBM带宽**
    *   图片提供了两个具体示例，直观展示了不同硬件的能力差异：
        *   **NVIDIA H100**：HBM带宽约为 **3.35 TB/s**。
        *   **Google TPU v6e**：HBM带宽约为 **1.6 TB/s**。
    *   这一对比表明，在数据搬运方面，H100具有显著的带宽优势。

### Communication between chips


1.  **核心问题：芯片间通信是分布式训练的瓶颈**
    *   当模型分布在多个加速器上时，芯片间的张量传输成为必然，其速度受到特定硬件链路（带宽）的限制。

2.  **通信时间量化**
    *   通信时间 $T_{comms}$ 可以通过一个基本公式估算：**传输的数据量（字节） / 网络或内存带宽（字节/秒）**。这与衡量计算时间 $T_{math}$ 的逻辑类似。

$$
T_{comms} = (通信的字节数) / (网络或内存带宽 (字节/秒))
$$

3.  **性能估算的上下界模型**
    *   **下界（理想情况）**： 假设计算和通信能够**完全重叠**（并行执行），总时间由较慢的那个阶段决定。这是优化的目标。
    *   **上界（最差情况）**： 假设计算和通信**顺序执行**（一个完成后才开始另一个），总时间为两者之和。

$$
T_lower = max(T_{math}, T_{comms})     
$$

$$
T_upper = T_{math} + T_{comms}
$$


4.  **关键性能状态：“计算受限” vs “通信受限”**
    *   **计算受限**： 当 $T_{math} > T_{comms}$ 时，计算是瓶颈，通信能够被很好地隐藏，硬件计算单元得到充分利用。这是理想状态。
    *   **通信受限**： 当 $T_{comms} > T_{math}$ 时，通信是瓶颈，计算单元需要空闲等待数据，导致算力浪费。这是需要尽力避免的状态。

### arithmetic intensity


1.  **核心概念：算术强度**
    *   定义：算法执行的**总浮点运算次数** 与需要**通信的数据总量（字节数）** 的比值。它衡量的是“每字节数据能完成多少次浮点运算”。

$$
Arithmetic Intensity = Computation FLOPs / Communication Bytes
$$

2.  **算术强度的意义**
    *   **高算术强度**：意味着计算量大而数据移动相对少，$T_{math}$（计算时间）通常远大于$T_{comms}$（通信时间），能充分利用硬件的计算能力（FLOPs），属于**计算受限**。
    *   **低算术强度**：意味着数据移动量大而计算相对轻量，$T_{comms}$会成为瓶颈，计算单元经常空闲等待数据，浪费算力，属于**通信受限**。

3.  **关键判断条件：与硬件能力比较**
    *   硬件本身有一个关键的“**峰值算术强度**”，即 $硬件峰值算力 (FLOPs/s) / 硬件带宽 (Bytes/s)$。
    *   判断法则：**算法的算术强度 > 硬件的峰值算术强度**，则操作是**计算受限**的；反之，则是**通信受限**的。

4.  **实例验证：向量点积**
    *   以bfloat16精度的向量点积为例，计算得出其算术强度随向量长度增大趋近于 **0.5 FLOPs/byte**。
    *   对比TPU v5e MXU的峰值算术强度（约 **240 FLOPs/byte**），0.5远小于240，因此向量点积是一个典型的**通信受限**操作。
$$
Intensity(dot product) = (N + N - 1) / (2N bytes + 2N bytes + 2 bytes) = (2N - 1) / (4N + 2)
$$

## 3. Visualizing rooflines

![rooflines](https://jax-ml.github.io/scaling-book/assets/img/roofline-improved-1400.webp)

### 内容概要
该图表在一个对数-对数坐标图中，将算法的**算术强度**与在特定硬件上能达到的**峰值计算吞吐率**关联起来，从而清晰地揭示出算法是受限于内存带宽还是计算能力。图中通过示例说明了不同算法所处的性能区域，并指出了提升性能的方向。

---

### 要点总结

1.  **核心目的**：屋顶线图用于**可视化内存带宽与计算能力之间的权衡关系**，帮助诊断算法在硬件上的性能瓶颈。
2.  **图表坐标**：
    *   **横轴**：算法的**算术强度**。即每字节数据能完成多少次浮点运算。
    *   **纵轴**：算法能达到的**实际性能**，单位为每秒浮点运算次数。它代表算法在硬件上实际能达到的计算速度。
3.  **关键要素**：
    *   **“屋顶”**：图表中的两条水平线代表硬件的**峰值计算能力**，这是一个固定上限。
    *   **“斜线屋顶”**：一条上升的斜线，其斜率代表**内存带宽**。带宽越大，斜线越靠右上方。
4.  **性能区域划分**：
    *   **带宽受限区**：当算法强度低于一个**临界值**时，性能位于斜线上。此时性能随强度增加而线性增长，但无法达到硬件峰值算力，造成算力浪费。
    *   **计算受限区**：当算法强度高于临界值时，性能触及水平“屋顶”。此时算法能完全利用硬件算力，性能达到最大，但再增加强度或带宽也无济于事。
5.  **核心洞察**：
    *   **临界算术强度**：是斜线屋顶与水平屋顶的交点，是判断瓶颈的关键阈值。图中以TPU v5e为例，该值为 **240 FLOPs/byte**。
    *   **优化方向**：
        *   对于位于**带宽受限区**的算法，应优先通过优化（如改变循环顺序、使用分块技术）来**提高其算术强度**。
        *   如果无法优化算法，则**升级具有更高带宽的硬件**可以将斜线屋顶推高，直接提升性能。

## 4. Matrix multiplication

### 内容概况
内容聚焦于计算**bfloat16精度下矩阵乘法的算术强度**，并通过合理的假设将其简化为一个非常简洁的表达式。基于此，图片得出了一个极具实用价值的经验法则，用于判断矩阵乘法操作是否能够充分利用硬件的计算能力（即达到“计算受限”状态）。

---

### 要点总结

1.  **分析对象**： 针对bfloat16数据类型的矩阵乘法 $ X * Y = Z $（其中 $ X $ 的形状为 $[B, D]$，$ Y $ 的形状为 $[D, F]$）。
2.  **核心简化**： 在Transformer模型常见的场景下（即**批次大小 $B$** 远小于矩阵维度 $D$ 和 $F$），矩阵乘法的算术强度可以极大地简化为近似等于批次大小 $B$。
    *   **公式推导**： 精确的算术强度公式为 $ \frac{BDF}{BD + DF + BF} $。当 $B$ 相对 $D$ 和 $F$ 很小时，$BD$ 和 $BF$ 项可忽略，公式简化为 $ \frac{BDF}{DF} = B $。
3.  **关键结论（经验法则）**：
    *   要使bfloat16矩阵乘法在TPU上达到**计算受限**（即完全利用硬件算力），需要满足：**算法的算术强度 > 硬件的峰值算术强度**。
    *   代入TPU v5e的峰值算力（~1.97e14 FLOPs/s）和HBM带宽（~8.20e11 Bytes/s），得出其峰值算术强度约为 **240 FLOPs/byte**。
    *   因此，判断条件简化为：**$B > 240$**。
4.  **实用指导**： 对于Transformer模型中的矩阵乘法，**只要每个副本（per-replica）处理的令牌批次大小（$B$）大于240**，就有很大可能达到计算受限状态，从而高效利用TPU的算力。对于GPU，此临界值略高（约300），但原则相同。
5.  **注意事项**： 图片也指出了该规则的例外情况，例如当进行量化（降低数据精度但仍进行全精度计算）或将大矩阵乘法分解为小矩阵乘法时，需要更细致的分析。

## 5. Network communication rooflines

### 内容概况

将经典的 **“屋顶线”性能分析模型**的应用场景从**单个芯片内部**扩展到了**多芯片（如多个TPU/GPU）之间的网络通信**。它通过一个具体的示例（将大矩阵乘法在2个TPU上沿D维度分片并行计算）来说明，在这种分布式计算中，性能瓶颈可能从芯片的计算能力或内存带宽，转变为**芯片间的网络通信带宽**。

---

### 要点总结

1.  **核心拓展：屋顶线模型的新维度**
    *   传统的屋顶线模型主要分析单个芯片内的**内存带宽**瓶颈。本图强调，在分布式深度学习中，**网络通信带宽**是更常见、更关键的瓶颈来源。

2.  **分布式计算模式**
    *   示例描述了一种常见的矩阵乘法并行化策略：将两个大矩阵 $X$ 和 $Y$ 沿内在维度 $D$ 切分，每个TPU计算一个局部结果（部分和），然后通过网络交换这些部分和并进行累加，得到最终结果。

3.  **计算与通信时间的变化**
    *   **计算时间 $T_{math}$**：由于计算工作量被平均分配到两个TPU上，因此每个TPU的计算时间变为单芯片计算时间的一半。
    $$
    T_math = (2 B D F) / (2 * Accelerator FLOPs/s) = (B D F) / (1.97e14)
    $$

    *   **通信时间 $T_{comms}$**：此时指的是**跨芯片网络通信**的时间。通信量主要来自交换两个大小为 $[B, F]$ 的部分和矩阵，总数据量为 $2 * B * F$ 个bfloat16元素（即 $4 * B * F$ 字节）。
    $$
    T_comms = (2 B F) / Network Bandwidth = (2 B F) / (4.5e10)
    $$

4.  **关键洞察：临界条件的转变**
    *   在单芯片分析中，判断是否“计算受限”的临界条件取决于**批次大小 $B$**。
    *   在**网络通信屋顶线**模型中，临界条件转变为取决于矩阵的**内在维度 $D$**（公式推导为 $D > 8755$）。
    *   **原因解释**：因为通信数据量 $2BF$ 与 $B$ 和 $F$ 成正比，而总计算量 $BDF$ 与 $B$、$D$、$F$ 都成正比。两者的比值（即算术强度）为 $(BDF) / (2BF) = D/2$。这意味着，**$D$ 的大小直接决定了每次通信能承载多少计算量**。$D$ 越大，算术强度越高，越容易掩盖通信开销，从而让计算成为瓶颈。

## 6. A Few Problems to Work

### Question 1 [int8 matmul]

Say we want to do the matmul X[B,D] Y[D,F] => Z[B,F] in int8 precision (1 byte per parameter) instead of bfloat16.
- How many bytes need to be loaded from memory? How many need to be written back to memory? => BD + DF; BF
- How many total OPs are performed? => 2BDF
- What is the arithmetic intensity?
    - 2BDF / (BD + DF + BF) => 2B > 3.94e14$ / 8.1e11  => B > 243
- What is a roofline estimate for $T_{math}$ and $T_{comms}$ ? What are reasonable upper and lower bounds for the runtime of the whole operation? 
    - $T_{math} = 2BDF / 3.94e14$  
    - $T_{comms} = (BD + DF + BF) / 8.1e11 
    - $upper = max(T_{math}, T_{comms})$
    - $lower = T_{math} + T_{comms}

### Question 2 [int8 + bf16 matmul]

In practice we often do different weight vs. activation quantization, so we might store our weights in very low precision but keep activations (and compute) in a higher precision. Say we want to quantize our weights in int8 but keep activations (and compute) in bfloat16. At what batch size do we become compute bound? Assume 1.97e14 bfloat16 FLOPs/s.
    - 2B > 1.97e14$ / 8.1e11  => B > 120

### Question 3

Taking the setup from Question 2, make a roofline plot of peak FLOPs/s vs. BB for F=D=4096F=D=4096 and F=D=1024F=D=1024 . Use the exact number of bytes loaded, not an approximation.

![Question 3](https://jax-ml.github.io/scaling-book/assets/img/roofline-plot-q3-1400.webp)

### Question 4

What if we wanted to perform int8[B, D]∗int8[B, D, F]→int8[B, F] where we imagine having a different matrix for each batch element. What is the arithmetic intensity of this operation?

- 2BDF / (BD + BDF + BF) = 2DF / (D + F + DF)

### Problem 5 [Memory Rooflines for GPUs]

 Using the spec sheet provided by NVIDIA for the H100, calculate the batch size at which a matrix multiplication will become compute-bound. 

- From the spec sheet, we see that the reported bfloat16 FLOPs value is 1.979e15 FLOPs/s with an asterisk noting “with sparsity”. **The true value is half this without sparsity, meaning close to 1e15 FLOPs/s**. The memory bandwidth is 3.35TB/s, or 3.35e12 bytes / second. Thus $B_{crit}$​ is 1e15 / 3.35e12 = 298, rather similar to the TPU.