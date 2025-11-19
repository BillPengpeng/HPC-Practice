本文主要整理Part 12 GPUs (How to Think About GPUs) 的主要内容。

## 1. 前言

This chapter takes a deep dive into the world of GPUs – how each chip works, how they’re networked together, and what that means for LLMs, especially compared to TPUs. 

## 2.0 What Is a GPU?

![GPU](https://jax-ml.github.io/scaling-book/assets/gpu/gpu-diagram.png)

1.  **核心架构理念**：现代ML GPU本质上是由大量并行计算单元（SMs）和一块高速内存（HBM）通过缓存层级结构连接而成的异构计算系统。
2.  **关键组件与功能**：
    *   **流式多处理器（SM）**：是GPU的核心计算单元，每个SM包含：
        *   **Tensor Cores**： 专门执行矩阵乘法的硬件单元，是GPU峰值算力（FLOPS）的主要来源。
        *   **Warp Scheduler**： 一个包含32个“CUDA核心”的SIMD（单指令多数据）向量单元及其调度器，负责指挥计算任务。
        *   **L1缓存/共享内存（SMEM）**： 位于每个SM内部的快速片上缓存，用于加速数据访问。
    *   **L2缓存**： 一个容量较大（如H100为50MB）、带宽较高的共享缓存，作为SM和HBM之间的缓冲区。
    *   **高带宽内存（HBM/DRAM）**： 存储所有训练或推理所需的数据，包括模型参数、激活值、优化器状态等。其容量巨大（H100为80GB，B200为192GB）。
3.  **设计特点**：
    *   **并行性**： 拥有大量SMs（H100有132个，B200有148个）以实现大规模并行计算。
    *   **内存层级**： 采用多级缓存（L1, L2）来缓解HBM与计算核心之间的速度差距（“内存墙”问题）。
    *   **专业化**： Tensor Core的专门化设计使其在执行深度学习中的核心运算（矩阵乘法）时极其高效。
4.  **与TPU的类比**： 该架构与谷歌的TPU（张量处理单元）在设计思路上有显著相似之处，例如Tensor Core类似于TPU的MXU（矩阵乘法单元），Warp Scheduler类似于TPU的VPU（向量处理单元）。

## 2.1 SM

![SM](https://jax-ml.github.io/scaling-book/assets/gpu/blackwell-sm.png)

1.  **核心设计思想：大规模并行与异构计算**
    *   **灵活性 vs. 专一性**：与TPU仅有少量强大计算核心不同，GPU采用大量相对较小、独立的SM（H100有132个）。这使得GPU不仅能处理大规模矩阵乘法，还能高效执行成千上万个并行的小任务，灵活性更高。
    *   **独立性与并行性**：每个SM基本独立，使得GPU可以同时执行数百项独立任务。

2.  **SM：GPU的基本构建块**
    *   SM是GPU的核心计算单元，每个SM都是一个功能齐全的微型处理器，包含：
        *   **Tensor Core**：专用矩阵乘法单元，是GPU算力的主要来源（占绝大部分FLOPS）。
        *   **CUDA核心**：属于SIMD向量单元，执行通用算术运算（如FP32/INT32加法）、激活函数（ReLU）和规约操作。
        *   **调度与存储**：包含Warp Scheduler（指令调度器）、Register File（寄存器文件）和SMEM（共享内存/L1缓存）。

3.  **SM的层次化结构**
    *   每个SM被进一步划分为**4个相同的子分区**。每个子分区都拥有自己的一套计算资源（Tensor Core、CUDA核心、寄存器），这实现了SM内部的细粒度并行。

4.  **Tensor Core的核心地位与演进**
    *   **算力主导**：Tensor Core提供了绝对主要的算力。例如，H100的Tensor Core提供990 BF16 TFLOP/s，而CUDA核心仅提供66 TFLOP/s。
    *   **低精度高性能**：支持低精度计算（如FP8, BF16）以获得更高吞吐量，这对AI训练和推理至关重要。
    *   **持续演进**：自Volta架构以来，Tensor Core的尺寸和性能逐代增长。B200的Tensor Core如此之大，以至于需要引入新的TMEM内存空间来容纳其输入数据。

## 2.2 CUDA cores are more flexible than a TPU’s VPU

![warps diverge](https://jax-ml.github.io/scaling-book/assets/gpu/warp-divergence.png)

1.  **核心对比：SIMT vs. SIMD**
    *   **GPU (CUDA Core)**： 使用 **SIMT** 模型。每个线程是独立的实体，有自己的指令指针和寄存器状态，编程模型更接近多线程CPU。
    *   **TPU (VPU)**： 使用 **SIMD** 模型。所有处理单元在同一周期内锁定步调，执行完全相同的操作。

2.  **SIMT模型的灵活性体现**
    *   **独立线程编程**： 程序员可以为每个线程编写不同的执行路径（如使用`if-else`语句），而无需显式地管理向量化。
    *   **独立内存访问**： CUDA线程可以灵活地访问共享寄存器中的单个数据元素，而TPU的VPU通常需要操作连续的内存块。

3.  **灵活性的代价：分支发散**
    *   **执行机制**： 当Warp内线程发生分支时（例如，部分线程执行`if`块，另一部分执行`else`块），GPU会**串行化**地执行所有分支路径。在执行某一路径时，会屏蔽掉不参与该路径的线程。
    *   **性能影响**： 图中时序图的“白色空格”直观显示了**部分物理CUDA核心在此期间被暂停**，导致计算资源未被充分利用。如果程序中分支发散现象非常普遍，将严重降低性能。

4.  **图示解析**
    *   **代码部分**： 一个简单的条件判断 `if (threadIdx.x < 4)`，导致一个包含8个线程的Warp（假设）被分为两组（4个线程一组）。
    *   **时序图部分**：
        *   **发散**： 线程束在条件判断处分裂，硬件先执行第一组线程的`A; B;`操作，同时暂停第二组线程。
        *   **执行**： 然后，硬件再执行第二组线程的`X; Y;`操作，同时暂停第一组线程。
        *   **汇合**： 在`if-else`语句结束后，所有线程重新同步，一起执行后续的`Z;`操作。

总而言之，这张图深刻地揭示了GPU编程中的一个关键权衡：**SIMT模型用潜在的硬件资源浪费（核心暂停）换取了编程上的极大简便性和灵活性**。高性能CUDA编程的艺术，在很大程度上就在于如何巧妙地组织数据和计算，以最大限度地减少线程束内的分支发散。

## 2.3 CUDA core scheduling is also more flexible

**SMs run a bit like multi-threaded CPUs**, in the sense that they can “schedule” many programs (warps) concurrently (up to 64 per SM) but each Warp Scheduler only ever executes a single program in each clock cycle. 8 The Warp Scheduler automatically switches between active warps to hide I/O operations like memory loads. **TPUs are generally single threaded by comparison**.

## 3. Memory

1.  **寄存器**
    *   **位置与功能**：位于每个SM子分区中，是**最快、最接近CUDA核心**的内存，用于存储线程的私有变量。
    *   **关键细节**：资源有限。每个线程可使用的寄存器数量有上限，这直接限制了SM内能同时保持活跃的线程束数量，是优化线程资源调度的关键因素。

2.  **共享内存 / L1缓存**
    *   **位置与功能**：位于每个SM内部的**片上缓存**，称为SMEM。它速度极快，容量为256kB。
    *   **核心特性与双重角色**：
        *   **可编程性**：其独特优势在于可由程序员显式控制作为**共享内存**，用于线程块内线程间的通信，或由硬件自动管理作为**L1缓存**。
        *   **关键用途**：是存储张量核心矩阵乘法所需的输入和激活值的重要空间。

3.  **L2缓存**
    *   **位置与功能**：一个**所有SM共享**的、容量较大（约50MB）的缓存，作为HBM和SM之间的缓冲区。
    *   **关键对比与挑战**：
        *   **与TPU对比**：虽然容量与TPU的VMEM相似，但**速度较慢且不可由程序员直接控制**。
        *   **编程影响**：程序员需要通过优化内存访问模式来间接确保L2缓存被有效利用，文中称之为“超距作用”，增加了优化难度。

4.  **高带宽内存**
    *   **位置与功能**：**主GPU内存**，用于存储所有模型数据（权重、梯度、激活值等）。
    *   **发展趋势**：容量和带宽持续快速提升，从Volta架构的32GB@约900GB/s，发展到Blackwell B200的192GB@8TB/s，以应对更大模型的需求。
    *   **“内存墙”问题**：HBM与计算核心之间的带宽差距是制约GPU算力发挥的瓶颈，因此整个内存层级结构的设计目的就是尽量减少对HBM的访问。

## 4. Summary of GPU specs


1.  **演进趋势**：从V100到B200，GPU在各个方面持续增强：
    *   **计算规模**：每芯片的**SM数量**持续增加（80 → 148）。
    *   **内存系统**：**L2缓存**和**HBM容量/带宽**大幅提升，以缓解“内存墙”瓶颈（如HBM带宽从V100的900GB/s增至B200的8TB/s）。
    *   **计算能力**：专为AI设计的**低精度算力（如FP8/BF16）** 呈数量级增长，且支持的数据精度越来越低（B200已支持FP4）。

2.  **代际亮点**：
    *   **A100（Ampere）**：引入了对FP8/BF16精度的Tensor Core支持，AI算力实现飞跃。
    *   **H100/H200（Hopper）**：SMEM容量增至256kB/SM，H200重点提升了HBM容量（141GB）和带宽。
    *   **B200（Blackwell）**：实现最大跨越，L2缓存（126MB）、HBM容量（192GB）及所有精度下的算力均达到新高，并新增了**TMEM**。

3.  **与TPU的对比**：对照表明确了GPU和TPU虽然术语不同，但架构思想相似，存在明确的对应关系，例如：
    *   **SM** 对应 **Tensor Core**（核心计算单元）。
    *   **Tensor Core** 对应 **MXU**（矩阵计算单元）。
    *   **SMEM** 对应 **VMEM**（片上缓存）。

**表1：GPU架构与内存容量规格摘要**

| GPU | 架构 | 时钟频率 | 每芯片SM数 | 每SM SMEM容量 | 每芯片L2容量 | 每芯片HBM容量 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| V100 | Volta | 1.25GHz/1.38GHz | 80 | 96kB | 6MB | 32GB |
| A100 | Ampere | 1.10GHz/1.41GHz | 108 | 192kB | 40MB | 80GB |
| H100 | Hopper | 1.59GHz/1.98GHz | 132 | 256kB | 50MB | 80GB |
| H200 | Hopper | 1.59GHz/1.98GHz | 132 | 256kB | 50MB | 141GB |
| B200 | Blackwell | ? | 148 | 256kB | 126MB | 192GB |

> **注：** 所有架构的每个SM都拥有256kB寄存器内存。Blackwell架构额外为每个SM增加了256kB的TMEM。

**表2：GPU带宽与算力规格摘要**

| GPU | 架构 | 每芯片HBM带宽 | 每芯片算力 (bf16/fp16) | 每芯片算力 (fp8/int8) | 每芯片算力 (fp4) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| V100 | Volta | 9.0e11 (900 GB/s) | - | - | - |
| A100 | Ampere | 2.0e12 (2 TB/s) | 3.1e14 (312 TFLOPS) | 6.2e14 (624 TFLOPS) | - |
| H100 | Hopper | 3.4e12 (3.4 TB/s) | 9.9e14 (990 TFLOPS) | 2.0e15 (2 PFLOPs) | - |
| H200 | Hopper | 4.8e12 (4.8 TB/s) | 9.9e14 (990 TFLOPS) | 2.0e15 (2 PFLOPs) | - |
| B200 | Blackwell | 8.0e12 (8 TB/s) | 2.3e15 (2.3 PFLOPs) | 4.5e15 (4.5 PFLOPs) | 9.0e15 (9 PFLOPs) |

> **注：** 为便于阅读，括号内已添加了单位换算。B100因未大规模生产而被排除。部分规格可能因GPU具体版本略有不同。

**表3：GPU与TPU组件对照表**

| GPU 组件 | TPU 组件 | 说明 |
| :--- | :--- | :--- |
| 流式多处理器（SM） | 张量核心（Tensor Core） | 包含其他功能单元的**核心“细胞”** |
| Warp调度器 | 向量处理单元（VPU） | 执行SIMD向量算术的单元 |
| CUDA核心 | VPU中的算术逻辑单元（ALU） | SIMD算术逻辑单元 |
| SMEM（L1缓存） | VMEM | 快速的**片上缓存** |
| Tensor Core | 矩阵乘法单元（MXU） | 专用于**矩阵乘法**的单元 |
| HBM（或称GMEM） | HBM | 高带宽、高容量的**主内存** |

## 5. GPUs vs. TPUs at the chip level

1.  **设计哲学与历史根源**
    *   **GPU**： 源于图形处理，目标是**通用加速器**。硬件设计追求“通用性”，能适应更多样化的任务，对编译器的依赖较小。
    *   **TPU**： 专为深度学习中的**矩阵乘法**而生，是**专用集成电路**。设计更纯粹，目标明确。

2.  **核心架构：模块化 vs 集中化**
    *   **GPU（模块化）**： 由大量小型、独立的计算单元（**SM**）组成。例如，H100拥有132个SM和528个小型SIMD单元。这种设计允许并发执行大量独立任务，灵活性高。
    *   **TPU（集中化）**： 仅包含1-2个大型**Tensor Core**，每个核心内有4个强大的**VPU**。这种设计控制逻辑简单，但要求编译器必须精确调度所有指令和数据流。

3.  **性能与成本的权衡**
    *   **单芯片能力**： 历史上，单个GPU（如H200）比同代TPU（v5p）**更强大且更昂贵**（算力约2倍，价格约2.5倍）。
    *   **系统级扩展**： TPU更倾向于通过高速网络将多个芯片紧密连接来构建大规模系统。

4.  **缓存内存的关键差异**
    *   **TPU优势**： TPU拥有**远多于GPU的快速片上缓存（VMEM）**。这使其在LLM推理等场景中，如果能将权重预存到VMEM，将获得极高的速度优势。
    *   **性能优化**： GPU的硬件自动管理更多，易于上手但性能瓶颈难分析；TPU依赖编译器手动优化，挑战大但一旦优化好，更容易接近理论峰值性能。

| GPU 组件 | TPU 组件 | H100 数量 | TPU v5p 数量 |
| :--- | :--- | :--- | :--- |
| **SM** (流式多处理器) | **Tensor Core** (张量核心) | 132 | 2* |
| **Warp Scheduler** (线程束调度器) | **VPU** (向量处理单元) | 528 (132 SM * 4) | 8 (2 Tensor Core * 4) |
| **SMEM** (L1缓存/共享内存) | **VMEM** | 32MB | 128MB |
| **Registers** (寄存器) | **Vector Registers** (向量寄存器) | 32MB | 256KB |
| **Tensor Core** (张量核心) | **MXU** (矩阵乘法单元) | 528 (132 SM * 4) | 8 (2 Tensor Core * 4) |

## 6. Quiz 1: GPU hardware

### Question 1 [CUDA cores]

How many fp32 CUDA cores (ALUs) does an H100 have? B200? How does this compare to the number of independent ALUs in a TPU v5p?

- An H100 has 132 SMs with 4 subpartitions each containing 32 fp32 CUDA cores, so we **132 * 4 * 32 = 16896 CUDA cores**. A B200 has has 148 SMs, so a total of 18944. A TPU v5p has 2 TensorCores (usually connected via Megacore), each with a VPU with (8, 128) lanes and 4 independent ALUs per lane, so 2 * 4 * 8 * 128 = 8192 ALUs. This is roughly half the number of vector lanes of an H100, running at roughly the same frequency.

### Question 2 [Vector FLOPs calculation]:

A single H100 has 132 SMs and runs at a clock speed of 1.59GHz (up to 1.98GHz boost). Assume it can do one vector op per cycle per ALU. How many vector fp32 FLOPs can be done per second? With boost? How does this compare to matmul FLOPs?

- **132 * 4 * 32 * 1.59e9 = 26.9TFLOPs/s**. With boost its 33.5 TFLOPs/s. This is half what’s reported in the spec sheet because technically we can do an FMA (fused-multiply-add) in one cycle which counts as two FLOPs, but this isn’t useful in most cases. We can do 990 bfloat16 matmul TFLOPs/s, so ignoring FMAs, Tensor Cores do around 30x more FLOPs/s.

### Question 3 [GPU matmul intensity]

What is the peak fp16 matmul intensity on an H100? A B200? What about fp8? By intensity we mean the ratio of matmul FLOPs/s to memory bandwidth.

- For an H100, we have a peak 990e12 fp16 FLOPs and 3.35e12 bytes / s of bandwidth. **So the critical intensity is 990e12 / 3.35e12 = 295, fairly similar to the 240 in a TPU. For B200 its 2250e12 / 8e12 = 281**, very similar. This means, similar to TPUs, that we need a batch size of around 280 to be compute-bound in a matmul.

For both H100 and B200 we have exactly 2x fp8 FLOPs, so the peak intensity also doubles to 590 and 562 respectively, although in some sense it stays constant if we take into account the fact that our weights will likely be loaded in fp8 as well.

### Question 4 [Matmul runtime]

Using the answer to Question 3, how long would you expect an fp16[64, 4096] * fp16[4096, 8192] matmul to take on a single B200? How about fp16[512, 4096] * fp16[4096, 8192]?

- From the above, we know we’ll be communication-bound below a batch size of 281 tokens. Thus the first is purely bandwidth bound. We read or write 2BD+2DF+2BF2BD+2DF+2BF bytes (2*64*4096 + 2*4096*8192 + 2*64*8192=69e6) with 8e12 bytes/s of bandwidth, so it will take about **69e6 / 8e12 = 8.6us**. In practice we likely get a fraction of the total bandwidth, so it may take closer to 10-12us. When we increase the batch size, we’re fully compute-bound, so we expect **T=2*512*4096*8192/2.3e15=15us**. We again only expect a fraction of the total FLOPs, so we may see closer to 20us.

### Question 5 [L1 cache capacity]

What is the total L1/SMEM capacity for an H100? What about register memory? How does this compare to TPU VMEM capacity?

- We have 256kB SMEM and 256kB of register memory per SM, so about 33MB (132 * 256kB) of each. Together, this gives us a total of about 66MB. This is about half the 120MB of a modern TPU’s VMEM, although a TPU only has 256kB of register memory total! TPU VMEM latency is lower than SMEM latency, which is one reason why register memory on TPUs is not that crucial (spills and fills to VMEM are cheap).

### Question 6 [Calculating B200 clock frequency]

NVIDIA reports here that a B200 can perform 80TFLOPs/s of vector fp32 compute. Given that each CUDA core can perform 2 FLOPs/cycle in a FMA (fused multiply add) op, estimate the peak clock cycle.

- We know we have 148 * 4 * 32 = 18944 CUDA cores, so we can do 18944 * 2 = 37888 FLOPs / cycle. Therefore **80e12 / 37888 = 2.1GHz**, a high but reasonable peak clock speed. B200s are generally liquid cooled, so the higher clock cycle is more reasonable.

### Question 7 [Estimating H100 add runtime]

Using the figures above, calculate how long it ought to take to add two fp32[N] vectors together on a single H100. Calculate both $T_{math}$​ and $T_{comms}$​. What is the arithmetic intensity of this operation? If you can get access, try running this operation in PyTorch or JAX as well for N = 1024 and N=1024 * 1024 * 1024. How does this compare?

- Firstly, adding two fp32[N] vectors performs N FLOPs and requires 4 * N * 2 bytes to be loaded and 4 * N bytes to be written back, for a total of 3 * 4 * N = 12N. Computing their ratio, we have total FLOPs / total bytes = N / 12N = 1 / 12, which is pretty abysmal.
- As we calculated above, we can do roughly 33.5 TFLOPs/s boost, ignoring FMA. This is only if all CUDA cores are used. For N = 1024, we can only use at most 1024 CUDA cores or 8 SMs, which will take longer (roughly 16x longer assuming we’re compute-bound). We also have a memory bandwidth of 3.35e12 bytes/s. Thus our peak hardware intensity is 33.5e12 / 3.35e12 = 10. 13 So we’re going to be horribly comms bound. Thus our runtime is just

$$T = max(T_{comms}, T_{math}) = 12N / 3.35e12$$

- For N = 65,536, this is about 0.23us. In practice we see a runtime of about 1.5us in JAX, which is fine because we expect to be super latency bound here. For N = 1024 * 1024 * 1024, we have a roofline of about 3.84ms, and we see 4.1ms, which is good!