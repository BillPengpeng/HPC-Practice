本文主要整理Part 2 TPUs (How to Think About TPUs) 的主要内容。

## 1. 前言

This section is all about how TPUs work, how they're networked together to enable multi-chip training and inference, and how this affects the performance of our favorite algorithms. 

## 2. What Is a TPU?

![TPUs](https://jax-ml.github.io/scaling-book/assets/img/tpu-chip.png)

A TPU is basically a compute core that specializes in matrix multiplication (called a TensorCore) attached to a stack of fast memory (called high-bandwidth memory or HBM).

1.  **专用计算核心**：TPU的核心是一个专门用于执行**矩阵乘法** 的硬件单元，这个单元被称为 **TensorCore**。这表明TPU并非通用处理器，而是为机器学习等需要大量矩阵运算的任务量身定制的。

2.  **高速内存集成**：这个专用的TensorCore直接连接到一个**高带宽内存（HBM）** 堆栈。这种设计是为了确保计算核心能够以极快的速度访问数据，从而避免因数据读取速度慢（内存墙）而影响整体计算效率。

## 2.0 TensorCore

### 1. **TensorCore 总体定位**
*   其核心是一个极其高效的**矩阵乘法机器**，专门为深度学习中的大规模矩阵运算设计。

### 2. **三大核心组件及功能**
TensorCore由三个各司其职的单元构成：

*   **MXU（矩阵乘法单元）**
    *   **功能**：TensorCore的**核心计算单元**，专门执行矩阵乘法。
    *   **性能**：性能极其强大。以TPU v5e为例，每个MXU峰值算力约为每秒5e13次半精度（f16）浮点运算。一个芯片通常有2个或4个MXU。
    *   **精度**：支持多种数据精度，精度越低，吞吐量越高（例如，int8精度下的算力可达f16精度的两倍）。

*   **VPU（矢量处理单元）**
    *   **功能**：执行**通用数学运算**，是MXU的补充。负责如ReLU等激活函数、向量间的逐点加法/乘法、以及求和等归约操作。

*   **VMEM（矢量内存）**
    *   **功能**：TensorCore内部的**高速片上内存**。
    *   **特点**：
        *   **容量小，速度快**：比片外HBM内存小得多（如128 MiB），但与MXU之间的带宽极高。
        *   **可编程控制**：类似于CPU缓存，但更大且可由程序员显式管理数据存放。**所有需要计算的数据必须先从HBM拷贝到VMEM中。**

### 3. **核心协作流程**
这张图片揭示了一个关键的工作流程：**HBM -> VMEM -> MXU/VPU**。这体现了“数据贴近计算”的设计原则，通过高速的VMEM为计算单元MXU持续喂料，以克服“内存墙”瓶颈，实现极高的计算效率。

### 一句话总结
**TensorCore是一个由专用矩阵计算单元（MXU）、通用矢量单元（VPU）和高速可编程片上内存（VMEM）协同工作的高效计算引擎，其设计核心是最大化矩阵乘法的吞吐量。**

## 2.1 TPU v5p

1.  **核心级性能：单核算力惊人**
    *   **数据**：每个TPU v5p**核心**每秒可进行 **2.5 × 10¹⁴** 次脑浮点数16（bfloat16）格式的浮点运算。
    *   **意义**：这一定量数据揭示了TPU最基础计算单元的原始速度。

2.  **芯片级与集群级性能：规模扩展带来顶级算力**
    *   **芯片层面**：每块**芯片**（内含多个核心）的算力翻倍，达到每秒 **5 × 10¹⁴** 次运算。
    *   **集群层面**：一个由**8960块芯片**组成的完整**TPU Pod**，其总算力高达每秒 **4 exaflops**。这是一个天文数字级别的计算能力。

3.  **核心结论：性能的全球定位与规模优势**
    *   **世界顶级**：文本明确指出，单个TPU v5p Pod的算力就堪称“全球最强大的超级计算机之一”。
    *   **规模庞大**：谷歌不仅拥有如此强大的硬件，更关键的是“拥有**大量**此类TPU”，这构成了其在人工智能基础设施领域的核心优势。

## 2.2 HBM

### 1. **HBM的核心功能与定位**
*   **角色**：HBM是TPU中用于存储**张量** 的**大容量、高速内存**。
*   **容量**：其容量通常在**数十GB**级别（例如，TPU v5e为16GiB），远大于TensorCore内部的VMEM。

### 2. **关键的数据流路径**
*   文字清晰地描述了一个核心工作流程：当需要进行计算时，数据（张量）的流动路径是 **HBM → VMEM → MXU**。
*   计算结果则通过 **VMEM 写回 HBM** 进行存储。
*   这一流程凸显了HBM作为**主要数据仓库**，而VMEM作为**高速数据中转站**的分工协作关系。

### 3. **HBM带宽的决定性影响**
*   **核心概念**：HBM与TensorCore（通过VMEM）之间的数据传输速率被称为 **“HBM带宽”**，其数值通常高达**每秒1-2TB**。
*   **性能瓶颈**：这个带宽是系统的一个关键性能指标。对于需要频繁与HBM交换数据的**内存受限型工作负载**，HBM带宽直接**限制**了计算任务能够达到的最快速度。

## 2.3 Generally, all TPU operations are pipelined and overlapped

![TPU example](https://jax-ml.github.io/scaling-book/assets/img/pointwise-product.gif)

1.  **基本工作流程：分块与搬运**
    TPU执行矩阵乘法等操作并非一次性处理全部数据。其标准流程是：将大数据从**HBM** 分批复制到高速缓存**VMEM**，再送入**MXU** 进行计算，最后将结果写回**HBM**。

2.  **核心优化设计：流水线与重叠**
    为了隐藏缓慢的内存访问延迟，TPU采用关键优化：**将“数据搬运”和“数学计算”这两个阶段重叠进行**。当MXU正在计算当前数据块时，VPU已经在为下一个数据块执行加载和存储操作。这种流水线设计确保了计算单元持续工作，避免因等待数据而空闲。

3.  **架构目标：保持“计算受限”**
    上述重叠操作的最终目的，是让矩阵乘法等核心运算始终处于**计算受限** 状态。这意味着MXU的计算能力被完全利用，其速度瓶颈在于自身算力，而非等待数据送达的速度。这是高性能计算的理想状态。

4.  **性能瓶颈：带宽限制**
    图片指出了TPU的根本性能边界。如果数据从HBM到VMEM的加载速度，慢于MXU或VPU消耗数据的速度，计算单元就会“饥饿”，系统则变为**内存/带宽受限**。因此，**HBM ↔ VMEM** 和 **VMEM ↔ MXU** 的带宽是决定TPU能高效执行何种计算的关键物理限制。

5.  **设计哲学：简单而专用**
    图片强调TPU的设计非常**简单直接**：其核心是一个强大的脉动阵列，专攻矩阵乘加运算。这种简化设计去除了通用处理器的复杂性，使其在特定任务上能实现极致的效率和吞吐量。

## 2.4 VMEM and arithmetic intensity

![VMEM and arithmetic intensity](https://jax-ml.github.io/scaling-book/assets/img/tpu-bandwidth.png)

### 1. VMEM的核心优势：超高带宽
*   VMEM虽然容量远小于HBM，但它与计算单元MXU之间的**数据传输带宽极高**，约为HBM带宽的**22倍**。
*   这意味着数据从VMEM到MXU的通信速度极快，能极大地缓解数据传输瓶颈。

### 2. 对计算性能的革命性影响
VMEM的高带宽特性带来了两个关键好处：
*   **降低“计算受限”的门槛**：对于矩阵乘法等操作，将权重等数据存放在VMEM中，意味着即使在**很小的批量大小** 下，计算也能达到**受算力限制（FLOPs bound）** 的理想状态，从而充分发挥MXU的强大算力。
*   **赋能低算术强度算法**：算术强度低（即计算操作与内存访问的比值低）的算法通常受限于内存带宽。而VMEM的高带宽使得这类算法只需**10-20**的算术强度即可实现峰值算力利用率，让更多类型的算法能在TPU上高效运行。

### 3. 主要的现实挑战：容量限制
*   尽管VMEM有巨大优势，但其**极小的容量**是实际应用中的主要挑战。将算法的所有输入/输出数据都放入VMEM中通常非常困难，需要精巧的数据管理和切分策略。

**一句话总结：TPU的VMEM通过其超高带宽，能够显著降低计算对算术强度的要求，使小批量计算和低算术强度算法也能高效运行，但其极小的容量是实际应用中需要克服的主要挑战。**

## 2.5 TPU Chip

![TPU Chip](https://jax-ml.github.io/scaling-book/assets/img/cores.png)
![PCIe](https://jax-ml.github.io/scaling-book/assets/img/pcie.png)

### 1. **芯片级架构：从多核独立到“超级核心”**
-   **演进**：自TPU v4起，一个芯片通常包含**两个TPU核心**，并且它们**共享内存**，可被视作一个算力翻倍的“超级核心”。
-   **对比**：旧版TPU（v3及更早）的芯片核心内存独立。为推理优化的芯片（如TPU v5e）每片仅有一个核心。

### 2. **系统级组织：芯片以“tray”形式集成**
-   在实际部署中，芯片以**4个为一组**安装在**tray**上。
-   每个tray通过**PCIe网络**与一个**主机CPU**连接。主机负责加载数据和执行控制程序。

### 3. **关键瓶颈：PCIe带宽限制**
-   PCIe连接（CPU ↔ TPU HBM）的**带宽非常有限**（例如TPU v4为16GB/秒）。
-   这个带宽比TPU内部的HBM带宽**慢近100倍**，成为数据在主机内存和TPU之间传输的**主要瓶颈**，限制了数据加载和卸载的速度。

## 3.0 Chips are connected to each other through the ICI network in a Pod

![Pod](https://jax-ml.github.io/scaling-book/assets/img/ici-wraparound.png)

### 1. **互联方式：芯片间直接高速链接**
*   ICI网络是TPU芯片之间的**直接硬件链接**，数据传输**不经过主机CPU**。
*   这种直连方式避免了主机总线瓶颈，为大规模分布式训练提供了极高的带宽和极低的延迟。

### 2. **拓扑结构演进：从2D环面到3D环面**
*   **老一代TPU**：如TPU v2/v3等，每个芯片连接**4个**最近邻，形成**2D环面** 网络。
*   **新一代TPU**：如TPU v4/v5p，每个芯片连接**6个**最近邻，形成更复杂的**3D环面** 网络。更高维度的连接提升了网络的整体带宽和效率。

### 3. **核心优势：优化大规模集群通信效率**
*   **环面结构** 的关键优势在于，它将网络中任意两个节点之间的**最大通信距离**从N降低到大约N/2。
*   部分TPU还采用 **“扭曲环面”**  配置，类似于莫比乌斯带拓扑，能进一步减少节点间的**平均距离**。
*   这些设计共同确保了即使在由数千个芯片组成的庞大Pod中，通信也能保持高效，这是支撑大规模模型并行和数据并行训练的基础。

**一句话总结：TPU通过ICI网络将芯片直接连接成环面拓扑结构，这种设计通过极大优化节点间的通信距离，为万卡级别的超级集群提供了高效通信的基础能力。**

## 3.1 TPU pods (connected by ICI) can get really big

### 1. **Pod的规模与基础构建块**
*   **超大规模**：TPU Pod的规模可以极其庞大。例如，TPU v4的最大Pod为16x16x16（4096芯片），而v5p可达16x20x28（8960芯片）。
*   **基础单元**：这些大型Pod由更小的、可重新配置的**4x4x4芯片立方体**（对应一个物理机架）作为基本构建块，通过**光学环绕链路** 连接而成。

### 2. **拓扑配置的灵活性与性能关键点**
*   **可配置性**：用户可以根据计算需求请求不同大小的拓扑结构（如2x2x2, 4x4x8等）。
*   **性能关键：环绕功能**：
    *   只有请求的拓扑在某个维度上是**4的倍数**（对于v4/v5p）或**16的倍数**（对于v5e）时，系统才能通过光开关提供**环绕** 连接。
    *   **不具备环绕功能的较小拓扑**（如2x2x1），其节点间的最远距离会增加，导致**大多数通信操作的时间几乎翻倍**，这是影响性能的重要注意事项。

### 3. **不同代际的架构差异**
*   **TPU v4/v5p**：采用**3D环面**拓扑，规模可扩展性极强。
*   **TPU v5e/Trillium**：采用规模较小的**16x16的2D环面**拓扑，无法像v4/v5p那样扩展成3D超级Pod。不过，多个Pod之间可以通过标准数据中心网络进行通信。

**一句话总结：TPU Pod通过光学链路将基础计算立方体组合成超大规模3D环面网络，其拓扑可灵活配置，但能否在配置维度上实现“环绕”功能，是决定集群内通信效率的关键因素。**

## 3.2 This nearest-neighbor connectivity is a key difference between TPUs and GPUs

### 1. **核心架构：本地连接 vs. 分层交换**
*   **TPU**：采用**本地连接**，即每个芯片只与物理上相邻的少数几个邻居直接相连，形成环面（2D/3D Torus）拓扑。
*   **GPU**：采用**分层交换机**，通过多级交换机网络（如NVLink Switch）来近似实现所有GPU之间的点对点连接。

### 2. **性能与规模特性：常数跳数 vs. 对数跳数**
*   **TPU**：在由其拓扑结构决定的N个节点的网络中，任意两个节点间的最大通信跳数是一个**常数**（约为N/2）。
*   **GPU**：在由交换机组成的层次化网络中，任意两个GPU之间的通信跳数随着集群规模（N）增大而增加，复杂度为 **O(log(N))**。

### 3. **综合优劣对比**
| 特性 | TPU（最近邻连接） | GPU（分层交换） |
| :--- | :--- | :--- |
| **优势** | **成本显著更低**（无需昂贵交换机）、**布线更简单**、能够**扩展到更庞大的拓扑**规模（因为每个设备的链路数和带宽是常数）。 | 在一个节点内（如8个H100或72个B200），GPU可以实现**高速直接互联**，任意两点间通信延迟很低。 |
| **劣势** | 数据在远距离节点间传输需要经过多跳，延迟较高。 | **交换机非常昂贵**、布线复杂，**大规模扩展性受限**（因为交换网络的成本和复杂性会急剧上升）。 |

**一句话总结：TPU为超大规模扩展而设计，采用简单、廉价的最近邻连接，牺牲了单跳延迟换取了极致的可扩展性和成本效益；而GPU则在有限规模内追求极高的任意点通信性能，通过昂贵复杂的分层交换网络实现，但大规模扩展性是其短板。**

## 3.3 ICI is very fast relative to DCN, but is still slower than HBM bandwidth

- 2.5e12 bytes/s (2.5 TB/s) of HBM bandwidth per chip.
- 9e10 bytes/s (90 GB/s) of ICI bandwidth per axis, with 3 axes per chip. 
- 6.25e9 bytes/s (6.25 GB/s) of DCN (egress) bandwidth per TPU (via 1-2 NICs on each host). 

## 3.4 Multi-slice training

### 1. **核心概念：切片**
*   一个**切片** 是指一组通过**高速芯片互连（ICI）** 网络直接通信的TPU集合。这是进行高效分布式训练的基本单位。

### 2. **多切片架构：通过DCN连接**
*   **多切片训练** 是指将多个独立的切片通过**数据中心网络（DCN）** 连接起来，共同完成一个训练任务。例如，可以连接位于不同物理机柜（Pod）上的切片。

### 3. **核心瓶颈：DCN的严重延迟**
*   DCN的带宽和速度**远低于** ICI。因此，在设计多切片训练任务时，必须尽量减少计算过程等待从DCN传输过来的数据的时间，否则慢速的DCN将成为整个训练的瓶颈。

### 4. **复杂的传输路径**
*   由于DCN是主机到主机的网络，TPU之间的数据通过DCN传输需要经过一条漫长而复杂的路径：
    **TPU内存 → PCIe总线 → 主机内存 → 网络出口 → 网络入口 → 目标主机内存 → PCIe总线 → 目标TPU内存**。
    这条路径进一步增加了通信开销和延迟。

**一句话总结：多切片训练通过DCN连接多个ICI切片以扩大算力规模，但DCN的低速和复杂的数据路径使其成为严重的性能瓶颈，需在算法设计上极力避免计算过程依赖DCN的数据传输。**

## 4.0 Key Takeaways

### 1. **基本设计哲学：简单与专用**
TPU可被简化为一个**高效的矩阵乘法单元**，其通过与不同层级存储/设备的连接来工作：连接自身内存（极快）、连接邻近芯片（较快）、连接数据中心网络（尚可）。

### 2. **通信性能层级：四级带宽瓶颈**
TPU的通信速度受限于一个严格的带宽层次结构，按速度从高到低排列为：
*   **HBM带宽**：芯片访问自身高频宽内存的速度，最快。
*   **ICI带宽**：与最近邻TPU芯片直连通信的速度。
*   **PCIe带宽**：TPU与主机CPU之间通信的速度，较慢。
*   **DCN带宽**：跨不同主机（切片间）通信的速度，最慢。

### 3. **切片内连接：最近邻与多跳通信**
在一个切片内，TPU芯片**仅通过ICI与物理上最近的几个邻居直接相连**。因此，非相邻芯片间的通信需要经过中间芯片的多次转发（“跳”），这会增加延迟。

### 4. **硬件计算约束：矩阵填充要求**
为了完全利用MXU的计算能力，参与矩阵乘法的权重矩阵在两个维度上的大小**必须填充（Padding）至至少128**（TPU v6为256）。较小的维度会被自动填充以满足此要求。

### 5. **性能优化技巧：低精度计算**
在支持的代际上，使用更低精度的数据类型（如int8, int4）进行矩阵乘法，其峰值算力远高于bfloat16（约2倍/4倍）。但需注意，矢量单元上的操作通常仍使用fp32精度。

### 6. **核心设计原则：通信与带宽匹配**
为避免TPU强大的计算单元因等待数据而空闲，必须确保跨越上述每个通信通道的数据量与其通道的速度成正比。即，应尽量减少慢速通道（如DCN）上的数据交换。

## 4.1 TPU Specs

1.  **性能代际演进**：从v3到最新的v6e，TPU在**算力（FLOPs/s）**、**HBM容量与带宽** 以及**ICI互联带宽** 上均呈现显著提升，尤其是v6e相比前几代有跨越式增长。
2.  **架构差异**：v4p和v5p采用规模更大的**3D Pod**（16x16x16, 16x20x28），而v5e和v6e采用**2D Pod**（16x16），这体现了针对不同计算规模的设计侧重。
3.  **关键瓶颈数值**：图片明确了两个关键外部接口的带宽限制：**PCIe带宽**（约16-32 GB/s/TPU）和更低的**DCN带宽**（约3-12.5 GB/s/TPU），这些是分布式训练中需要极力避免的通信瓶颈。
4.  **精度与算力**：所有型号都支持多种计算精度（如bf16, int8）。对于int8运算，其峰值算力通常是bf16的两倍，这为低精度推理和高效率训练提供了可能。

**表1: TPU 核心规格**
| 型号 | Pod大小 | 主机大小 | HBM容量/芯片 | HBM带宽/芯片 (bytes/s) | FLOPs/s/芯片 (bf16) | FLOPs/s/芯片 (int8) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TPU v3 | 32x32 | 4x2 | 32 GB | 9.0e11 | 1.4e14 | 1.4e14 |
| TPU v4p | 16x16x16 | 2x2x1 | 32 GB | 1.2e12 | 2.75e14 | 2.75e14 |
| TPU v5p | 16x20x28 | 2x2x1 | 96 GB | 2.8e12 | 4.59e14 | 9.18e14 |
| TPU v5e | 16x16 | 4x2 | 16 GB | 8.1e11 | 1.97e14 | 3.94e14 |
| TPU v6e | 16x16 | 4x2 | 32 GB | 1.6e12 | 9.20e14 | 1.84e15 |

**表2: TPU 互联带宽 (ICI)**
| 型号 | ICI带宽/链路（单向, bytes/s） | ICI带宽/链路（双向, bytes/s） |
| :--- | :--- | :--- |
| TPU v3 | 1e11 | 2e11 |
| TPU v4p | 4.5e110 | 9.0e10 |
| TPU v5p | 9.0e10 | 1.8e11 |
| TPU v5e | 4.5e10 | 9.0e1 |
| TPU v6e | 9e10 | 1.8e11 |

**补充说明文本：**
*   **主机大小** 指的是连接到单个主机的TPU拓扑结构（例如，TPU v5e 有一个CPU主机连接到8个TPU，拓扑为4x2）。
*   同时列出**单向**和**双向**带宽是因为单向带宽更贴近硬件本质，但在涉及完整环的方程中，双向带宽出现得更频繁。
*   **PCIe带宽** 通常约为每TPU 1.6e10 bytes/s（TPU v6e 为 3.2e10）。
*   **DCN带宽** 通常约为每TPU 6.25e9 bytes/s（TPU v6e 为 12.5e9，TPU v5e 为 3.125e9）。

## 5. Worked Problems

### Question 1 [bounding LLM latency]

Say you want to sample from a 200B parameter model in bf16 that’s split across 32 TPU v4p. How long would it take to load all the parameters from HBM into the systolic array?

- We’re loading sizeof(bf16) * 200e9 = 400e9 bytes on 32 chips, meaning 12.5e9 bytes / chip, each with an HBM bandwidth of 1.23e12. So the load takes around 10ms.

### Question 2 [TPU details]

Consider a full TPU v5e pod. How many total CPU hosts are there? How many TPU TensorCores? What is the total FLOPs/s for the whole pod? What is the total HBM? Do the same exercise for TPU v5p pod.

- For TPU v5e, each pod is 16x16 and each host is a 4x2 slice, so we have 16*16 / 8 = 32 hosts. For TPU v5e, **each TPU has only one core**, so we have 256 TensorCores. The total FLOPs/s is 16*16*2e14 = 5.1e16 in bfloat16. Each chip has 16GB of HBM, so that 256 * 16 = 4TB of memory.
- For a full TPU v5p pod, we have 16x20x28 chips and each host is 2x2x1, so we have 16*20*28 / 2*2 = 2,240 hosts. For TPU v5p, **each TPU has two TensorCores**, so we have 8960 * 2 = 17,920 cores. The total FLOPs/s is 8960 * 4.5e14 = 4e18 in bfloat16. Each chip has 96GB of HBM, so that’s 8960 * 96 = 860TB of memory.

### Question 3 [PCIe operational intensity]

Imagine we’re forced to store a big weight matrix A of type bfloat16[D,F] , and a batch of activations x of type bfloat16[B,D] in host DRAM and want to do a matrix multiplication on them. This is running on a single host, and we’re using a single TPU v6e chip attached to it. You can assume B≪DB≪D , and F=4DF=4D (we’ll see in future chapters why these are reasonable assumptions). What is the smallest batch size BB we need to remain FLOPs bound over PCIe? Assume PCIe bandwidth of 1.5e10 bytes / second.

- We have to perform 2BDF2BDF floating point operations, and each chip can perform 9.2e14 floating point operations per second. This then requires 2BDF/9.2e14 seconds to perform. We have to load 2DF+2BD bytes from DRAM, and write 2BF2BF bytes back to it. We are bottlenecked by PCIe transfer speeds, so we need 2⋅(BD+DF+BF)/1.5e10 seconds to transfer data to and from the TPU. Since we want computation to take longer than weight loading, assuming we can overlap all weight loading with computation, we want $$2BDF/9.2e14>2⋅(BD+DF+BF)/1.5e10$$

### Question 4 [general matmul latency]

Let’s say we want to multiply a weight matrix int8[16384, 4096] by an activation matrix of size int8[B, 4096] where B is some unknown batch size. Let’s say we’re on 1 TPUv5e to start.
- How long will this multiplication take as a function of B? Hint: it may help to calculate how long it will take to load the arrays from HBM and how long the multiplication will actually take. Which is bottlenecking you?
- What if we wanted to run this operation out of VMEM? How long would it take as a function of B?

### Question 5 [ICI bandwidth]

Let’s say we have a TPU v5e 4x4 slice. Let’s say we want to send an array of type bfloat16[8, 128, 8192] from TPU{0,0} to TPU{3, 3}. Let’s say the per-hop latency for TPU v5e is 1μs .
- How soon will the first byte arrive at its destination?
- How long will the total transfer take?

- In a TPUv5e we have 2D connectivity. Because we have only a 4x4 slice (with no axes of size 16), we have no wraparound connections. Thus there are two ports from which our target chip can receive data, and likewise two ports from which our source chip can send data. The amount of data we have to transfer is 2 * 8 * 128 * 8192 = 1.7e7 bytes. We can transfer from both ports simultaneously (i.e. send half the array right and half down), so we get 2 * 4.5e10 = 9e10 bytes transferred per second, which means it’ll take about 1.7e7 / 9e10 = 188us to transfer the whole array through (assuming we’re bandwidth bound). In a 4x4 slice, we have six hops between chips (0,0)(0,0) and (3,3)(3,3) , since there are no wraparound links for axes with fewer than 16 chips. **Since the latency of each hop is about 1μs, the first byte will arrive in about 6us and the total transfer will take 188us**.

### Question 6 [pulling it all together, hard]

Imagine you have a big matrix A: int8[128 * 1024, 128 * 1024] sharded evenly across a TPU v5e 4x4 slice but offloaded to host DRAM on each chip. Let’s say you want to copy the entire array to TPU{0, 0} and multiply it by a vector bf16[8, 128 * 1024]. How long will this take? Hint: use the numbers above.

![a TPU v5e host has a 4x2 topology](https://jax-ml.github.io/scaling-book/assets/img/challenge-problem.png)

