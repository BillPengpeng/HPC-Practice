本文主要整理Part 12 GPUs (How to Think About GPUs) 的主要内容。

## 7.0 Networking

![Networking](https://jax-ml.github.io/scaling-book/assets/gpu/superpod-diagram.png)

### 内容概况

核心内容是**对比谷歌TPU与NVIDIA GPU在构建大规模计算集群时，所采用的网络互连技术的根本性差异**。文章指出，TPU使用简单的**2D或3D环面**拓扑，而GPU采用基于交换机的、更传统的**分层树形**网络。图片通过一个典型的NVIDIA H100 GPU集群网络拓扑图，详细展示了GPU如何通过**NVLink**在单个“节点”内实现高速互联，再通过**InfiniBand**网络将多个节点连接成大规模集群。这种设计在灵活性和通用性上占优，但与TPU的环面网络相比，在极大规模下的带宽可扩展性方面面临不同挑战。

---

### 要点总结

1.  **根本差异：拓扑结构**
    *   **TPU（环面拓扑）**：芯片像网格上的点，每个TPU仅与相邻的TPU直接连接。通信需经过中间所有TPU接力。优点是链接数量固定，可以扩展到任意大的“Pod”而不会损失带宽。
    *   **GPU（分层树形拓扑）**：采用层级化的交换机网络（如“胖树”结构），提供更高的连接灵活性。

2.  **GPU的两级网络架构**
    *   **第一级：节点内高速互联（NVLink域）**
        *   **组成**：通常由**8个GPU**构成一个“节点”。
        *   **技术**：通过**NVLink交换机**实现全互联，带宽极高（每个H100在节点内拥有约450GB/s的出口带宽）。
        *   **目的**：为紧密协作的GPU任务（如模型并行）提供超低延迟和高带宽通道。
    *   **第二级：节点间扩展互联（InfiniBand/Ethernet）**
        *   **技术**：各个节点通过**InfiniBand交换机**连接成大规模集群。
        *   **带宽**：每个节点到IB网络的总出口带宽较低（约400GB/s）。
        *   **架构**：通常采用“胖树”结构，旨在提供全二分带宽。

3.  **设计哲学与权衡**
    *   **TPU**：为特定的、规整的通信模式（如All-Reduce）优化，追求极大规模下的可预测带宽。但通信模式受限。
    *   **GPU**：提供更通用、灵活的网络，支持任意通信模式。但构建极大规模、无带宽损失的网络成本和复杂度更高。


## 7.1 At the node level

![NVLink](https://jax-ml.github.io/scaling-book/assets/gpu/nvlink-nodes.png)

1.  **演进目标：追求极致的节点内通信带宽**
    *   核心目标是实现GPU间的**全互联**、**高带宽**和**低延迟**通信，以支撑大规模AI训练和HPC应用中的密集数据交换（如All-Reduce操作）。

2.  **代际演进亮点：**
    *   **Volta (V100) / Ampere (A100)**：引入NVSwitch，实现了节点内所有GPU的任意对任意直接通信，告别了P100时代的环状网络限制。
    *   **Hopper (H100)**：NVLink 4.0单链路带宽保持25GB/s，但通过将每个GPU的NVLink端口数增至18个，使节点内GPU到GPU带宽提升至450GB/s。节点拓扑稳定为8 GPU + 4个NVSwitch。
    *   **Blackwell (B200)**：实现巨大飞跃，NVLink 5.0单链路带宽翻倍至50GB/s，节点内带宽随之跃升至900GB/s。并首次支持**超大规模NVLink域**，推出集成72个GPU的节点（GB200 NVL72），需18个NVSwitch。

3.  **分工协作模式：NVLink为路，NVSwitch为枢纽**
    *   **NVLink**：是**点对点的高速通信车道**。它负责在GPU与GPU之间，以及GPU与NVSwitch之间建立直接、高速的数据传输通道。
    *   **NVSwitch**：是**节点内部的交通枢纽或交换中心**。它作为一个智能交换机，连接节点内的所有GPU，负责在任意两个GPU之间高效地路由数据包，从而实现全互联网络。

| NVLink 世代 | NVSwitch 世代 | GPU 架构 | NVLink 带宽 (GB/s, 全双工) | NVLink 端口数 / GPU | 节点内 GPU 到 GPU 带宽 (GB/s, 全双工) | 节点规模 (NVLink 域) | 每节点 NVSwitch 数量 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 3.0 | 2.0 | Ampere (A100) | 25 | 12 | 300 | 8 | 6 |
| 4.0 | 3.0 | Hopper (H100) | 25 | 18 | 450 | 8 | 4 |
| 5.0 | 4.0 | Blackwell (B200) | 50 | 18 | 900 | 8 / 72 | 2 / 18 |

4.  **“NVLink Bandwidth”指的是单条链路的带宽**
    *   这个数值（例如Blackwell的50 GB/s）指的是**单一条NVLink物理链路**在全双工模式下的峰值带宽。可以把它想象成一条单车道的最高限速。

5.  **“Node GPU to GPU Bandwidth”指的是聚合带宽**
    *   这个数值（例如Blackwell的900 GB/s）指的是**一个GPU能够同时使用其所有的NVLink端口与其他GPU通信时，所能达到的总带宽上限**。它衡量的是GPU的整体通信能力。
    *   **计算方式**：对于Blackwell GPU，它有18个NVLink端口，每个端口带宽为50 GB/s。那么理论上，当所有这些端口都被用于对外通信时，它的总出口带宽就是 `18 ports * 50 GB/s/port = 900 GB/s`。

## 7.2 Quiz 2: GPU nodes

### Question 1 [Total bandwidth for H100 node]

How much total bandwidth do we have per node in an 8xH100 node with 4 switches? Hint: consider both the NVLink and NVSwitch bandwidth.

- We have Gen4 4xNVSwitches, each with 64 * 25e9=1.6TB/s of unidirectional bandwidth. That would give us **4 * 1.6e12=6.4e12** bandwidth at the switch level. However, note that each GPU can only handle 450GB/s of unidirectional bandwidth, so that means we have at most **450e9 * 8 = 3.6TB/s** bandwidth. Since this is smaller, the peak bandwidth is 3.6TB/s.

### Question 2 [Bisection bandwidth]
 
Bisection bandwidth is defined as the smallest bandwidth available between any even partition of a network. In other words, if split a network into two equal halves, how much bandwidth crosses between the two halves? Can you calculate the bisection bandwidth of an 8x H100 node? Hint: bisection bandwidth typically includes flow in both directions.

- Any even partition will have 4 GPUs in each half, each of which can egress 4 * 450GB/s to the other half. Taking flow in both directions, this gives us **8 * 450GB/s** of bytes cross the partition, or 3.6TB/s of bisection bandwidth. This is what NVIDIA reports e.g. here.
- 这指的是全双工通信。在现实中，网络通信很少是单向的。当一半的GPU向另一半发送数据时，另一半月可能也在同时回复数据。因此，计算总带宽时需要把A到B和B到A的流量都算上。

### Question 3 [AllGather cost]

Given an array of B bytes, how long would a (throughput-bound) AllGather take on an 8xH100 node? Do the math for bf16[DX, F] where D=4096, F=65,536.

- For the given array, we have B=4096 * 65536 * 2=512MB, so the total time is **536e6 * (8 - 1) / 3.6e12 = 1.04ms**. This could be latency-bound, so it may take longer than this in practice (in practice it takes about 1.5ms).

## 8.0 Beyond the node level

![Beyond the node level](https://jax-ml.github.io/scaling-book/assets/gpu/h100-superpod.png)

1.  **设计目标与规模**：
    *   **核心目标**：实现远超单个节点规模的GPU集群互联，为大规模AI训练和HPC应用提供基础设施。
    *   **系统规模**：图示为一个**1024个H100 GPU**的集群，由**128个计算节点**构成（每个节点包含8个H100 GPU）。

2.  **层级化网络架构**：
    *   **计算节点层**：系统的最底层是128个“8xH100节点”。每个节点是一个通过NVSwitch实现内部全互联的基础单元。
    *   **叶交换机层**：多个计算节点上行连接到**叶交换机**。叶交换机是节点接入大规模网络的第一跳。
    *   **脊交换机层**：**脊交换机**位于网络顶层，负责连接不同的叶交换机，从而实现所有节点之间的任意互连。
    *   **可扩展单元**：32个节点（256个GPU）被定义为一个“可扩展单元”，这可能是集群管理和部署的基本模块。

3.  **关键技术特性**：
    *   **互联技术**：节点间的规模扩展采用**InfiniBand**技术，图中标注了NDR（400 Gbps和200 Gbps）等高速链路。
    *   **网络性能**：该“叶-脊”网络架构被设计为能够提供**全二分带宽**，这意味着无论数据如何在节点间流动，网络都能提供近乎一致的、无阻塞的高带宽性能，这是保证分布式计算效率的关键。

## 8.1 Scalable Units

1.  **核心定义：集群的“基础乐高模块”**
    *   **可扩展单元** 是构建更大规模集群的基本组成单位。一个SU包含32个节点，总计256个GPU。
    *   这种模块化设计简化了大规模集群的部署、管理和扩展。

2.  **网络架构：清晰的层级关系**
    *   **节点内网络**：每个计算节点内部通过4个**NVSwitch** 连接8个GPU，形成高速的NVLink域。
    *   **单元内网络**：整个SU的32个节点，由一组**8台InfiniBand叶交换机** 进行统一互联和管理。

3.  **技术规格：高速互联保障**
    *   **互联标准**：全部采用**InfiniBand NDR**，线缆和交换机端口速率均为**50 GB/s（全双工）**。
    *   **交换设备**：使用**64端口的NDR InfiniBand交换机**。

4.  **关键洞见：带宽层级设计**
    *   图片特别强调了一个重要事实：**InfiniBand叶交换机的总带宽是节点内NVSwitch的两倍**。
    *   **意义解读**：这样的设计确保了**节点间通信的带宽足以支撑节点内GPU的全速通信需求**，避免了网络成为整个系统性能的瓶颈。当256个GPU在SU内协同工作时，数据在节点间流动的管道足够宽，不会限制计算性能的发挥。

## 8.2 SuperPod

1.  **模块化构建**：SuperPod的核心设计思想是**模块化**。它以之前定义的**可扩展单元（256个GPU）** 为基本构建块，通过连接4个SU，快速、标准地构建出包含**1024个GPU**的大规模集群。

2.  **三层网络架构**：系统呈现出清晰的三层网络拓扑(8 * 32 * 4)：
    *   **节点层**：512个**节点级NVSwitch**，负责每个节点内8个GPU的高速互联。
    *   **接入层**：32个**Leaf IB交换机**，每个SU配备8台，负责连接该SU内的32个计算节点。8 * 4
    *   **核心层**：16个**Spine IB交换机**，作为网络骨干，所有Leaf交换机均与所有Spine交换机全互联，确保任意节点间的高效通信。

3.  **规模与设备统计**：图片提供了精确的设备数量，总计达**560个交换机**。这凸显了构建如此大规模集群所需的巨大互联复杂度。

4.  **关键连接方式**：
    *   **单元内**：Leaf交换机以32个节点为一组进行连接和管理。
    *   **单元间**：采用“全互联”模式，即所有Leaf交换机都上连到所有Spine交换机。这种设计通常旨在提供**全二分带宽**，避免通信瓶颈

## 8.3 How much bandwidth do we have? 

1.  **核心网络架构：胖树拓扑与全二分带宽**
    *   **胖树拓扑**：InfiniBand网络采用类似传统数据中心的“叶-脊”结构，这种设计旨在消除网络瓶颈。
    *   **全二分带宽**：该设计的核心优势是，当将整个集群的节点对半划分时，两个分区之间可以同时以最大带宽进行通信。这意味着集群在进行大规模数据交换时不会出现拥堵。

2.  **关键带宽数据对比**
    *   **节点内 vs. 节点外**：H100节点内部通过NVLink实现的GPU到GPU带宽高达**450 GB/s**。而一旦通信需要跨越节点，带宽则降至**400 GB/s**（节点到节点）。这个差异对分布式训练的通信效率至关重要。
    *   **带宽层级**：表格显示，从Node（节点）到Leaf（叶）再到Spine（脊）层级，总带宽（Bandwidth per Unit）在增加，但每个GPU能享用的跨节点带宽（Fat Tree Bandwidth）稳定在400 GB/s。

3.  **与TPU v5p的对比**
    *   **GPU（InfiniBand）**：网络灵活，支持任意通信模式，可通过增加交换机层级扩展到任意规模，但代价是增加延迟和昂贵的网络设备成本。
    *   **TPU（3D环面）**：每个TPU链路的带宽约为90 GB/s，整个3D环面可提供高达540 GB/s的出口带宽。其优势在于可以低成本、低延迟地扩展到极大规模（如8960个TPU），但通信模式受限，依赖于规整的、均匀的通信模式（如环面缩减）。

4.  **核心结论**
    *   文末的“Takeaway”强调：**H100节点内部的超高带宽（450 GB/s）与节点间的稍低带宽（400 GB/s）之间的差异，将是决定通信原语性能的关键因素。** 优化分布式训练时必须考虑这一瓶颈。

| Level | GPUs | Switches per Unit | Switch Type | Bandwidth per Unit (TB/s, full-duplex) | GPU-to-GPU Bandwidth (GB/s, full-duplex) | Fat Tree Bandwidth (GB/s, full-duplex) |
| :---- | :--- | :---------------- | :---------- | :------------------------------------- | :--------------------------------------- | :------------------------------------ |
| **Node** | 8    | 4                 | NVL         | 3.6                                    | 450                                      | 450                                    |
| **Leaf** | 256  | 8                 | IB          | 12.8                                   | 50                                       | 400                                    |
| **Spine** | 1024 | 16                | IB          | 51.2                                   | 50                                       | 400                                    |

**表格说明：**
*   **Level**： 网络层级（节点、叶交换机、脊交换机）。
*   **GPUs**： 该层级一个“单元”所覆盖的GPU总数。
*   **Switches per Unit**： 每个“单元”所需的交换机数量。
*   **Switch Type**： 交换机类型（NVL = NVSwitch, IB = InfiniBand Switch）。
*   **Bandwidth per Unit**： 整个“单元”的总带宽（太字节每秒）。
*   **GPU-to-GPU Bandwidth**： 在此层级下，GPU到GPU通信的带宽。在Node级，这就是NVLink带宽；在Leaf/Spine级，这是单个InfiniBand链路的带宽。
*   **Fat Tree Bandwidth**： 在胖树网络保证下，每个GPU能获得的、稳定的跨节点通信带宽。

## 8.4 GB200 NVL72s

![GB200 NVL72s](https://jax-ml.github.io/scaling-book/assets/gpu/gb200-superpod.png)

1.  **革命性的节点规模：单节点72个GPU**
    *   GB200 NVL72将一个集群的“节点”定义从传统的**8个GPU**（H100/B200）提升至**72个GPU**。这72个GPU通过NVLink全互联，内部带宽高达900 GB/s。

2.  **颠覆性的带宽特性：节点出口带宽远超内部带宽**
    *   这是最关键的突破。在传统架构（H100/B200）中，跨节点通信带宽（400 GB/s）低于节点内带宽（450/900 GB/s），因此优化重点在于避免跨节点通信。
    *   而在GB200 NVL72中，**单个节点的对外总带宽高达3.6 TB/s**（通过18条400 Gbps的InfiniBand链路实现）。这意味着，**对于这72个GPU组成的“超级节点”来说，其访问外部网络的速度可能比节点内部某些GPU间的通信路径更快**。这将彻底改变分布式训练算法的优化策略。

3.  **系统级扩展：构建更大规模的SuperPod**
    *   基于这种“超级节点”，可以构建规模极大的集群。图示为一个由8个GB200 NVL72节点（共576个GPU）组成的SuperPod，并通过9台脊柱交换机提供成比例提升的网络带宽，以支撑庞大的节点间数据流。

4.  **性能模型的根本改变**
    *   总结点明，GB200 NVL72**显著提升了性能模型的rooflines**。其巨大的节点规模和极高的节点出口带宽，使得通信瓶颈发生了转移，为万卡级别的AI大模型训练提供了新的硬件基础。

| 节点类型 | 每节点GPU数量 | GPU出口带宽 | 节点出口带宽 |
| :--- | :--- | :--- | :--- |
| **H100** | 8 | 450 GB/s | 400 GB/s |
| **B200** | 8 | 900 GB/s | 400 GB/s |
| **GB200 NVL72** | **72** | 900 GB/s | **3,600 GB/s (3.6 TB/s)** |

**表格解读：**
*   **GPU出口带宽**：指单个GPU能够访问节点外部网络的带宽。B200和GB200中的GPU均受益于NVLink 5.0，故带宽相同。
*   **节点出口带宽**：指整个节点（包含所有GPU）对外部网络的总带宽。GB200 NVL72的节点出口带宽是H100节点的**9倍**，这与它集成的GPU数量增长倍数一致。

## 8.5 Quiz 3: Beyond the node level

### Question 1 [Fat tree topology]

Using the DGX H100 diagram above, calculate the bisection bandwidth of the entire 1024 GPU pod at the node level. Show that the bandwidth of each link is chosen to ensure full bisection bandwidth.

- First, each node has 8x400Gbps NDR IB cables connecting it to the leaf switches, giving each node **400 GB/s** of bandwidth to the leaf. We have 8 leaf switches with 3.2TB/s each (64 400 GBps links), but we can only use 32 of the 64 ports to ingress from the SU, so that’s **32 * 400 = 12.8TB/s** for 32 nodes, again exactly 400GB/s.
- Then at the spine level we have 8 * 32 400Gbps NDR IB cables connecting each SU to the spine, giving each SU **8 * 16 * 2 * 400 / 8 = 12.8 TB/s** of bandwidth to the leaf. Again, this is 400GB/s per node. We have 16 spine switches, each with 3.2TB/s, giving us **16 * 3.2 = 51.2 TB/s**, which with 128 nodes is again 400GB/s.
- * 2是关键。它表示为了实现全互联无阻塞的Fat Tree拓扑，这些连接被设计成“每台Leaf交换机连接到2组不同的Spine交换机”或等效的链路聚合。这个“2”是一个拓扑乘数，表示实际使用的链路数量是 8 * 16的2倍。
- 无论我们以何种方式将节点集群划分为两半，GPU之间的通信带宽都将稳定在400 GB/s。网络的每一个组件都拥有恰好足够的带宽来确保这是一个真正的“胖树”结构。

### Question 2 [Scaling to a larger DGX pod]

Say we wanted to train on 2048 GPUs instead of 1024. What would be the simplest/best way to modify the above DGX topology to handle this? What about 4096? 

**1. 扩展到2048个GPU的方案：**

*   **核心策略**：**增加模块数量，并相应倍增顶层交换设备**。
    *   保持基础模块——**可扩展单元（SU）** 的结构不变（即每个SU仍包含32个节点/256个GPU，由8台叶子交换机管理）。
    *   要容纳2048个GPU，需要 `2048 / 256 = 8` 个SU。
    *   相应地，脊柱交换机的数量也需要倍增，例如配置32台脊柱交换机来连接这8个SU，以确保足够的带宽。
*   **一个关键优化**：为了节省叶子交换机的端口资源，可以将每个到脊柱交换机的连接从2条400Gbps线缆改为1条。**因为单条400Gbps NDR线缆已是全双工，总带宽不变**，但此举释放了宝贵的端口用于连接更多SU。

**2. 扩展到4096个GPU的方案：**

*   **核心挑战**：**设备端口数量达到上限**。即使进行上述优化，叶子交换机和脊柱交换机的端口资源也最终会耗尽，无法直接扩展。
*   **解决方案**：**引入新的网络层级**。
    *   在脊柱交换机之上，再增加一层 **“核心交换机”**。
    *   具体架构：构建一个由128台脊柱交换机和64台核心交换机组成的网络，来管理和连接16个SU（`4096 / 256 = 16`），从而支撑起4096个GPU的庞大集群。
    *   这种“叶子-脊柱-核心”的三层架构是典型的超大规模数据中心网络设计，它通过增加层级来解决大规模设备互联的寻址和带宽问题。

