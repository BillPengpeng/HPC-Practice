本文主要整理Assignment 2 (systems): Systems and Parallelism的主要内容。

## 2.4 4D Parallelism

### 1. **五大基础并行方法**

| 并行方法 | 核心思想 | 关键特点 |
|---------|---------|---------|
| **数据并行(DP)** | 数据批次分割到不同设备 | 各设备计算局部梯度，需全局平均 |
| **全分片数据并行(FSDP)** | 优化器状态、梯度、权重分片 | 前向/反向传播前需收集权重分片 |
| **张量并行(TP)** | 激活值沿新维度分片 | 可按操作输入或输出维度分片 |
| **流水线并行(PP)** | 模型按层分阶段 | 不同阶段运行在不同设备 |
| **专家并行(EP)** | 专家模型中的专家分离 | 专用于MoE（专家混合）模型 |

### 2. **4D并行化框架构建**

**框架整合逻辑：**
- **FSDP + TP 组合** → 视为单一并行轴（互补性强）
- **保留DP** → 数据维度并行
- **保留PP** → 模型层次维度并行  
- **排除EP** → 聚焦稠密模型（非MoE）
- **最终得到4个并行轴**：DP、FSDP/TP、PP

### 3. **设备网格组织理念**

**核心概念：** 将计算集群抽象为**设备网格**(Device Mesh)，每个网格轴对应一个并行维度。

**实例说明：** 16个GPU可组织为4×4网格：
- **第一维(4)** → 数据并行(DP)
- **第二维(4)** → FSDP/TP组合并行

### 4. **技术选型指导**

**推荐组合：** FSDP与TP协同使用（权重与激活值沿对应维度分片）
**应用聚焦：** 主要针对**稠密模型**（非稀疏的MoE模型）
**规模适配：** 根据不同模型规模和硬件配置灵活调整并行策略

### 5. **学习资源指引**

- **理论基础**：TPU Scaling Book Part 5（通信与内存成本分析）
- **实践深入**：Ultra-Scale Playbook（流水线并行详解）
- **综合参考**：相关书籍提供的完整技术体系

### 技术价值与启示

该4D并行框架代表了当前大规模深度学习训练的**最先进范式**，通过多维度的并行策略组合，有效解决了单一模型在单个设备上无法容纳的挑战。这种分层并行的思想不仅适用于当前的Transformer架构，也为未来更大规模模型的高效训练提供了理论基础和实践路径。

### Problem (communication_accounting): 10 points

Consider a new model config, XXL, with d_model=16384, d_ff=53248, and num_blocks=126. Be-
cause for very large models, the vast majority of FLOPs are in the feedforward networks, we make
some simplifying assumptions. First, we omit attention, input embeddings, and output linear layers.
Then, we assume that each FFN is simply two linear layers (ignoring the activation function), where
the first has input size d_model and output size d_ff, and the second has input size d_ff and output
size d_model. Your model consists of num_blocks blocks of these two linear layers. Don’t do any
activation checkpointing, and keep your activations and gradient communications in BF16, while your
accumulated gradients, master weights and optimizer state should be in FP32.

- (a) How much memory would it take to store the master model weights, accumulated gradients and
optimizer states in FP32 on a single device? How much memory is saved for backward (these will
be in BF16)? How many H100 80GB GPUs worth of memory is this?

    - the master model weights, accumulated gradients and optimizer states: num_blocks*(2*d_model*d_ff)*(4 + 4 + K) = 126*2*16384*53248*(8+K) = 204.75GB * (K + 8), K = 16情况为3276GB， K = 20情况4095GB
    - saved for backward包括激化和梯度: num_blocks*L*(B*d_ff+B*d_model)*2 + num_blocks*(2*d_model*d_ff)*2 = L*B*16.73MB + 409.5GB
    => K = 16、B = 4、L = 1024情况，476.42GB

- (b) Now assume your master weights, optimizer state, gradients and half of your activations (in
practice every second layer) are sharded across $N_{FSDP}$ devices. Write an expression for how
much memory this would take per device. What value does $N_{FSDP}$ need to be for the total
memory cost to be less than 1 v5p TPU (95GB per device)?
    - 内存：num_blocks*(2*d_model*d_ff)*(4 + 4 + K) + num_blocks*L*B*d_model*2 = 204.75GB * (K + 8) + L*B*3.9375MB
    - (204.75GB * (K + 8) + L*B*3.9375MB) / $N_{FSDP}$ < 95GB => K = 16、B = 4、L = 1024情况，$N_{FSDP} = 4929.75 / 95 = 51.89$
    
- (c) Consider only the forward pass. Use the communication bandwidth of $W_{ici} = 2 · 9 · 10^{10}$ and
FLOPS/s of $C = 4.6 · 10^{14}$ for TPU v5p as given in the TPU Scaling Book. Following the
notation of the Scaling Book, use $M_X = 2$, $M_Y = 1$ (a 3D mesh), with X = 16 being your FSDP
dimension, and Y = 4 being your TP dimension. At what per-device batch size is this model
compute bound? What is the overall batch size in this setting?
    - $\alpha = C / $W_{ici} = (4.6 · 10^{14}) / (2 · 9 · 10^{10}) = 2556$
    - $$B/N > α² / (M_X * M_Y * F) =  2556^2 / (2 * 1* 53248) = 61.35$$

- (d) In practice, we want the overall batch size to be as small as possible, and we also always use our
compute effectively (in other words we want to never be communication bound). What other
tricks can we employ to reduce the batch size of our model but retain high throughput?
    - **梯度累积**：通过多次前向-反向传播累积梯度，再执行一次参数更新，从而在不增加单次迭代内存消耗的情况下，模拟大批次训练的效果。
    - **激活重计算/检查点**：主动丢弃部分中间结果（激活值），在需要时重新计算，以节省内存，从而允许在相同内存限制下使用更大的批次大小。
    - **优化并行策略配置**：更精细地调整FSDP、张量并行和流水线并行的组合与维度，以平衡计算、通信和内存开销。
    - **使用更高效的优化器或通信原语**：例如，采用通信量更小的优化器变种，或使用异步通信来重叠计算与通信。

## 3 Optimizer State Sharding

### 要点总结

1. **问题核心**
- 传统分布式数据并行训练要求每个计算节点存储完整的模型参数和优化器状态副本
- 以AdamW优化器为例，其内存占用达到模型权重本身的两倍（每个参数需维护两个浮点数）

2. **技术方案**
- 引用Rajbhandari等人（2020）研究提出三级分片策略：
  - 优化器状态分片
  - 梯度分片
  - 参数分片
- 本作业实现简化版方案：每个节点的优化器仅处理参数子集（约总参数量的1/world_size）

3. **实现机制**
- 优化步骤中各节点仅更新其分片参数
- 通过广播通信实现全局参数同步
- 通过分区管理避免冗余存储，显著降低单节点内存压力

4. **技术价值**
- 为解决大模型训练中的内存瓶颈提供有效路径
- 为后续FSDP等高级优化技术奠定理论基础
- 通过通信-计算权衡实现内存效率提升

### Problem (optimizer_state_sharding): 15 points

Implement a Python class to handle optimizer state sharding. The class should wrap an arbitrary in-
put PyTorch optim.Optimizer and take care of synchronizing updated parameters after each optimizer
step.

=> 完成

### Problem (optimizer_state_sharding_accounting): 5 points

(a) Create a script to profile the peak memory usage when training language models with and without
optimizer state sharding. Using the standard configuration (1 node, 2 GPUs, XL model size), report the peak memory usage after model initialization, directly before the optimizer step, and directly after the optimizer step. Do the results align with your expectations? Break down the memory usage in each setting (e.g., how much memory for parameters, how much for optimizer
states, etc.).

