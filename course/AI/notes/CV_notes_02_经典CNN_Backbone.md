本文持续总结典型的有监督训练范式的CNN Backbone。

## ShuffleNet（2018）

ShuffleNet的核心创新点在于通过**Channel Shuffle**和**分组卷积优化**，如下：

### **1. Channel Shuffle**
- **问题背景**：传统组卷积（Group Convolution）导致不同组的通道间信息隔离，降低特征融合能力。
- **解决方案**：  
  - 在组卷积后对输出通道进行**均匀重排**，打破组间壁垒。  
  - 操作方式：将输出通道按组数分割，按序交叉重组（如输入通道分为G组，每组通道按序交叉排列）。  
- **数学实现**：  
  $$
  \text{Shuffle}(x)_{i,j,k} = x_{i, (j \mod G) \times C/G + \lfloor j/G \rfloor, k}
  $$
  - \( G \)：组数，\( C \)：通道总数，\( i, j, k \)：批次、通道、空间索引。
- **效果**：  
  - 无需额外参数，实现跨组信息交互。  
  - 提升模型精度（ImageNet Top-1错误率降低约1.5%）。

### **2. 逐点组卷积（Pointwise Group Convolution）**
- **结构优化**：将标准1×1卷积替换为**分组1×1卷积**，减少计算量。  
- **计算量对比**：  
  $$
  \text{标准卷积：} C_{\text{in}} \times C_{\text{out}} \times 1 \times 1
  $$
  $$
  \text{分组卷积：} \frac{C_{\text{in}} \times C_{\text{out}} \times 1 \times 1}{G}
  $$
  - 以 $ C_{\text{in}} = C_{\text{out}} = 256 $，$ G=4 $ 为例，计算量减少至25%。

### **3. ShuffleNet Unit**
- **基础单元结构**：  
  ```plaintext
  输入 → 分组1×1卷积 → Channel Shuffle → 3×3深度卷积 → 分组1×1卷积 → 跳跃连接
  ```
- **残差连接**：当输入输出维度不匹配时，通过带步长的深度卷积或平均池化对齐尺寸。

![ShuffleNet Unit](https://user-images.githubusercontent.com/26739999/142575730-dc2f616d-80df-4fb1-93e1-77ebb2b835cf.png)


### **4. 网络整体架构**
- **分阶段降采样**：  
  - 包含3个主要阶段（Stage 2-4），每阶段通过步长=2的ShuffleNet单元下采样。  
  - 通道数倍增规则：从初始24通道逐步扩展至192/384/768通道（ShuffleNet 1×/0.5×配置）。
- **全局平均池化**：末端使用GAP替代全连接层，减少参数量。


### **5. 性能与效率优势**
#### **(1) 计算量对比（ImageNet分类）**
| 模型            | Top-1错误率 | FLOPs (M) | 参数量 (M) |  
|-----------------|-------------|-----------|------------|  
| MobileNet V1    | 29.4%       | 569       | 4.2        |  
| **ShuffleNet 1× (G=3)** | **32.4%** | **140**   | **1.9**    |  
| ShuffleNet 2× (G=3) | 26.3%    | 524       | 5.3        |  

#### **(2) 移动端推理速度**  
- 在ARM Cortex-A9上，ShuffleNet 1×的推理速度较AlexNet快18倍，精度相当。

## ShuffleNetV2（2018）

ShuffleNetV2在ShuffleNet基础上，通过**硬件感知设计**和**效率优化策略**，进一步提升了模型的实际推理速度与能效比。以下是其核心创新点的系统概括：

### **1. 硬件感知设计原则**
ShuffleNetV2提出四条硬件感知优化准则，直接针对移动端设备的计算特性进行优化：

#### **(1) 平衡输入输出通道数**
- **问题**：卷积层的输入输出通道数差异过大会增加内存访问成本（MAC）。  
- **优化**：设计模块时尽量使输入输出通道数相等（如1:1比例），减少内存带宽压力。

#### **(2) 减少分组数**
- **问题**：极端分组（如G=8）会降低GPU/CPU的并行计算效率。  
- **优化**：限制分组数（如G=1或G=2），在计算量与硬件利用率间取得平衡。

#### **(3) 减少网络碎片化**
- **问题**：多分支结构（如Inception）增加调度开销，降低硬件利用率。  
- **优化**：采用单路径结构，减少分支数量，提升并行度。

#### **(4) 减少逐元素操作**
- **问题**：ReLU、Add等逐元素操作虽计算量小，但内存访问频繁，成为瓶颈。  
- **优化**：减少不必要的逐元素操作（如合并ReLU与Add）。

### **2. ShuffleNetV2单元设计**
#### **(1) 基础单元结构**
- **单路径设计**：  
  ```plaintext
  输入 → 1×1卷积 → Channel Shuffle → 3×3深度卷积 → 1×1卷积 → 跳跃连接
  ```
- **通道分割**：  
  将输入通道分为两部分，一部分直接跳跃连接，另一部分通过卷积处理，最后拼接输出。

#### **(2) 下采样单元**
- **结构改进**：  
  - 使用步长=2的3×3深度卷积进行下采样。  
  - 跳跃连接分支使用平均池化对齐尺寸。

![ShuffleNetV2 Unit](https://user-images.githubusercontent.com/26739999/142576336-e0db2866-3add-44e6-a792-14d4f11bd983.png)

### **3. 性能与效率优势**
#### **(1) 计算量对比（ImageNet分类）**
| 模型            | Top-1错误率 | FLOPs (M) | 参数量 (M) |  
|-----------------|-------------|-----------|------------|  
| ShuffleNet 1×   | 32.4%       | 140       | 1.9        |  
| **ShuffleNetV2 1×** | **29.1%** | **146**   | **2.3**    |  
| MobileNet V2    | 28.0%       | 300       | 3.4        |  

#### **(2) 移动端推理速度**  
- 在ARM Cortex-A53上，ShuffleNetV2 1×的推理速度较ShuffleNet 1×快约20%，精度提升3.3%。

## RepVGG (2021)

RepVGG的核心创新点在于提出一种**训练-推理解耦的架构设计**，通过**Structural Re-parameterization**技术，在训练时使用多分支结构提升性能，在推理时转换为单路径VGG式结构以提升效率。以下是其创新点的系统概括：

### **1. 训练-推理解耦设计**
#### **(1) 训练阶段：多分支结构**
- **结构组成**：  
  - **主分支**：3×3卷积（核心特征提取）。  
  - **旁路分支**：1×1卷积（增强局部特征）。  
  - **跳跃连接**：恒等映射（保留原始信息）。  
  ```plaintext
  输入 → 3×3卷积 → 输出
       ↘ 1×1卷积 → 输出
       ↘ 恒等映射 → 输出
       → 求和 → 激活
  ```
- **优势**：多分支结构提供丰富的梯度流，提升训练效果。

![RepVGG](https://user-images.githubusercontent.com/26739999/142573223-f7f14d32-ea08-43a1-81ad-5a6a83ee0122.png)

#### **(2) 推理阶段：单路径结构**
- **结构转换**：通过结构重参数化，将多分支融合为单路径3×3卷积。  
- **优势**：  
  - 减少内存访问成本（MAC）。  
  - 提升硬件并行度，加速推理。

### **2. 结构重参数化技术**
#### **(1) 卷积与BN融合**
- **公式**：  
  将卷积层与BN层合并为单一卷积：  
  $$
  W' = \frac{W}{\sqrt{\sigma^2 + \epsilon}}, \quad b' = \frac{b - \mu}{\sqrt{\sigma^2 + \epsilon}}
  $$
  - $ W, b $：卷积核权重与偏置。  
  - $ \mu, \sigma^2 $：BN层的均值与方差。

#### **(2) 多分支融合**
- **1×1卷积转换**：  
  将1×1卷积扩展为3×3卷积（外围填充0）。  
- **恒等映射转换**：  
  将恒等映射视为1×1单位卷积，再扩展为3×3卷积（中心为1，其余为0）。  
- **分支合并**：  
  将转换后的3×3卷积权重与偏置相加，得到等效单路径卷积。

### **3. 性能与效率优势**
#### **(1) ImageNet实验结果对比**
| 模型              | 参数量 (M) | Top-1 Acc | FLOPs (G) | 推理速度（GPU, ms） |  
|-------------------|------------|-----------|-----------|---------------------|  
| ResNet-50         | 25.6       | 76.1%     | 4.1       | 7.2                 |  
| **RepVGG-A1**     | **24.1**   | **78.5%** | **1.8**   | **3.1**             |  
| EfficientNet-B0   | 5.3        | 77.1%     | 0.39      | 5.4                 |  

#### **(2) 推理速度优势**  
- 在NVIDIA V100上，RepVGG-A1的推理速度较ResNet-50快约2.3倍，精度提升2.4%。

## ResNeXt (2017)

ResNeXt的核心创新点在于引入**基数（Cardinality）**，通过**分组卷积（Grouped Convolutions）**和**多分支残差结构**，在保持计算效率的同时提升模型的表达能力。其核心思想如下：

### **1. 基数（Cardinality）作为新维度**
- **定义**：基数指残差块中**并行变换路径的数量**，即分支数。  
- **与传统维度对比**：  
  | 维度     | 作用                           | ResNet优化方向 | ResNeXt优化方向 |  
  |----------|--------------------------------|---------------|-----------------|  
  | 深度     | 网络层数                       | 增加          | 固定            |  
  | 宽度     | 每层通道数                     | 固定          | 固定            |  
  | **基数** | **并行路径数（分组数）**       | 无            | **增加**        |  
- **优势**：通过增加基数（而非深度或宽度），在相同计算量下提升模型表达能力。

### **2. 分组卷积的多分支残差块**
- **结构设计**：  
  每个残差块包含 **C个同构分支**（C为基数），每个分支执行相同的拓扑结构（如1×1→3×3→1×1卷积），最后合并输出。  
  ```plaintext
  输入 → 分组卷积（C组） → 合并 → 跳跃连接 → 输出
  ```
- **数学表达**：  
  $$
  y = x + \sum_{i=1}^C \mathcal{T}_i(x)
  $$
  - $ \mathcal{T}_i $：第i个分支的变换函数（同构结构）。

![ResNeXt Block](https://user-images.githubusercontent.com/26739999/142574479-21fb00a2-e63e-4bc6-a9f2-989cd6e15528.png)

### **3. 等效实现的优化形式**
- **分组卷积等效性**：  
  多分支结构可等效转换为**分组卷积+通道拼接**，减少实现复杂度。  
  - **原始多分支**：每个分支独立计算后拼接。  
  - **等效实现**：单层卷积中设置`groups=C`，直接分组处理。  
- **参数效率**：  
  以基数=32为例，参数量仅为传统多分支模型的1/32，同时保留多分支的表示能力。

### **4. 性能与效率优势**
#### **(1) ImageNet实验结果对比**
| 模型              | 参数量 (M) | Top-1错误率 | 计算量 (FLOPs) |  
|-------------------|------------|-------------|----------------|  
| ResNet-50         | 25.6       | 23.9%       | 3.8B           |  
| **ResNeXt-50 (32×4d)** | **25.0** | **22.2%**   | **4.2B**       |  
| ResNet-101        | 44.5       | 22.0%       | 7.6B           |  
| **ResNeXt-101 (32×4d)** | **44.2** | **21.2%**   | **8.0B**       |  

#### **(2) 基数与深度/宽度的效率对比**  
在相同计算量约束下：  
- **提升基数（C=32）**：错误率从25.1%降至22.2%。  
- **提升深度（层数+50%）**：错误率仅降至23.8%。  
- **提升宽度（通道+50%）**：错误率降至23.4%。  

## Res2Net (2021)

Res2Net的核心创新点在于提出了**层级残差连接（Hierarchical Residual Connections）**和**多尺度特征增强机制**，通过在每个残差块内部构建分层的多分支结构，显著提升了模型对多尺度特征的捕捉能力。以下是其创新点的系统概括：


### **1. 层级残差结构（Hierarchical Residual Structure）**
- **特征分组与逐级处理**：  
  将输入特征图分为 \( s \) 个**子特征组**（通常 \( s=4 \)），每个子组依次通过3×3卷积处理，并**逐级融合相邻子组的信息**。  
  ```plaintext
  输入 → 分组1 → 卷积 → 输出1  
            ↘ 分组2 + 输出1 → 卷积 → 输出2  
                     ↘ 分组3 + 输出2 → 卷积 → 输出3  
                              ↘ 分组4 + 输出3 → 卷积 → 输出4  
  ```
- **数学表达**：  
  $$
  y_i = 
  \begin{cases} 
  x_i & i=1 \\
  \text{Conv}(x_i + y_{i-1}) & 2 \leq i \leq s 
  \end{cases}
  $$
  - \( x_i \)：第 \( i \) 个子组输入特征  
  - \( y_i \)：第 \( i \) 个子组输出特征  

![Res2Net](https://user-images.githubusercontent.com/26739999/142573547-cde68abf-287b-46db-a848-5cffe3068faf.png)

### **2. 多尺度特征融合**
- **跨组感受野扩展**：  
  每个子组的有效感受野随层级递增，例如：  
  - 分组1：3×3 → 分组2：5×5（叠加两次3×3） → 分组4：等效9×9。  
- **自适应特征组合**：  
  最终将所有子组输出拼接，通过1×1卷积调整通道数，保留多尺度信息。

### **3. 计算效率优化**
- **分组卷积参数控制**：  
  每组通道数减少至 $ C/s $（$ C $ 为总通道数），确保FLOPs与标准残差块相近。  
- **对比ResNeXt**：  
  | 模型          | 参数量（M） | FLOPs（G） | ImageNet Top-1 Acc |  
  |---------------|------------|------------|--------------------|  
  | ResNet-50     | 25.6       | 4.1        | 76.1%              |  
  | ResNeXt-50    | 25.0       | 4.2        | 77.8%              |  
  | **Res2Net-50**| **25.7**   | **4.3**    | **79.2%**          |  

## HRNet (2025)

HRNet（High-Resolution Network）的核心创新点在于提出了一种**并行多分辨率特征融合架构**，通过在整个网络中**保持高分辨率特征**并**动态融合多尺度信息**，显著提升了密集预测任务（如姿态估计、语义分割）的性能。以下是其创新点的系统概括：

### **1. 并行多分辨率特征流**
- **结构设计**：  
  - **多分支并行**：网络包含多个分辨率分支（如1/4、1/8、1/16、1/32），每个分支独立处理不同尺度的特征。  
  - **高分辨率保持**：始终保留高分辨率分支（如1/4），避免传统方法中通过下采样丢失细节信息。  
- **对比传统方法**：  
  | 方法          | 特征分辨率变化          | 高分辨率信息保留 |  
  |---------------|------------------------|------------------|  
  | FCN/U-Net     | 逐步下采样→上采样      | 部分恢复         |  
  | **HRNet**     | **并行多分辨率处理**   | **全程保留**     | 

![HRNet](https://user-images.githubusercontent.com/26739999/149920446-cbe05670-989d-4fe6-accc-df20ae2984eb.png)

### **2. 动态特征融合机制**
- **跨分辨率交互**：  
  通过**重复的多分辨率融合模块**，在不同分辨率分支间交换信息：  
  - **高→低分辨率**：上采样+特征拼接。  
  - **低→高分辨率**：下采样+特征拼接。  
- **融合公式**：  
  $$
  F_i = \text{Conv}(\text{Concat}(F_i, \text{Up}(F_{i+1}), \text{Down}(F_{i-1}))
  $$
  - $ F_i $：第 $ i $ 个分辨率分支的特征。  
  - $ \text{Up}, \text{Down} $：上采样与下采样操作。  

### **3. 网络整体架构**
- **阶段划分**：  
  - **阶段1**：单高分辨率分支（1/4）。  
  - **阶段2-4**：逐步增加低分辨率分支（1/8、1/16、1/32）。  
- **模块堆叠**：  
  每个阶段包含多个**多分辨率融合模块**，确保信息充分交互。

### **4. 性能与效率优势**
#### **(1) 密集预测任务表现**
| 任务            | 数据集      | 指标         | HRNet性能 | 对比模型性能 |  
|-----------------|-------------|--------------|-----------|--------------|  
| 人体姿态估计    | COCO        | AP           | 77.0      | 73.7 (ResNet) |  
| 语义分割        | Cityscapes  | mIoU         | 81.5      | 78.5 (DeepLab) |  
| 目标检测        | COCO        | AP           | 43.1      | 41.5 (ResNet) |  

#### **(2) 计算效率**  
- 在相同FLOPs下，HRNet较ResNet-50提升约3-5%的AP/mIoU。  
- 高分辨率分支的引入仅增加约10%的计算量。
