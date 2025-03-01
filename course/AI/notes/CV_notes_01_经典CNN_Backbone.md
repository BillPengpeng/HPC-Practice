本文持续总结典型的有监督训练范式的CNN Backbone。

## VGG（2014）

VGG的核心创新点可概括为以下四个方面：

**1. 小卷积核的深度堆叠**  
• 全网络统一使用**3×3小卷积核**（仅最后一层含1×1卷积），替代AlexNet中的大卷积核（11×11, 5×5）  
• 通过**多层小卷积串联**等效替代大感受野（2层3×3卷积=5×5感受野，3层等效7×7），在保持相同感受野的同时：  
  ✓ 减少参数量（3×3×3=27 vs 7×7=49参数）  
  ✓ 增加非线性激活次数（每层后接ReLU）  

**2. 深度架构的突破性验证**  
• 构建了当时最深的**16-19层网络**（VGG16/VGG19），证明了**深度增加对特征抽象能力的显著提升**  
• 通过实验证实：当网络深度从11层增至19层时，ImageNet top-5错误率从10.1%降至7.5%

**3. 模块化结构设计**  
• 采用**分阶段特征提取策略**，每阶段包含：  
  ✓ 2-4个卷积层堆叠  
  ✓ 1个最大池化层（2×2，stride=2）  
• 特征图尺寸逐级减半（224→112→56→28→14→7），通道数倍增（64→128→256→512→512）

**4. 全连接层标准化**  
• 末端使用**3个全连接层**（4096→4096→1000），首次将全连接层标准化为固定结构  
• 尽管后续研究证明全连接层存在冗余（如ResNet改用全局平均池化），但该设计成为早期CNN的标准范式

**关键影响**：  
✓ 启发了后续ResNet、Inception等深层网络的设计思路  
✓ 3×3卷积堆叠成为现代CNN的基础构建模块  
✓ 预训练的VGG特征至今仍用于迁移学习（如风格迁移、目标检测）  

![VGG](https://user-images.githubusercontent.com/26739999/142578905-9be586ec-f6fd-4bfb-bbba-432f599d3b9b.png)

## ResNet（2016）

ResNet的核心创新点可系统概括为以下五个方面：

### **1. 残差学习框架（Residual Learning）**
- **核心思想**：将网络层的学习目标从完整映射$H(x)$改为残差映射$F(x) = H(x) - x$，通过**Skip Connection**实现恒等映射的捷径。
- **数学表达**：  
  $$
  H(x) = F(x) + x
  $$
  若最优解接近恒等映射，则残差 $F(x)$ 趋近于零，比直接学习 $H(x) = x$ 更易优化。
- **解决退化问题**：允许梯度直接通过跳跃连接回传，缓解了深度网络中的梯度消失/爆炸问题。

### **2. 残差块结构设计**
- **基础残差块**：由两个3×3卷积层构成（适用浅层网络如ResNet-34）。
- **瓶颈残差块（Bottleneck）**：  
  $$
  1\times1 \text{卷积（降维）} \rightarrow 3\times3 \text{卷积} \rightarrow 1\times1 \text{卷积（升维）}
  $$
  减少计算量（如ResNet-50/101/152），使千层网络训练成为可能。
- **自适应Shortcut**：当输入输出维度不一致时，使用**1×1卷积**调整通道数或步长（Stride=2下采样）。

![残差模块](https://user-images.githubusercontent.com/26739999/142574068-60cfdeea-c4ec-4c49-abb2-5dc2facafc3b.png)

### **3. 极深网络架构的实现**
- **网络深度突破**：首次成功训练超过100层的网络（如ResNet-152），相比VGG19（19层）深度提升近8倍。
- **分阶段特征提取**：  
  划分为多个阶段（如ResNet-50的4个阶段），每阶段通过步长=2的卷积或残差块进行下采样。
  通道数逐阶段倍增（64→256→512→1024→2048），特征图尺寸逐级减半。

### **4. 训练优化技术**
- **全网络批归一化（BatchNorm）**：  
  每个卷积层后接BN层，加速收敛并提升训练稳定性。
- **预激活结构（ResNet v2改进）**：  
  将BN和ReLU置于卷积操作前（`BN→ReLU→Conv`），进一步改善梯度流动。
- **全局平均池化替代全连接层**：  
  末端使用全局平均池化（GAP）生成特征向量，大幅减少参数（如ResNet-50参数量为25.5M，远低于VGG16的138M）。

### **5. 性能与影响**
- **ImageNet 2015夺冠**：Top-5错误率低至3.57%，较前一年冠军（VGG）提升近40%。
- **跨任务泛化性**：成为计算机视觉任务（检测、分割等）的骨干网络标配。
- **后续衍生模型**：  
  - ResNeXt：引入分组卷积提升特征多样性。
  - DenseNet：密集跳跃连接加强特征复用。
  - Transformer适配：如ViT中的残差结构。


### **6. 关键创新对比（ResNet vs VGG）**
| 特性                | ResNet                     | VGG                        |
| :--:    | :--:    | :--:       |
| 核心结构            | 残差块 + 跳跃连接           | 3×3卷积堆叠                |
| 最大深度            | 152层                     | 19层                       |
| 参数效率            | 高（Bottleneck设计）       | 低（全连接层占90%参数）     |
| 梯度传播            | 多路径缓解梯度消失          | 单一路径易梯度消失          |
| 典型应用            | 检测、分割、分类            | 分类、风格迁移              |

## MobileNet（2017）

MobileNet V1的核心创新在于**深度可分离卷积**（Depthwise Separable Convolution）。

### **1. 核心创新：深度可分离卷积**
#### **(1) 结构分解**
将标准卷积分解为两步独立操作：**深度卷积（Depthwise Convolution）**、**逐点卷积（Pointwise Convolution）**。

#### **(2) 计算量对比**
- **标准卷积计算量**：$D_K \times D_K \times M \times N \times D_F \times D_F$
- **深度可分离卷积计算量**：  
  $$
  \text{深度卷积} + \text{逐点卷积} = (D_K^2 \times M \times D_F^2) + (M \times N \times D_F^2)
  $$
- **节省比例**：  
  $$
  \frac{\text{深度可分离计算量}}{\text{标准计算量}} = \frac{1}{N} + \frac{1}{D_K^2}
  $$
  - 以3×3卷积为例（$ D_K=3 $, $ N=64 $），计算量减少约 **8~9倍**。

### **2. 辅助优化策略**
#### **(1) 宽度乘数（Width Multiplier, α）**  
- **作用**：按比例 $ \alpha \in (0, 1] $ **统一减少所有层的通道数**，实现模型轻量化。  
- **效果**：计算量和参数量近似减少 $ \alpha^2 $ 倍，例如，$ \alpha=0.75 $ 时，模型大小和计算量约为原版的56%。

#### **(2) 分辨率乘数（Resolution Multiplier, ρ）**  
- **作用**：降低输入图像分辨率（如从224×224→192×192→128×128），进一步减少计算量。  
- **效果**：计算量减少 $ \rho^2 $ 倍（因特征图尺寸平方级缩小）。

### **3. 性能与优势**
#### **(1) 轻量化效果**  
| 模型             | 参数量 (M) | FLOPs (Billion) | ImageNet Top-1 Acc |
|------------------|------------|-----------------|--------------------|
| MobileNet V1 (1.0) | 4.2        | 0.569           | 70.6%             |
| VGG16            | 138        | 15.5            | 71.5%             |
| **节省比例**     | **97%↓**   | **96%↓**        | **精度接近**      |

#### **(2) 应用场景适配**  
- **移动端推理**：单核CPU可实现实时处理（如100ms内完成ImageNet分类）。  
- **嵌入式设备**：低内存占用（模型文件仅约16MB）。  
- **下游任务兼容性**：作为骨干网络支持目标检测（如MobileNet-SSD）、语义分割等。

### **4. 设计哲学与影响**
- **核心思想**：通过**分解卷积的通道与空间维度**，消除标准卷积中的计算冗余。  
- **后续影响**：  
  - 启发了MobileNet V2/V3、EfficientNet等轻量化模型的设计。  
  - 成为移动端AI应用（如手机相机场景识别、AR滤镜）的标配骨干网络。  
- **局限性与改进方向**：  
  - 深度卷积的参数量占比低（仅3%），但计算量占比高（95%），后续版本通过倒置残差结构进一步优化。  
  - 缺乏动态特征调整能力（V3引入SE模块改进）。


## MobileNet V2（2018）

MobileNet V2在V1基础上通过结构重构与优化，显著提升了模型性能与效率，其核心创新点可系统概括如下：

### **1. 倒置残差结构倒置残差结构（Inverted Residuals）**
- **传统残差 vs 倒置残差**：
  | 结构特性        | 传统残差（ResNet）       | 倒置残差（MobileNet V2） |
  |----------------|------------------------|-------------------------|
  | **维度变化**    | 降维→处理→升维          | **升维→处理→降维**       |
  | **通道数变化**  | 输入→减少→恢复          | 输入→**扩展6倍**→压缩    |
  | **跳跃连接**    | 高低维间连接（需匹配）   | 仅在输入输出维度相同时连接 |

- **结构细节**：
  - **扩展层**：1×1卷积将输入通道扩展6倍（如输入24→扩展至144），增加特征表达能力。
  - **深度卷积**：3×3深度可分离卷积处理扩展后的高维特征。
  - **压缩层**：1×1卷积将通道压缩回原尺寸（如144→24），减少计算量。

![倒置残差结构](https://user-images.githubusercontent.com/26739999/142563365-7a9ea577-8f79-4c21-a750-ebcaad9bcc2f.png)

### **2. 线性瓶颈（Linear Bottleneck）**
- **问题背景**：低维空间中使用ReLU易造成信息丢失（ReLU将负值置零，破坏低维紧凑特征）。
- **解决方案**：
  - **压缩层去除非线性**：在降维的1×1卷积后**移除ReLU激活**，保留线性变换。
  - **仅在扩展层使用ReLU**：高维空间（扩展6倍后）使用ReLU可安全激活，减少信息损失。

### **3. 模块化设计优化**
- **残差块流程**：
  ```plaintext
  输入 → 1x1 Conv (升维, ReLU6) → 3x3 DWConv (ReLU6) → 1x1 Conv (降维, 线性) → 跳跃连接（若维度匹配）
  ```
- **ReLU6激活**：限制输出最大值为6（`min(max(x, 0), 6)`），增强低精度计算的鲁棒性。

### **4. 性能与效率提升**
#### **(1) 计算量对比（同精度下）**
| 模型              | Top-1 Acc (ImageNet) | FLOPs (B) | 参数量 (M) |
|-------------------|----------------------|-----------|------------|
| MobileNet V1 (1.0)| 70.6%                | 0.569     | 4.2        |
| **MobileNet V2 (1.0)** | **72.0%**      | **0.3**   | **3.4**    |

#### **(2) 结构效率分析**
- **扩展比（Expansion Ratio）**：升维6倍平衡了特征多样性与计算成本。
- **跳跃连接频率**：仅部分层使用残差连接（如V2-1.0含7个残差块），避免冗余计算。

### **5. 下游任务适配性**
- **目标检测**：作为骨干网络，MobileNet V2-SSD在COCO数据集上较V1提速20%，mAP提升3%。
- **语义分割**：结合轻量级解码器（如DeepLabv3+），实现移动端实时分割（30 FPS）。

### **6. 影响与总结**
- **设计哲学**：通过**高维扩展→深度处理→低维压缩**的倒置流程，最大化特征表达能力与计算效率的平衡。
- **工业应用**：成为移动端CV任务（如手机拍照场景识别、实时视频处理）的主流骨干网络。
- **后续发展**：启发了MobileNet V3（引入NAS搜索）与EfficientNet（复合缩放）的进一步优化。

## MobileNet V3（2019）

MobileNet V3结合神经架构搜索（NAS）与人工设计优化，进一步提升精度与效率。

### **1. 神经架构搜索（NAS）驱动的网络设计**
- **多目标优化框架**：  
  - 使用**MnasNet**框架，以**延迟（Latency）**和**准确率（Accuracy）**为双目标进行搜索，平衡移动端部署需求。
  - 搜索空间涵盖卷积类型（常规/深度可分离）、核大小、激活函数等。
- **搜索结果应用**：  
  - 发现更高效的层配置（如减少早期层通道数，增加后期层深度）。
  - 优化残差块的位置与数量（如V3-Large的最后一阶段采用密集残差连接）。

### **2. 硬件感知的激活函数优化**
- **h-swish替代ReLU6**：  
  $$
  \text{h-swish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}
  $$
  - **优势**：近似swish函数（平滑梯度），但避免指数计算，适配低精度硬件。
  - **部署策略**：仅在深层使用h-swish（如网络后半段），浅层保留ReLU以降低延迟。

### **3. 轻量化SE模块（Squeeze-and-Excite）**
- **精简通道注意力**：  
  - 传统SE模块计算量：$ \frac{2}{r} \times C^2 $（r为压缩比）。
  - V3：固定压缩比（如r=4），并仅在部分残差块中应用（如V3-Large的最后一阶段）。
- **计算量对比**：  
  SE模块参数量减少25%（如从56K→42K），对FLOPs影响小于0.5%。

![MobileNet V3 Block](https://user-images.githubusercontent.com/26739999/142563801-ef4feacc-ecd7-4d14-a411-8c9d63571749.png)

### **4. 复合缩放策略（Compound Scaling）**
- **联合调整三维度**：  
  $$
  \text{深度}: d = \alpha^\phi, \quad \text{宽度}: w = \beta^\phi, \quad \text{分辨率}: r = \gamma^\phi
  $$
  - 通过网格搜索确定最优比例 $\alpha=1.2, \beta=1.4, \gamma=1.15$。
  - 实现模型系列化扩展（如V3-Small/Large/EdgeTPU）。

### **5. 网络结构细节优化**
- **高效初始层**：使用3×3卷积（stride=2）替代V2的初始3层堆叠，减少早期计算量（FLOPs降低10%）。  
- **动态扩展比**：不同阶段的扩展比调整（如V3-Large阶段1扩展比=1，阶段5扩展比=6）。 
- **分类头轻量化** ：移除V2末端的部分1×1卷积，直接连接池化层与分类器，减少参数15%。

### **6. 版本分支与性能对比**
| 模型               | Top-1 Acc | FLOPs (B) | 延迟（Pixel 4 CPU） | 核心改进点                     |
|--------------------|-----------|-----------|---------------------|--------------------------------|
| **V3-Large**       | 75.2%     | 0.66      | 51ms               | NAS优化 + h-swish + SE         |
| **V3-Small**       | 67.4%     | 0.16      | 22ms               | 极简架构 + 复合缩放            |
| V2 (1.0)           | 72.0%     | 0.3       | 37ms               | 对比基准                       |

### **7. 硬件友好性改进**
- **减少分支结构**：避免碎片化算子（如Inception中的多路径），适配移动端加速器（DSP/NPU）。
- **量化友好设计**：限制激活值范围（如h-swish的[0,6]），提升8-bit量化后的精度保持率（<1%下降）。

## InceptionV1（GoogLeNet）（2014）

InceptionV1（即GoogLeNet）的核心创新点可概括为以下五个方面：

### **1. Inception模块（多尺度并行卷积）**
- **并行结构设计**：在同一层级集成四种操作：
  - 1×1卷积（特征压缩+非线性增强）
  - 3×3卷积（局部特征提取）
  - 5×5卷积（大范围特征捕获）
  - 3×3最大池化（空间信息保留）
- **特征融合**：将不同尺度的输出在通道维度拼接，形成多尺度特征融合。  
- **计算优化**：通过1×1卷积先降维（如将输入通道从192压缩至64），使后续3×3和5×5卷积的计算量减少3-5倍。

![Inception Block](https://i-blog.csdnimg.cn/blog_migrate/19df1a2b05932df2ccf40dc151407480.png)

### **2. 1×1卷积的双重作用**
- **降维（Bottleneck）**：减少输入通道数，抑制计算复杂度（如将256通道压缩至64通道）。
- **非线性增强**：每个1×1卷积后接ReLU激活，提升模型表达能力（相比直接使用大卷积核，参数量减少90%）。

### **3. 辅助分类器（Auxiliary Classifiers）**
- **位置**：在网络中部（Inception4a和Inception4d输出端）插入两个辅助分类器。
- **功能**：
  - 缓解梯度消失：通过中层损失反向传播，增强浅层梯度信号。
  - 正则化效果：相当于隐式多任务学习，防止过拟合。
- **推理阶段**：仅保留主分类器，辅助分支被移除，不增加推理开销。

### **4. 全局平均池化替代全连接层**
- **结构变革**：网络末端使用全局平均池化（GAP）生成特征向量，替代传统全连接层。
- **优势**：
  - 参数减少：AlexNet全连接层占比90%参数，InceptionV1仅保留5%参数用于分类。
  - 防止过拟合：降低模型复杂度，提升泛化能力。

### **5. 深度与效率的平衡**
- **22层深度结构**：比VGG16（13层卷积+3层全连接）更深，但计算量仅1.5B FLOPs（VGG16为15B FLOPs）。
- **模块堆叠策略**：9个Inception模块的链式组合，逐步抽象特征。

### **性能对比（2014 ImageNet）**
| 模型       | Top-5错误率 | 参数量 | FLOPs  | 核心创新点               |
|------------|-------------|--------|--------|--------------------------|
| AlexNet    | 16.4%       | 60M    | 0.7B   | ReLU、Dropout            |
| **GoogLeNet** | **6.67%**   | **6.8M** | **1.5B** | Inception模块、辅助分类器 |
| VGG16      | 7.3%        | 138M   | 15B    | 3×3卷积堆叠              |

### **影响与总结**
InceptionV1通过**多尺度特征融合**与**高效计算设计**，奠定了现代深度网络模块化设计的基础。其核心思想被后续模型广泛借鉴：
- **Inception系列**：V2/V3引入BN、分解卷积；V4整合残差连接。
- **ResNet**：借鉴Bottleneck结构与深度堆叠理念。
- **MobileNet**：采用1×1卷积进行通道操作优化。  
该模型证明：通过精心设计的模块化结构，可以在不显著增加计算成本的前提下实现更深、更强的特征提取能力。

## InceptionV2（2015）

InceptionV2的核心创新点在于通过**批归一化（Batch Normalization）**与**卷积分解策略**显著提升了训练速度和模型性能。

### **1. 批归一化（Batch Normalization）的引入**
### **2. 卷积核分解优化**
#### **(1) 大卷积核分解为小卷积堆叠** 
- **5×5 → 双3×3**：  
  将单个5×5卷积替换为两个3×3卷积，保持相同感受野（5×5），同时：  
  - **参数减少**：\( 5^2 = 25 \) → \( 3^2 \times 2 = 18 \)（减少28%）  
  - **非线性增强**：增加一层ReLU激活。  

#### **(2) 非对称卷积分解**  
- **n×n → 1×n + n×1**：  
  将3×3卷积分解为1×3和3×1卷积的串联，例如：  
  - **参数减少**：\( 3^2 = 9 \) → \( 3 + 3 = 6 \)（减少33%）  
  - **保持感受野**：适用于中等特征图尺寸（如12×12以上）。
![Inception E Block](https://user-images.githubusercontent.com/26739999/177241797-c103eff4-79bb-414d-aef6-eac323b65a50.png)

### **3. Inception模块结构优化**
#### **(1) 模块重构**  
- **分支简化**：移除InceptionV1中的5×5卷积分支，优先使用3×3分解结构。  
- **池化分支调整**：将最大池化层前置，后接1×1卷积降维（减少计算量）。  

#### **(2) 典型模块结构**  
```plaintext
输入 → 1x1卷积 → 分支1（输出）
       ↘ 1x1 → 3x3 → 分支2（输出）
       ↘ 1x1 → 3x3 → 3x3 → 分支3（输出）
       ↘ 3x3 MaxPool → 1x1 → 分支4（输出）
       → 通道拼接
```
![InceptionV2](https://i-blog.csdnimg.cn/blog_migrate/5f79db9dcf67486f7596d571e901a4f7.png)

### **4. 标签平滑（Label Smoothing）**
- **目的**：缓解过拟合，避免模型对训练样本过度自信。  
- **公式**：调整真实标签的概率分布：  
  $$
  q'(k) = (1 - \epsilon) \cdot q(k) + \frac{\epsilon}{K}
  $$
  - $ \epsilon $: 平滑系数（通常0.1）  
  - $ K $: 类别总数  
- **效果**：提升模型泛化性，降低验证集错误率约0.2%。

### **5. 训练策略优化**
- **学习率调整**：采用更激进的初始学习率（如0.045），配合指数衰减。  
- **数据增强**：引入更高效的光度畸变（Photometric Distortions）替代传统PCA噪声。  

### **6. 影响与总结**
InceptionV2通过**批归一化**与**卷积分解**两大核心技术，其设计哲学为后续模型奠定了基础：  
- **BN层**成为现代深度网络的标配组件。  
- **分解卷积思想**启发了EfficientNet的复合缩放策略。  
- **轻量化设计**推动移动端模型发展（如MobileNet的深度可分离卷积）。  

InceptionV3在V2基础上额外添加策略，包括：RMSProp、Label Smooth、Factorized 7x7 卷积（将7×7卷积层分解为两个3×3卷积层）、辅助任务引入BN层。

![InceptionV3](https://i-blog.csdnimg.cn/blog_migrate/32b56372e65e9c3e83a7568259b292dd.png)

``` 
mmpretrain实现版本

@MODELS.register_module()
class InceptionV3(BaseBackbone):
    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = False,
        dropout: float = 0.5,
        init_cfg: Optional[dict] = [
            dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.1),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ],
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.aux_logits = aux_logits
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)  # [branch1x1, branch5x5, branch3x3dbl, branch_pool] 对应论文A
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)                    # [branch3x3, branch3x3dbl, branch_pool]
        self.Mixed_6b = InceptionC(768, channels_7x7=128)  # [branch1x1, branch7x7, branch7x7dbl, branch_pool] 对应论文B
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)                    # [branch3x3, branch7x7x3, branch_pool]
        self.Mixed_7b = InceptionE(1280)                   # [branch1x1, branch3x3, branch3x3dbl, branch_pool] 对应论文C
        self.Mixed_7c = InceptionE(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, num_classes)

```

参考博文《Inception v1~v4和Inception-ResNet v1~v2总结》[link](https://blog.csdn.net/qq_40635082/article/details/124999323)。