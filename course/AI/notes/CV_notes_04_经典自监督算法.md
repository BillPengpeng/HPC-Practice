本文主要整理经典的自监督对比学习算法。

## MoCo V1 (2020)

MocoV1（Momentum Contrast for Unsupervised Visual Representation Learning）核心创新点在于通过改进对比学习框架，解决了传统方法对大批量训练数据的依赖，同时提升了特征表示的稳定性和质量。

### 1. **动量更新的编码器（Momentum-based Key Encoder）**
  - **动机**：传统对比学习（如SimCLR）使用同一编码器处理正负样本，导致特征不一致问题。而基于Memory Bank的方法因存储历史特征缺乏更新，存在特征陈旧性。
  - **解决方案**：引入**两个编码器**：**查询编码器**（Query Encoder，参数即时更新）和**键编码器**（Key Encoder，参数通过动量更新）,键编码器的参数更新公式如下，$m \in [0, 1)$为动量系数（如0.999）。

  $$
  \theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q
  $$

### 2. **动态队列（Dynamic Queue）管理负样本**
  - **动机**：端到端对比学习需在大批量中构造负样本，计算成本高；而Memory Bank存储所有历史样本，占用内存且特征陈旧。
  - **解决方案**：维护一个**固定大小的队列**，存储近期批次中键编码器生成的负样本特征；队列按先进先出（FIFO）更新，使负样本规模远超单批数据量（如队列容量可设10,000+），无需依赖大批量；结合动量编码器，队列中的负样本特征具有**时间一致性**，避免特征冲突。

```
@torch.no_grad()
def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
    """Update queue."""
    # gather keys before updating queue
    keys = torch.cat(all_gather(keys), dim=0)

    batch_size = keys.shape[0]

    ptr = int(self.queue_ptr)
    assert self.queue_len % batch_size == 0  # for simplicity

    # replace the keys at ptr (dequeue and enqueue)
    self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
    ptr = (ptr + batch_size) % self.queue_len  # move pointer

    self.queue_ptr[0] = ptr
```

### 3. **对比学习作为字典查找任务**
  - **任务形式化**：将对比学习视为**字典查询问题**，其中查询（Query）是当前样本的增强视图，键（Key）是队列中的负样本及其正样本对。  
  - **损失函数**：采用InfoNCE损失，最大化查询与正键的相似度，同时最小化与负键的相似度，$\tau$ 为温度系数，控制分布尖锐程度。
    $$
    \mathcal{L} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_{k^-} \exp(q \cdot k^- / \tau)}
    $$  
```
    def loss(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """Forward function to compute contrastive loss.

        Args:
            pos (torch.Tensor): Nx1 positive similarity.
            neg (torch.Tensor): Nxk negative similarity.

        Returns:
            torch.Tensor: The contrastive loss.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).to(pos.device)

        loss = self.loss_module(logits, labels)
        return loss
```
  - **Shuffling BN**：由于每个batch内的样本之间计算mean和std导致信息泄露，产生退化解。MoCo通过多GPU训练，分开计算BN，并且shuffle不同GPU上产生的BN信息来解决这个问题。

```
    # compute key features
    with torch.no_grad():  # no gradient to keys
        # update the key encoder
        self.encoder_k.update_parameters(
            nn.Sequential(self.backbone, self.neck))

        # shuffle for making use of BN
        im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

        k = self.encoder_k(im_k)[0]  # keys: NxC
        k = nn.functional.normalize(k, dim=1)

        # undo shuffle
        k = batch_unshuffle_ddp(k, idx_unshuffle)
```

### 4. **对比传统方法**
| **方法**       | **负样本来源**       | **一致性** | **内存消耗** | 批量依赖 |
|----------------|----------------------|------------|--------------|----------|
| **端到端**     | 同批其他样本         | 高         | 高（需大批量）| 强       |
| **Memory Bank**| 历史所有样本         | 低（陈旧） | 极高         | 弱       |
| **MoCoV1**     | 动态队列+动量编码器  | 高         | 中等         | 无       |


## SimCLR (2020)

<div align=center>
<img  src="https://user-images.githubusercontent.com/36138628/149723851-cf5f309e-d891-454d-90c0-e5337e5a11ed.png" width="400" />
</div>

参考博文[解析SimCLR：一种视觉表征对比学习的新框架](https://zhuanlan.zhihu.com/p/661129551)。

SimCLR（Simple Framework for Contrastive Learning of Visual Representations）的创新点主要体现在**数据增强策略、架构设计、损失函数优化**等方面。**没有动量法、没有memory bank、网络编码器也只用了一个**。

### 1. **系统化的数据增强组合**
  - **动机**：数据增强是生成正样本对的关键，但传统方法对增强策略的选择缺乏系统性分析。
  - **创新点**：提出**两阶段增强流程**：对同一图像依次应用两种随机增强操作（如裁剪、颜色失真、高斯模糊等），生成正样本对。连续地应用三种简单的增强方法：随机裁剪然后将大小调整回原始大小，随机颜色失真，以及随机高斯模糊。

### 2. **引入非线性投影头（Projection Head）**
  - **动机**：编码器输出的特征空间可能不适合直接用于对比学习。  
  - **创新点**：在编码器（如ResNet）后添加一个**多层感知机（MLP）投影头**，将特征映射到低维对比空间，结构为“线性层 → ReLU → 线性层”，增强非线性表达能力。  

```
neck=dict(
    type='NonLinearNeck',  # SimCLR non-linear neck
    in_channels=2048,
    hid_channels=2048,
    out_channels=128,
    num_layers=2,
    with_avg_pool=True)
```

### 3. **归一化温度标度交叉熵损失（NT-Xent Loss）**
  - **动机**：传统对比损失（如三元组损失）对负样本的利用效率低，且超参数敏感。  
  - **创新点**：提出Normalized Temperature-Scaled Cross Entropy Loss），$z_i, z_j$ 是同一图像的两个增强视图的特征，$\tau$为温度系数，控制相似度分布的尖锐程度。  
    $$
    \mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}
    $$
        
![参考图片](https://pic4.zhimg.com/v2-faa5cb397450a2dc876ba261e4a2c6d9_r.jpg)

### 4. **端到端的大规模对比学习框架**
   - **动机**：传统方法（如MoCo）依赖外部存储库（Memory Bank）或队列管理负样本，复杂度高。  
   - **创新点**：采用**纯端到端训练**，直接利用同一批次内的其他样本作为负样本，无需额外存储结构。实验表明，增大批量规模（如4096）可显著增加负样本数量，提升对比学习效果，缺点是对计算资源要求较高；采用**Global BN**，对应SyncBatchNorm，通过跨设备同步均值和方差，确保所有设备使用全局一致的统计量。

```
backbone=dict(
    type='ResNet',
    depth=50,
    norm_cfg=dict(type='SyncBN'),
    zero_init_residual=True),
```

### **5. 与同期方法（如MoCo）的对比**
| **维度**         | **SimCLR**                     | **MoCoV1**                     |
|------------------|--------------------------------|--------------------------------|
| **负样本来源**   | 同批次其他样本                 | 动态队列（历史批次）           |
| **训练方式**     | 端到端（依赖大批量）           | 动量编码器+队列（支持小批量）  |
| **投影头**       | 包含MLP投影头                  | 初始版本无，MoCoV2后引入       |
| **计算成本**     | 高（需大批量）                 | 低（队列机制降低计算需求）     |

## MoCo V2 (2020)

<div align=center>
<img  src="https://user-images.githubusercontent.com/36138628/149720067-b65e5736-d425-45b3-93ed-6f2427fc6217.png" width="500" />
</div>

参考博文[源码解析MoCo-v2：动量对比学习的加强版本](https://zhuanlan.zhihu.com/p/661467769)。

MocoV2在保持MocoV1核心框架（动量编码器、动态队列）的基础上，借鉴了SimCLR的一些即插即用的方法（增加MLP头+更强的数据增强），MoCo-v2算法更加亲民（只需要一台8卡机，SimCLR需要8台）；MoCo-v2计算效率更高，因为负对是从字典中直接获取的，而SimCLR是需要大量计算的。

### 1. **引入MLP投影头（Projection Head）**
   - **改进动机**：MocoV1直接使用编码器输出的特征进行对比学习，而特征空间可能未充分优化。
   - **解决方案**：在编码器后增加一个**多层感知机（MLP）**作为投影头，将特征映射到更适合对比学习的低维空间。MLP结构通常为“线性层 → ReLU → 线性层”，增强非线性表达能力。  

```
class MoCoV2Neck(BaseModule):
    def __init__(self,
                 in_channels: int,
                 hid_channels: int,
                 out_channels: int,
                 with_avg_pool: bool = True,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def forward(self, x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return (self.mlp(x.view(x.size(0), -1)), )
```

### 2. **更强的数据增强策略**
   - **改进动机**：MocoV1使用的数据增强（如随机裁剪、翻转）相对简单，限制了对特征不变性的学习。
   - **解决方案**：引入**更复杂的增强组合**，包括**颜色失真**（Color Jittering）、**高斯模糊**（Gaussian Blur）、**随机灰度化**（Random Grayscale）等。


### 3. **余弦学习率调度（Cosine Learning Rate Schedule）**
   - **改进动机**：固定学习率可能导致训练后期收敛不稳定。
   - **解决方案**：采用**余弦退火策略**调整学习率，延长训练时长至800 epoch，MoCo v2达到了71.1%，超过了SimCLR在1000轮中的69.3%。
   $$
   \eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})(1 + \cos(\pi t / T))
   $$  

### 4. **简化训练流程（去除端到端依赖）**
   - **改进动机**：MocoV1虽无需大批量，但仍需维护动态队列与动量编码器。
   - **解决方案**：结合SimCLR的部分思想（如MLP头和增强策略），但**摒弃SimCLR对大批量的依赖**，保留Moco的队列机制，在更小的批量（如256）下实现高性能。  

### 5. **与MocoV1的对比**
| **改进点**       | **MocoV1**               | **MocoV2**                     |
|------------------|--------------------------|---------------------------------|
| **投影头**       | 无                       | 添加MLP投影头                   |
| **数据增强**     | 基础增强（裁剪、翻转）   | 强化增强（模糊、颜色失真等）     |
| **学习率调度**   | 固定学习率               | 余弦退火调度                    |
| **负样本规模**   | 队列容量10,000+          | 保持队列机制，进一步优化特征质量 |

## BYOL (2020)

<div align=center>
<img src="https://user-images.githubusercontent.com/36138628/149720208-5ffbee78-1437-44c7-9ddb-b8caab60d2c3.png" width="800" />
</div>

BYOL（Bootstrap Your Own Latent）通过独特的架构设计和训练机制，摒弃了传统对比学习对负样本的依赖。

### 1. **无需负样本的对比机制**
   - **问题背景**：传统对比学习（如SimCLR、MoCo）依赖大量负样本来防止模型坍塌（即所有特征退化为相同向量），但负样本的构造增加了计算复杂度和内存消耗。
   - **解决方案**：BYOL完全**摒弃负样本**，仅利用同一图像的两个增强视图（正样本对）进行训练。通过**在线网络（Online Network）**和**目标网络（Target Network）**的协同优化，强制模型从不同增强视图中提取一致的特征，避免坍塌。

### 2. **双网络架构与动量更新**
   - **在线网络与目标网络**：  
     - **在线网络**：包含编码器（Encoder）、投影头（Projection Head）和预测器（Predictor），通过梯度下降更新参数。  
     - **目标网络**：结构与在线网络相同（无预测器），参数通过在线网络的**指数移动平均（EMA）**更新，公式为：  
       $$
       \theta_{\text{target}} \leftarrow \tau \cdot \theta_{\text{target}} + (1 - \tau) \cdot \theta_{\text{online}}
       $$  
       （$\tau$ 接近1，如0.99，确保目标网络参数缓慢变化）。  
   - **作用**：目标网络提供稳定的学习目标，引导在线网络学习鲁棒特征。


### 3. **预测器（Predictor）与停止梯度（Stop-Gradient）**
   - **预测器的作用**：  
     - 在线网络的预测器将投影特征映射到与目标网络特征对齐的空间，增强特征调整能力。  
     - 防止目标网络的特征被直接模仿，迫使在线网络学习更泛化的表示。  
   - **停止梯度操作**：  
     - 在计算目标网络的输出时，**冻结其梯度**（不反向传播），避免目标网络参数被在线网络干扰，维持其稳定性。

### 4. **对称损失函数**
   - **损失设计**：  
     - 对同一图像的两个增强视图分别计算预测误差，损失函数为对称的均方误差（MSE）：  
       $$
       \mathcal{L} = \| q(z_{\text{online}}^1) - z_{\text{target}}^2 \|_2^2 + \| q(z_{\text{online}}^2) - z_{\text{target}}^1 \|_2^2
       $$  
       （$q$为预测器，$z_{\text{online}}$和$z_{\text{target}}$为在线网络和目标网络的特征）。  
   - **优点**：通过对称优化增强特征一致性，提升训练稳定性。

### 5. **对数据增强的鲁棒性**
   - **增强策略**：BYOL使用标准增强组合（如随机裁剪、颜色失真、模糊等），但通过双网络架构减少了对增强强度的敏感度。  
   - **关键发现**：即使移除某些增强（如颜色失真），BYOL仍能保持较高性能，而对比学习方法（如SimCLR）性能显著下降。

### 6. **计算效率与性能优势**
   - **无需大批量**：与SimCLR（需4096批量）不同，BYOL在小批量（如256）下即可高效训练。  
   - **性能表现**：在ImageNet线性评估中，BYOL的ResNet-50达到74.3% Top-1准确率，超越同期对比学习方法（如SimCLR的69.3%），且接近监督学习（76.5%）。


### 7. **与对比学习方法的差异**
| **维度**         | **BYOL**                     | **SimCLR/MoCo**               |
|------------------|------------------------------|-------------------------------|
| **负样本依赖**   | 无需负样本                   | 依赖负样本                    |
| **网络结构**     | 双网络+预测器               | 单网络或双网络（无预测器）    |
| **训练目标**     | 特征一致性（MSE损失）        | 对比正负样本（InfoNCE损失）   |
| **计算成本**     | 低（小批量高效）             | 高（大批量或队列管理）        |

## SWAV (2020)

<div align=center>
<img  src="https://user-images.githubusercontent.com/36138628/149724517-9f1e7bdf-04c7-43e3-92f4-2b8fc1399123.png" width="500" />
</div>

SWAV（Swapping Assignments between Views）是一种基于**在线聚类**的自监督学习方法，其核心创新在于通过**交换不同增强视图的聚类分配**实现特征学习，避免了传统对比学习对显式负样本的依赖，同时提升了计算效率。

### 1. **在线聚类与交换预测（Swapped Prediction）**
   - **动机**：传统对比学习依赖大量负样本，计算成本高；而聚类方法需离线生成伪标签，效率低。
   - **解决方案**：  
     - **在线聚类**：在训练过程中动态生成聚类中心（prototypes），将图像特征映射到聚类空间，生成伪标签。  
     - **交换预测**：对同一图像的两个增强视图（如全局视图和局部视图），交换它们的聚类分配作为监督信号。例如，强制用视图A的特征预测视图B的聚类分配，反之亦然。  
     - **损失函数**：基于交叉熵的交换损失，$P$ 为聚类分配的伪标签，$Q$ 为预测的概率分布。 
       $$
       \mathcal{L} = -\sum_{i} \left( P^{(A)}(i) \log Q^{(B)}(i) + P^{(B)}(i) \log Q^{(A)}(i) \right)
       $$  

### 2. **Sinkhorn-Knopp算法优化聚类分配**
   - **问题背景**：直接对聚类分配进行硬标签（如K-means）会导致类别不平衡或训练不稳定。  
   - **创新点**：  
     - 使用**Sinkhorn-Knopp（SK）算法**对聚类分配矩阵进行**均匀化约束**，确保每个批次中样本均匀分配到不同聚类中心。  
     - 通过迭代归一化（行归一化+列归一化）生成平滑的伪标签分布，避免某些聚类中心被过度使用。  
     - **优势**：提升聚类的多样性和稳定性，防止模型坍塌（所有样本分配到同一聚类）。

```
for i, crop_id in enumerate(self.crops_for_assign):
    with torch.no_grad():
        out = output[bs * crop_id:bs * (crop_id + 1)].detach()
        # time to use the queue
        if self.queue is not None:
            if self.use_queue or not torch.all(self.queue[i, -1, :] == 0):
                self.use_queue = True
                out = torch.cat((torch.mm(self.queue[i], self.prototypes.weight.t()), out))
            # fill the queue
            self.queue[i, bs:] = self.queue[i, :-bs].clone()
            self.queue[i, :bs] = embedding[crop_id * bs:(crop_id + 1) * bs]

        # get assignments (batch_size * num_prototypes)
        q = distributed_sinkhorn(out, self.sinkhorn_iterations, self.world_size, self.epsilon)[-bs:]

    # cluster assignment prediction
    subloss = 0
    for v in np.delete(np.arange(np.sum(self.num_crops)), crop_id):
        x = output[bs * v:bs * (v + 1)] / self.temperature
        subloss -= torch.mean(
            torch.sum(q * nn.functional.log_softmax(x, dim=1), dim=1))
    loss += subloss / (np.sum(self.num_crops) - 1)
loss /= len(self.crops_for_assign)
```

### 3. **多裁剪策略（Multi-Crop Augmentation）**
   - **动机**：传统方法仅使用两个增强视图（如SimCLR），可能限制对局部语义的学习。  
   - **创新点**：  
     - 引入**多尺度裁剪**：生成一个全局视图（覆盖大部分图像）和多个局部视图（小区域裁剪）。  
     - 仅对全局视图之间或全局与局部视图之间进行交换预测，降低计算开销。  
     - **作用**：增强模型对局部细节和全局结构的感知能力，提升特征鲁棒性。

### 4. **小批量友好的高效训练**
   - **优化设计**：  
     - 聚类中心（prototypes）作为可学习的参数，与编码器联合优化，无需存储历史样本或队列。  
     - 通过小批量（如256）即可实现稳定训练，无需SimCLR式的大批量（4096）。  
   - **计算优势**：相比对比学习方法（需计算大量负样本相似度），SWAV的计算复杂度显著降低。


### 5. **性能与影响**
   - **ImageNet线性评估**：ResNet-50达到75.3% Top-1准确率（优于SimCLR的69.3%和BYOL的74.3%）。  
   - **低资源迁移**：在小样本（1% ImageNet）场景下表现优异，验证了特征的强泛化能力。  
   - **效率**：训练速度比SimCLR快约2倍，显存占用更低。
   
### **与传统方法的对比**
| **维度**         | **SWAV**                          | **SimCLR/MoCo**                | **BYOL**                       |
|------------------|-----------------------------------|--------------------------------|--------------------------------|
| **监督信号**     | 聚类分配交换                     | 正负样本对比                  | 特征一致性（无负样本）         |
| **负样本依赖**   | 无需显式负样本                   | 依赖负样本                    | 无需负样本                     |
| **计算开销**     | 低（小批量+聚类优化）            | 高（大批量或队列）            | 中等（双网络+动量更新）        |
| **增强策略**     | 多裁剪（全局+局部视图）          | 双视图标准增强                | 双视图标准增强                 |

## DenseCL (2021)

<div align=center>
<img src="https://user-images.githubusercontent.com/36138628/149721111-bab03a6d-a30d-418e-b338-43c3689cfc65.png" width="900" />
</div>

DenseCL（Dense Contrastive Learning）是一种针对密集预测任务（如目标检测、语义分割）设计的自监督学习方法，其核心创新在于将对比学习从图像级别扩展到**像素或局部区域级别**，从而提升模型对细粒度特征的捕捉能力。

### 1. **密集区域对比学习（Dense Region Contrast）**
   - **动机**：传统对比学习（如MoCo、SimCLR）仅关注全局图像特征，难以满足密集任务（检测、分割）对局部细节的需求。
   - **解决方案**：  
     - 在图像中随机采样**多个局部区域**（如RoI区域或像素块），对每个区域生成两种增强视图。  
     - 对**同一区域的不同增强视图**特征视为正样本对，**不同区域的特征**视为负样本对，构建密集对比学习任务。  
   - **优势**：迫使网络学习局部区域的语义一致性，增强特征的空间敏感性。

### 2. **全局-局部对比双任务框架**
   - **架构设计**：  
     - **全局对比任务**：沿用MoCo的全局图像对比，学习整体语义表示。  
     - **局部对比任务**：新增密集区域对比，优化局部特征判别力。  
     - 两个任务共享同一编码器，通过联合训练实现多粒度特征学习。  
   - **损失函数**：结合全局和局部对比损失，公式为：  
     $$
     \mathcal{L} = \mathcal{L}_{\text{global}} + \lambda \mathcal{L}_{\text{local}}
     $$  
     - $\mathcal{L}_{\text{global}}$为全局InfoNCE损失，$\mathcal{L}_{\text{local}}$为局部对比损失，$\lambda$为平衡系数。

### 3. **动态区域采样与匹配策略**
   - **区域采样**：  
     - 使用**随机区域提议**或**均匀网格划分**生成局部区域，覆盖不同尺度和位置。  
     - 对每个区域应用独立的数据增强（如裁剪、颜色变换），增强多样性。  
   - **特征匹配**：  
     - 匹配的规则是第一个view中的每个特征向量与另一个view中具有最高相似度的向量进行匹配（其实就是，从两张图中的一张上某位置利用特征来找另一张图中与其最相似的位置。
```
# feat point set sim
backbone_sim_matrix = torch.matmul(q_b.permute(0, 2, 1), k_b)
densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]  # NxS^2
indexed_k_grid = torch.gather(k_grid, 2, densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1))  # NxCxS^2
densecl_sim_q = (q_grid * indexed_k_grid).sum(1)  # NxS^2

# dense positive logits: NS^2X1
l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)

q_grid = q_grid.permute(0, 2, 1)
q_grid = q_grid.reshape(-1, q_grid.size(2))
# dense negative logits: NS^2xK
l_neg_dense = torch.einsum('nc,ck->nk', [q_grid, self.queue2.clone().detach()])
```

### 4. **与传统对比学习的对比**
| **维度**         | **传统对比学习（MoCo/SimCLR）** | **DenseCL**                     |
|------------------|---------------------------------|----------------------------------|
| **对比粒度**     | 图像级别                        | 图像级别 + 区域级别              |
| **负样本来源**   | 全局图像特征                    | 全局特征 + 局部区域特征          |
| **任务目标**     | 全局语义一致性                  | 全局-局部多粒度一致性            |
| **适用场景**     | 分类任务                        | 检测、分割等密集任务             |

## SimSiam (2021)

<div align=center>
<img  src="https://user-images.githubusercontent.com/36138628/149724180-bc7bac6a-fcb8-421e-b8f1-9550c624d154.png" width="500" />
</div>

SimSiam（Simple Siamese Network）核心创新在于通过**极简架构**和**停止梯度（Stop-Gradient）**机制，在不依赖负样本、动量编码器或大批量训练的条件下，实现了高性能的特征学习。

### 1. **停止梯度（Stop-Gradient）机制**
   - **问题背景**：传统Siamese网络（如BYOL）依赖动量编码器防止模型坍塌（输出退化为常数），但SimSiam发现**停止梯度操作**足以替代动量更新。
   - **解决方案**：  
     - 对目标分支（Target Branch）的输出**冻结梯度**（不参与反向传播），迫使在线分支（Online Branch）主动学习特征表示。  
     - 公式化对比损失如下，$p_1, p_2$为在线分支的预测输出，$z_1, z_2$为目标分支的特征，$D$为相似度度量（如余弦相似度）。
       $$
       \mathcal{L} = \frac{1}{2} \left[ D(p_1, \text{stop\_grad}(z_2)) + D(p_2, \text{stop\_grad}(z_1)) \right]
       $$  
   - **作用**：通过非对称梯度传播，避免模型陷入平凡解（如所有特征相同）。

```
class SimSiam(BaseSelfSupervisor):
    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        assert isinstance(inputs, list)
        img_v1 = inputs[0]
        img_v2 = inputs[1]

        z1 = self.neck(self.backbone(img_v1))[0]  # NxC
        z2 = self.neck(self.backbone(img_v2))[0]  # NxC

        loss_1 = self.head.loss(z1, z2)
        loss_2 = self.head.loss(z2, z1)

        losses = dict(loss=0.5 * (loss_1 + loss_2))
        return losses
```

### 2. **极简对称架构**
   - **双分支结构**：  
     - **在线分支**：包含编码器（Encoder）和预测头（Predictor），参数通过梯度更新。  
     - **目标分支**：仅包含编码器（与在线分支共享参数），输出时冻结梯度。  
   - **对称损失**：对同一图像的两个增强视图分别计算预测误差，确保特征一致性。  
   - **无动量编码器**：摒弃BYOL的动量更新机制，简化训练流程。

### 3. **预测头（Predictor）的关键作用**
   - **结构设计**：预测头为轻量级MLP（如2层全连接），将在线分支的特征映射到与目标分支对齐的空间。  
   - **必要性**：实验表明，移除预测头会导致模型坍塌，因其迫使在线分支学习非平凡变换，避免直接复制目标特征。

### 4. **无需负样本与大批量**
   - **对比学习简化**：仅依赖正样本对的相似性优化，无需构造负样本（如SimCLR）或维护队列（如MoCo）。  
   - **小批量友好**：在批量小至256时仍能稳定训练，显著降低计算成本（相比SimCLR需4096批量）。

### 5. **理论解释与实验验证**
   - **坍塌避免机制**：通过停止梯度强制在线分支与目标分支的参数更新解耦，形成动态平衡。  
   - **隐式优化目标**：类比EM算法，将特征学习分解为“预测”和“投影”交替优化。  
   - **性能表现**：在ImageNet线性评估中，ResNet-50达到71.3% Top-1准确率，接近BYOL（74.3%）且远超SimCLR（69.3%）。

### 6. **与BYOL的对比**
| **维度**         | **SimSiam**                    | **BYOL**                       |
|------------------|--------------------------------|--------------------------------|
| **动量编码器**   | 无（共享编码器+停止梯度）      | 有（动量更新目标网络）         |
| **预测头**       | 必需（防止坍塌）               | 必需（特征对齐）               |
| **负样本**       | 无需                           | 无需                           |
| **参数量**       | 更少（无动量分支）             | 更多（维护动量编码器）         |
| **理论复杂度**   | 更低                           | 较高                           |

## MoCo V3 (2021)

参考博文[MoCo V3：视觉自监督迎来Transformer](https://zhuanlan.zhihu.com/p/543222924)。

MocoV3（Momentum Contrast V3）核心创新在于将**Vision Transformer（ViT）**引入对比学习框架，并针对自监督训练中ViT的稳定性问题提出了优化策略。相较于MocoV2（基于CNN），**取消Memory Queue，引入预测头（Prediction head）**。

### 1. **Vision Transformer（ViT）作为主干网络**
   - **动机**：此前对比学习主要依赖CNN（如ResNet），而ViT在监督学习中的表现已超越CNN，但其在自监督场景下的潜力未被充分挖掘。
   - **解决方案**：将ViT作为默认编码器，通过对比学习预训练ViT的Patch Embedding层与Transformer层；证明ViT在无监督学习中同样具备强大的特征提取能力，且能通过对比学习解决ViT对大规模标注数据的依赖问题。

![参考图片](https://pic3.zhimg.com/v2-3ce6edec7d95f0203541f2584bea9844_1440w.jpg)

### 2. **训练稳定性优化**
   - **问题发现**：ViT在自监督对比学习中容易出现训练**崩溃（Collapse）**，表现为特征相似度矩阵的急剧退化（如对角线元素接近1，非对角线接近0）。  
   - **改进策略**：**random patch projection**，冻结patch embedding层参数。

```
def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
          **kwargs) -> Dict[str, torch.Tensor]:
    assert isinstance(inputs, list)
    view_1 = inputs[0]
    view_2 = inputs[1]

    # compute query features, [N, C] each
    q1 = self.neck(self.backbone(view_1))[0]
    q2 = self.neck(self.backbone(view_2))[0]

    # compute key features, [N, C] each, no gradient
    with torch.no_grad():
        # update momentum encoder
        self.momentum_encoder.update_parameters(
            nn.Sequential(self.backbone, self.neck))

        k1 = self.momentum_encoder(view_1)[0]
        k2 = self.momentum_encoder(view_2)[0]

    loss = self.head.loss(q1, k2) + self.head.loss(q2, k1)

    losses = dict(loss=loss)
    return losses
```
