本文主要整理经典的多模态对比学习算法。

## CLIP (2021)

参考博文[多模态超详细解读 (一)：CLIP：大规模语言-图像对比预训练实现不俗 Zero-Shot 性能](https://zhuanlan.zhihu.com/p/625165635)。

<div align=center>
<img src="https://raw.githubusercontent.com/Scarecrow0/figures_cache/main/clip_main_fig.png" width="100%"/>
</div>

CLIP（Contrastive Language–Image Pretraining）是由OpenAI提出的多模态学习模型，其核心创新在于通过**大规模图像-文本对对比学习**，实现了跨模态语义对齐，并显著提升了零样本（zero-shot）泛化能力。以下是其主要创新点：

### **1. 多模态对比学习框架**
- **目标设计**：  
  - 将图像和文本映射到**共享的嵌入空间**，通过对比损失最大化匹配的图像-文本对的相似性，最小化不匹配对的相似性。  
  - 损失函数（InfoNCE）：  
    $$
    \mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(I_i \cdot T_i / \tau)}{\sum_{j=1}^N \exp(I_i \cdot T_j / \tau)}
    $$
    - $I_i$为图像特征，$T_i$为文本特征，$\tau$为温度系数。

- **模型结构**：  
  - **双编码器架构**：  
    - **图像编码器**：支持ResNet或Vision Transformer（ViT）提取视觉特征。  
    - **文本编码器**：基于Transformer提取文本特征。  
  - 两模态编码器独立训练，仅通过对比损失对齐特征空间。

```
    image_features = self.extract_image_feat(images=images)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = image_features @ self.text_prototype_embeds.to(
        image_features.device) * self.logit_scale.exp()

    pred_scores = F.softmax(logits_per_image, dim=1)
    pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()
```

### **2. 零样本迁移能力**
- **动态分类器生成**：  
  - 将分类任务转化为**图像-文本匹配问题**，通过文本提示（prompt）生成类别标签的描述（如“一张{类别}的照片”），无需固定类别标签。  
  - 示例：在ImageNet分类中，直接对比图像与所有类别文本描述的相似性，选择最高分作为预测结果。  
- **优势**：  
  - 无需微调即可适应新任务，突破传统模型对预定义类别的依赖。  
  - 在30个视觉数据集上的零样本分类表现接近全监督模型。

```
# prompt示例

OPENAI_IMAGENET_PROMPT_SUB = [
    lambda c: f'itap of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'art of the {c}.',
    lambda c: f'a photo of the small {c}.',
]
```

### **3. 超大规模数据训练**
- **数据集规模**：  
  - 使用**4亿个公开图像-文本对**（WebImageText），涵盖开放域的真实世界语义关联。  
  - 数据多样性远超传统标注数据集（如ImageNet的1400万张图）。  
- **数据清洗**：  
  - 通过过滤重复、低质量样本，提升训练数据的有效性和鲁棒性。


## Chinese CLIP (2022)

参考博文[多模态表征—CLIP及中文版Chinese-CLIP：理论讲解、代码微调与论文阅读](https://zhuanlan.zhihu.com/p/690361706)。

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/36138628/4d05e51f-d834-4ef5-bbf0-0e2f80fea461" width="80%"/>
</div>

Chinese CLIP是在OpenAI的CLIP模型基础上，针对中文语境和多模态任务进行优化的版本。

### **1. 中文适配的文本编码器**
- **中文分词与语义建模**：
  - 采用**中文Roberta**作为文本编码器，替代原版CLIP的英文Transformer，支持中文分词（如基于字符或词粒度的输入）。
  - 针对中文语法特性（如成语、缩略语）优化预训练任务，增强语义理解。
- **多粒度Prompt工程**：
  - 设计中文友好的提示模板（如“这是一张关于{类别}的图片”），减少直译英文Prompt导致的语义偏差。

```
# prompt示例

OPENAI_PROMPT = [
    lambda c: f'{c}的照片',
    lambda c: f'质量差的{c}的照片',
    lambda c: f'许多{c}的照片',
    lambda c: f'{c}的雕塑',
    lambda c: f'难以看到{c}的照片',
    lambda c: f'{c}的低分辨率照片',
    lambda c: f'{c}的渲染',
    lambda c: f'涂鸦{c}',
    lambda c: f'{c}的糟糕照片',
    ...
]
```

### **2. 大规模中文图文数据集**
- **数据构建**：
  - 大规模的中文image-text-pairs（约 2 亿规模），其中包括来自 LAION-5B 中文子集、Wukong 的中文数据、以及来自 COCO、Visual Genome 的翻译图文数据等。

### **3. 两阶段训练**
- 第一阶段：冻结 image encoder 的所有参数，只训练 text encoder，这一动作是基于一个假设：训练好的 vision backbone 已经有很强的能力来抽取视觉特征了。第一阶段的训练直到对下游任务没有明显的提升而结束；
- 第二阶段，联合训练image encoder、text encoder。


## BLIP (2022)

参考博文[多模态超详细解读 (六)：BLIP：统一理解和生成的自举多模态模型](https://zhuanlan.zhihu.com/p/627481137)。

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/236374275-94d2f94b-d9a7-4f12-b694-f15a2be00be6.png" width="90%"/>
</div>

BLIP（Bootstrapping Language-Image Pre-training）是一种面向多模态理解与生成任务的新型预训练框架，其核心创新在于通过**自举数据增强**和**多任务联合优化**，显著提升了模型在噪声数据下的鲁棒性及跨模态对齐能力。


### **1. 多模态混合架构（Unified Encoder-Decoder Architecture）**
#### **架构设计**
- **多模式兼容性**：  
  BLIP采用统一的Transformer架构，支持三种模态处理模式：  
  - **单模态编码器**：分别提取图像和文本特征（类似CLIP）。  
  - **图像-文本交叉编码器**：通过跨模态注意力实现细粒度对齐（用于理解任务）。  
  - **文本生成解码器**：以图像为条件生成文本描述（用于生成任务）。  
- **参数共享**：  
  所有模块共享底层Transformer参数，减少模型体积并增强多任务协同。

#### **优势**
- **灵活适配任务**：同一模型可无缝切换至分类、检索、生成等任务。  
- **端到端优化**：联合训练提升视觉-语言联合表征能力。


### **2. 自举数据增强（Bootstrapping Data Augmentation）**
#### **Captioner-Filter 机制**
- **Captioner（生成器）**：  
  - 利用预训练模型为未标注图像生成合成文本描述（伪标签）。  
  - 生成多样化的候选描述，扩展训练数据多样性。  
- **Filter（过滤器）**：  
  - 通过对比学习筛选高质量图像-文本对（如去除语义模糊或噪声样本）。  
  - 计算图像与合成文本的相似度，保留高置信度样本加入训练集。  

#### **迭代优化流程**
1. **初始训练**：在原始标注数据（如COCO）上预训练BLIP。  
2. **生成伪数据**：使用Captioner为未标注网络图像生成描述。  
3. **过滤噪声**：通过Filter保留高质量伪数据。  
4. **混合训练**：联合标注数据与伪数据迭代优化模型。  

#### **优势**
- **缓解数据稀缺**：利用海量未标注网络图像（如1.4亿对）提升模型泛化性。  
- **抗噪声能力**：动态过滤低质量数据，避免模型过拟合噪声。


### **3. 多任务联合预训练目标**
#### **三任务协同优化**
- **图像-文本对比学习（ITC）**：  
  - 对齐图像与文本的全局特征，优化跨模态检索能力。  
  - 损失函数：InfoNCE损失，最大化匹配对的相似性。  
- **图像-文本匹配（ITM）**：  
  - 细粒度判断图像与文本是否匹配（二分类任务），增强局部语义对齐。  
  - 通过交叉编码器计算匹配分数，使用交叉熵损失优化。  
- **条件文本生成（LM）**：  
  - 以图像为输入，自回归生成文本描述（如“这张图片描述了什么？”）。  
  - 损失函数：标准语言模型损失（交叉熵）。  

#### **优势**
- **互补学习**：ITC学习全局对齐，ITM增强局部推理，LM提升生成能力。  
- **综合性能**：在理解与生成任务上均达到SOTA（如VQA、图像描述生成）。

### **4. 下游任务应用**
#### **Image-To-Text / Text-To-Image Retrieval**
参数fast_match (bool): If False, **select topk similarity as candidates and compute the matching score**. If True, return the similarity as the matching score directly. Defaults to False.

```
def compute_score_matrix_i2t(self, img_feats, img_embeds, text_feats,
                                text_ids, text_atts):
    """Compare the score matrix for image-to-text retrieval. Every image
    should compare to all the text features.

    Args:
        img_feats (torch.Tensor): The input img feats tensor with shape
            (M, C). M stands for numbers of samples on a single GPU.
        img_embeds (torch.Tensor): The input img embeds tensor with shape
            (M, C). M stands for numbers of samples on a single GPU.
        text_feats (torch.Tensor): The input text feats tensor with shape
            (N, C). N stands for numbers of all samples on all GPUs.
        text_ids (torch.Tensor): The input tensor with shape (N, C).
        text_atts (torch.Tensor): The input tensor with shape (N, C).

    Returns:
        torch.Tensor: Score matrix of image-to-text retrieval.
    """

    # compute i2t sim matrix
    sim_matrix_i2t = img_feats @ text_feats.t()
    if self.fast_match:
        return sim_matrix_i2t

    score_matrix_i2t = torch.full((img_feats.size(0), text_feats.size(0)),
                                    -100.0).to(self.device)
    for i in track_on_main_process(
            range(img_feats.size(0)), 'Compute I2T scores...'):
        sims = sim_matrix_i2t[i]
        topk_sim, topk_idx = sims.topk(k=self.topk, dim=0)

        encoder_output = img_embeds[i].repeat(self.topk, 1, 1)
        encoder_att = torch.ones(
            encoder_output.size()[:-1], dtype=torch.long).to(self.device)
        output = self.text_backbone(
            text_ids[topk_idx],
            attention_mask=text_atts[topk_idx],
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )
        score = self.multimodal_head(
            (output.last_hidden_state[:, 0, :], ))[:, 1]
        score_matrix_i2t[i, topk_idx] = score + topk_sim

    return score_matrix_i2t

def compute_score_matrix_t2i(self, img_feats, img_embeds, text_feats,
                                text_ids, text_atts):
    """Compare the score matrix for text-to-image retrieval. Every text
    should compare to all the image features.

    Args:
        img_feats (torch.Tensor): The input img feats tensor with shape
            (M, C). M stands for numbers of samples on a single GPU.
        img_embeds (torch.Tensor): The input img embeds tensor with shape
            (M, C). M stands for numbers of samples on a single GPU.
        text_feats (torch.Tensor): The input text feats tensor with shape
            (N, C). N stands for numbers of all samples on all GPUs.
        text_ids (torch.Tensor): The input tensor with shape (M, C).
        text_atts (torch.Tensor): The input tensor with shape (M, C).

    Returns:
        torch.Tensor: Score matrix of text-to-image retrieval.
    """

    # compute t2i sim matrix
    sim_matrix_t2i = text_feats @ img_feats.t()
    if self.fast_match:
        return sim_matrix_t2i

    score_matrix_t2i = torch.full((text_feats.size(0), img_feats.size(0)),
                                    -100.0).to(self.device)
    for i in track_on_main_process(
            range(text_feats.size(0)), 'Compute T2I scores...'):
        sims = sim_matrix_t2i[i]
        topk_sim, topk_idx = sims.topk(k=self.topk, dim=0)

        encoder_output = img_embeds[topk_idx]
        encoder_att = torch.ones(
            encoder_output.size()[:-1], dtype=torch.long).to(self.device)
        output = self.text_backbone(
            text_ids[i].repeat(self.topk, 1),
            attention_mask=text_atts[i].repeat(self.topk, 1),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )
        score = self.multimodal_head(
            (output.last_hidden_state[:, 0, :], ))[:, 1]
        score_matrix_t2i[i, topk_idx] = score + topk_sim

    return score_matrix_t2i
```

#### **Image Caption**

```
def predict(self, images, data_samples=None, **kwargs):
    """Predict captions from a batch of inputs.

    Args:
        images (torch.Tensor): The input images tensor with shape
            (N, C, ...) in general.
        data_samples (List[DataSample], optional): The annotation
            data of every samples. Defaults to None.
        **kwargs: Other keyword arguments accepted by the ``predict``
            method of :attr:`head`.

    Returns:
        List[DataSample]: Return list of data samples.
    """
    # prepare inputs for decoder generation.
    image_embeds = self.visual_encoder(images)[0]
    # 将每个图像的特征沿批次维度复制 self.num_captions 次，为每个图像生成多个候选描述（如生成3个不同标题）
    image_embeds = torch.repeat_interleave(image_embeds, self.num_captions,
                                            0)

    prompt = [self.prompt] * image_embeds.size(0)
    # padding='longest'：填充至批次内最长文本长度。
    prompt = self.tokenizer(
        prompt, padding='longest',
        return_tensors='pt').to(image_embeds.device)

    # bos_token_id：将首字符设为序列开始符（如 [CLS] 或 <s>）
    prompt.input_ids[:, 0] = self.tokenizer.bos_token_id
    # 截断最后一个字符（可能是为了移除默认添加的结束符）
    prompt.input_ids = prompt.input_ids[:, :-1]

    # encoder_hidden_states=image_embeds：将图像特征作为解码器的交叉注意力输入
    # sep_token_id 和 pad_token_id：控制生成终止和填充
    decoder_out = self.seq_gen_head.predict(
        input_ids=prompt.input_ids,
        encoder_hidden_states=image_embeds,
        sep_token_id=self.tokenizer.sep_token_id,
        pad_token_id=self.tokenizer.pad_token_id,
        output_attentions=True,
        return_dict_in_generate=True,
    )

    # batch_decode：将Token ID序列转换为字符串，跳过特殊符号（如 [PAD], [SEP]）
    decode_tokens = self.tokenizer.batch_decode(
        decoder_out.sequences, skip_special_tokens=True)

    out_data_samples = []
    if data_samples is None:
        data_samples = [None for _ in range(len(decode_tokens))]

    for data_sample, decode_token in zip(data_samples, decode_tokens):
        if data_sample is None:
            data_sample = DataSample()
        # decode_token[len(self.prompt):]：去除输入提示部分，仅保留生成的描述（如输入提示为“描述这张图片：”，生成结果为“一只猫在沙发上”，最终提取“一只猫在沙发上”）。
        data_sample.pred_caption = decode_token[len(self.prompt):]
        out_data_samples.append(data_sample)

    return out_data_samples
```

#### **Visual Question Answering**

![参考图片](https://pic4.zhimg.com/v2-4bd0f5503879975e1d4c49795dd96fdb_1440w.jpg)

```
# 配置文件
model = dict(
    type='BlipVQA',
    tokenizer=dict(type='BlipTokenizer', name_or_path='bert-base-uncased'),
    vision_backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=480,
        patch_size=16,
        out_type='raw'),
    multimodal_backbone=dict(
        type='XBertEncoder',
        med_config=dict(
            architectures=['BertModel'],
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            model_type='bert',
            num_attention_heads=12,
            num_hidden_layers=12,
            pad_token_id=0,
            add_type_embeddings=False,
            vocab_size=30524,
            encoder_width=768,
            add_cross_attention=True),
    ),
    head=dict(
        type='VQAGenerationHead',
        decoder=dict(
            type='XBertLMHeadDecoder',
            med_config=dict(
                architectures=['BertModel'],
                attention_probs_dropout_prob=0.1,
                hidden_act='gelu',
                hidden_dropout_prob=0.1,
                hidden_size=768,
                initializer_range=0.02,
                intermediate_size=3072,
                layer_norm_eps=1e-12,
                max_position_embeddings=512,
                model_type='bert',
                num_attention_heads=12,
                num_hidden_layers=12,
                pad_token_id=0,
                add_type_embeddings=False,
                vocab_size=30524,
                encoder_width=768,
                add_cross_attention=True),
        ),
        inference_method='rank',  # or 'generate'
        answer_list_path=
        'https://storage.googleapis.com/sfr-vision-language-research/datasets/answer_list.json',  # noqa: E501
    ),
)
```

```
# VQA总流程
def predict(
    self,
    images: torch.Tensor,
    data_samples: Optional[List[DataSample]] = None,
):
    """update data_samples that contain pred_answer for each question.

    Args:
        images (Tensor): A batch of images. The shape of it should be
            (B, C, H, W) for images and (B, T, C, H, W) for videos.
        data_samples (List[DataSample], optional): The annotation
            data of every samples.

    Returns:
        Dict[torch.Tensor]: The losses features.
    """
    visual_embeds = self.extract_feat(images)
    # image_atts：生成全1的注意力掩码，表示所有视觉特征位置均有效（无填充）
    image_atts = torch.ones(
        visual_embeds.size()[:-1], dtype=torch.long).to(self.device)

    questions = []
    # 从 data_samples 中提取问题文本
    for sample in data_samples:
        questions.append(sample.get('question'))
    questions = self.tokenizer(
        questions, padding='longest', return_tensors='pt').to(self.device)
    # 特殊标记用于标识问题起始，需确保分词器的词汇表中已定义该标记。
    questions.input_ids[:, 0] = \
        self.tokenizer.additional_special_tokens_ids[0]

    # multimodal fusion
    # 多模态模型通常采用交叉注意力机制（Cross-Attention），使文本关注相关视觉区域
    multimodal_embeds = self.multimodal_backbone(
        questions.input_ids,
        attention_mask=questions.attention_mask,
        encoder_hidden_states=visual_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    # Rank模式：将预定义答案列表（answer_list）转换为Token，首字符设为起始标记（bos_token_id）。
    # Generate模式：无需候选答案，直接生成自由文本。
    if self.vqa_head.inference_method == 'rank':
        answer_candidates = self.tokenizer(
            self.vqa_head.answer_list,
            padding='longest',
            return_tensors='pt').to(self.device)
        answer_candidates.input_ids[:, 0] = self.tokenizer.bos_token_id
    elif self.vqa_head.inference_method == 'generate':
        answer_candidates = None

    head_feats = dict(
        multimodal_embeds=multimodal_embeds.last_hidden_state,
        question_atts=questions.attention_mask,
        answer_candidates=answer_candidates,
        bos_token_id=self.tokenizer.bos_token_id,
        sep_token_id=self.tokenizer.sep_token_id,
        pad_token_id=self.tokenizer.pad_token_id,
    )

    # Rank模式：计算多模态特征与候选答案的匹配分数，选择最高分答案
    if self.vqa_head.inference_method == 'rank':
        answers = self.vqa_head.predict(head_feats)
        for answer, data_sample in zip(answers, data_samples):
            data_sample.pred_answer = answer

    # Generate模式：以多模态特征为条件，自回归生成答案文本（类似GPT）。
    elif self.vqa_head.inference_method == 'generate':
        outputs = self.vqa_head.predict(head_feats)
        for output, data_sample in zip(outputs, data_samples):
            data_sample.pred_answer = self.tokenizer.decode(
                output, skip_special_tokens=True)

    return data_samples
```

```
def predict_rank(self, feats: dict, data_samples=None):
    """Predict rank in a close-set answer list."""
    question_states = feats['multimodal_embeds']
    question_atts = feats['question_atts']
    answer_candidates = feats['answer_candidates']
    assert answer_candidates is not None

    answer_ids = answer_candidates.input_ids
    answer_atts = answer_candidates.attention_mask
    num_ques = question_states.size(0)
    # 使用第一个候选答案的起始符（BOS）初始化，用于生成所有问题的初始预测
    start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

    start_output = self.decoder(
        start_ids,
        encoder_hidden_states=question_states,
        encoder_attention_mask=question_atts,
        return_dict=True,
        reduction='none',
    )
    # 第一个生成位置的logits [B, vocab_size]
    logits = start_output.logits[:, 0, :]  # first token's logit

    # topk_probs: top-k probability
    # topk_ids: [num_question, k]
    # 所有候选答案的第二个Token（首词）[C]
    answer_first_token = answer_ids[:, 1]
    prob_first_token = F.softmax(
        logits, dim=1).index_select(
            dim=1, index=answer_first_token)
    # 对每个问题，选择首词概率最高的 K 个候选答案。
    topk_probs, topk_ids = prob_first_token.topk(
        self.num_ans_candidates, dim=1)

    # answer input: [num_question*k, answer_len]
    input_ids = []
    input_atts = []
    for b, topk_id in enumerate(topk_ids):
        input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
        input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
    input_ids = torch.cat(input_ids, dim=0)
    input_atts = torch.cat(input_atts, dim=0)
    # 忽略填充位置
    targets_ids = input_ids.masked_fill(input_ids == feats['pad_token_id'],
                                        -100)

    def tile(x, dim, n_tile):
        init_dim = x.size(dim)
        repeat_idx = [1] * x.dim()
        repeat_idx[dim] = n_tile
        x = x.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([
                init_dim * np.arange(n_tile) + i for i in range(init_dim)
            ]))
        return torch.index_select(x, dim, order_index.to(x.device))

    # repeat encoder's output for top-k answers
    # 将每个问题的多模态特征复制 K 次，以匹配 B*K 个候选答案
    question_states = tile(question_states, 0, self.num_ans_candidates)
    question_atts = tile(question_atts, 0, self.num_ans_candidates)

    #  Labels for computing the left-to-right language modeling loss (next word prediction)
    output = self.decoder(
        input_ids,
        attention_mask=input_atts,
        encoder_hidden_states=question_states,
        encoder_attention_mask=question_atts,
        labels=targets_ids,
        return_dict=True,
        reduction='none',
    )

    log_probs_sum = -output.loss
    # 损失越低，概率越高
    log_probs_sum = log_probs_sum.view(num_ques, self.num_ans_candidates)

    max_topk_ids = log_probs_sum.argmax(dim=1)
    # 转换为原始候选索引
    max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]

    answers = [self.answer_list[max_id] for max_id in max_ids]

    return answers
```

#### **Natural Language Visual Reasoning, NLVR2**

要求模型预测一个句子是否描述了一对图像，是个二分类任务。

![参考图片](https://pic2.zhimg.com/v2-0dbae041e4306be06a52207505e47f6b_1440w.jpg)

```
# NLVR2总流程
def predict(self, images, data_samples=None):
    """Predict caption."""
    # prepare inputs for decoder generation.
    image_embeds = self.vision_backbone(images)[0]
    texts = self.preprocess_text(data_samples)
    image_atts = torch.ones(
        image_embeds.size()[:-1], dtype=torch.long).to(self.device)

    image0_embeds, image1_embeds = torch.split(image_embeds,
                                                texts.input_ids.size(0))

    # multimodal fusion
    multimodal_embeds = self.multimodal_backbone(
        texts.input_ids,
        attention_mask=texts.attention_mask,
        encoder_hidden_states=[image0_embeds, image1_embeds],
        encoder_attention_mask=[
            image_atts[:image0_embeds.size(0)],
            image_atts[image0_embeds.size(0):],
        ],
        return_dict=True,
    )

    # get prediction
    outputs = self.head(multimodal_embeds.last_hidden_state[:, 0, :])

    pred_scores = F.softmax(outputs, dim=1)

    for pred_score, data_sample in zip(pred_scores, data_samples):
        data_sample.set_pred_score(pred_score)
        data_sample.set_pred_label(pred_score.argmax(dim=0))

    return data_samples
```

#### **Visual Grounding**

```
# 配置文件
model = dict(
    type='BlipGrounding',
    visual_encoder=dict(
        type='VisionTransformer',
        arch='b',
        img_size=384,
        patch_size=16,
        out_type='raw',
    ),
    text_encoder=dict(
        type='XBertEncoder',
        med_config=med_config,
    ),
    multimodal_encoder=dict(
        type='XBertEncoder',
        med_config=med_config,
    ),
    tokenizer=dict(type='BlipTokenizer', name_or_path='bert-base-uncased'),
    head=dict(
        type='GroundingHead',
        decoder=dict(
            type='XBertLMHeadDecoder',
            med_config=med_config,
        ),
        box_l1_loss_coeff=4.0,
        box_giou_loss_coeff=2.0,
    ),
)
```

```
# 推理总流程
def predict(self, images, data_samples=None):
    """"""

    # extract image feature
    image_embeds = self.extract_feat(images)
    image_atts = image_embeds.new_ones(
        image_embeds.size()[:-1], dtype=torch.long)

    raw_text = []
    for ds in data_samples:
        raw_text.append(ds.text)

    text = self.tokenizer(
        raw_text,
        padding='longest',
        truncation=True,
        max_length=128,
        return_tensors='pt',
    ).to(image_embeds.device)

    text_embeds = self.text_encoder(
        text.input_ids,
        attention_mask=text.attention_mask,
        mode='text',
        return_dict=True)  # bz, seq_len, hid

    # multimodal fusion
    multimodal_embeds = self.multimodal_encoder(
        encoder_embeds=text_embeds.last_hidden_state,
        attention_mask=text.attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    # put answer from data_samples into tensor form
    output_boxes = self.grounding_head.predict(
        text_embedding=multimodal_embeds.last_hidden_state,
        text_embedding_mask=text.attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
    )  # xyxy 0-1

    out_data_samples = []
    for bbox, data_sample, img in zip(output_boxes, data_samples, images):
        if data_sample is None:
            data_sample = DataSample()

        img_size = img.shape[-2:]
        scale_factor = data_sample.get('scale_factor', (1, 1))
        bbox[0::2] = bbox[0::2] * img_size[1] / scale_factor[0]
        bbox[1::2] = bbox[1::2] * img_size[0] / scale_factor[1]
        bbox = bbox[None, :]
        data_sample.pred_bboxes = bbox

        if 'gt_bboxes' in data_sample:
            gt_bboxes = torch.Tensor(data_sample.get('gt_bboxes'))
            gt_bboxes[:, 0::2] /= scale_factor[0]
            gt_bboxes[:, 1::2] /= scale_factor[1]
            data_sample.gt_bboxes = gt_bboxes

        out_data_samples.append(data_sample)

    return out_data_samples
```

### **5. 与CLIP、ALIGN的对比**
| **维度**         | **CLIP/ALIGN**                | **BLIP**                          |
|------------------|-------------------------------|-----------------------------------|
| **架构设计**     | 双编码器（独立视觉/文本）     | 统一编码器-解码器（多任务兼容）   |
| **训练目标**     | 单一对比学习                  | 多任务联合优化（ITC+ITM+LM）      |
| **数据利用**     | 静态清洗后的标注数据          | 动态自举增强（生成+过滤伪数据）   |
| **生成能力**     | 不支持文本生成                | 支持条件文本生成                  |
| **抗噪声能力**   | 依赖高质量数据                | 主动过滤噪声，鲁棒性强            |

## BLIP v2 (2023)

参考博文[多模态超详细解读 (七)：BLIP-2：节约多模态训练成本：冻结预训练好的视觉语言模型参数](https://zhuanlan.zhihu.com/p/628375255)。

<div align=center>
<img src="https://user-images.githubusercontent.com/30762564/236385045-dc22a621-0a9c-4352-afa4-ca3888044850.png" width="70%"/>
</div>

BLIP-2（Bootstrapping Language-Image Pre-training Version 2）在多模态预训练领域提出了多项创新设计，其核心目标是通过**高效参数利用**和**轻量级架构设计**，实现大规模视觉-语言模型（VLMs）的联合训练，同时减少计算成本。

### **1. 轻量级查询 Transformer（Querying Transformer, Q-Former）**
#### **架构设计**
- **双分支结构**：  
  - **图像分支**：将冻结的视觉编码器（如ViT、EVA-CLIP）输出的图像特征输入至可训练的Q-Former。  
  - **文本分支**：将冻结的大语言模型（LLM，如FlanT5、OPT）的文本特征与Q-Former交互。  
- **Q-Former作用**：  
  - 通过一组**可学习的查询向量（learnable queries）**，从视觉编码器中提取与文本相关的关键特征。  
  - 仅需训练0.1%的参数量（约188M参数），即可桥接视觉与语言模态。

![参考图片](https://pica.zhimg.com/v2-a2a58d3db4409d4321c6d629a51725fe_1440w.jpg)

```
def _extract_feat(self, inputs: Union[torch.Tensor, dict],
                    modality: str) -> Tuple[torch.Tensor]:
    """Extract features from the single modality.
    Args:
        inputs (Union[torch.Tensor, dict]): A batch of inputs.
            For image, a tensor of shape (N, C, ...) in general.
            For text, a dict of tokenized text inputs.
        modality (str): Modality feature to be extracted. Only two
            options are supported.

            - ``images``: Only extract image features, mostly used for
                inference.
            - ``texts``: Only extract text features, mostly used for
                inference.
    Returns:
        Tuple[torch.Tensor]: The output features.
    """
    if modality == 'images':
        # extract image features
        # TODO:
        # Add layernorm inside backbone and handle the concat outside
        image_embeds = self.ln_vision_backbone(
            self.vision_backbone(inputs)[0])
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1,
                                                -1)
        query_output = self.multimodal_backbone.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        image_feat = F.normalize(
            self.vision_neck([query_output.last_hidden_state]), dim=-1)
        return {
            'image_embeds': image_embeds,
            'image_feat': image_feat,
            'query_output': query_output
        }
    elif modality == 'texts':
        # extract text features
        text_output = self.multimodal_backbone.bert(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            return_dict=True,
        )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(
            self.text_neck([text_embeds[:, 0, :]]), dim=-1)
        return {'text_embeds': text_embeds, 'text_feat': text_feat}
    else:
        raise RuntimeError(f'Invalid modality "{modality}".')
```

#### **优势**
- **参数高效**：冻结视觉和文本编码器，仅微调Q-Former和线性投影层。  
- **通用适配性**：支持任意视觉编码器与LLM组合（如ViT-G + OPT-6.7B）。

### **2. 两阶段预训练策略**
#### **阶段一：视觉-语言表征学习**
- **训练目标**：  
  - **图像-文本对比学习（ITC）**：对齐图像与文本的全局特征。  
  - **图像-文本匹配（ITM）**：细粒度判断图文是否匹配（二分类）。  
  - **基于图像的文本生成（ITG）**：以图像为条件生成文本描述。  
- **数据增强**：  
  - 利用Q-Former生成合成文本，通过ITM过滤低质量数据，形成自举增强循环。

#### **阶段二：视觉-语言生成式预训练**
- **桥接LLM**：  
  - 将Q-Former输出的视觉特征通过线性投影输入冻结的LLM，指导其生成与图像相关的文本。  
  - 训练目标为标准的语言模型损失（自回归生成）。  
- **优势**：  
  - 无需修改LLM参数，即可使其理解视觉语义。  
  - 支持零样本视觉问答、图像描述生成等任务。

![参考图片](https://pic3.zhimg.com/v2-322028c3cc0e96e9681278ada0ab0978_1440w.jpg)

### **3. 零样本推理能力**
#### **无需任务微调**
- **视觉问答（VQA）**：  
  - 将问题与图像特征输入LLM，直接生成答案（如“Q: 图中有什么？ A: [生成文本]”）。  
- **图像描述生成**：  
  - 输入提示（prompt）如“描述这张图片：”，LLM基于视觉特征生成描述。  
- **优势**：  
  - 在12个VQA数据集上零样本性能超越Flamingo-80B（仅需1/50参数量）。

### **4. 下游任务**

#### Image-To-Text Retrieval

```
# 配置文件如下，推理流程同BLIP V1
model = dict(
    type='Blip2Retrieval',
    tokenizer=dict(type='Blip2Tokenizer', name_or_path='bert-base-uncased'),
    vision_backbone=dict(
        type='BEiTViT',
        # eva-g without the final layer
        arch=dict(
            embed_dims=1408,
            num_layers=39,
            num_heads=16,
            feedforward_channels=6144,
        ),
        img_size=364,
        patch_size=14,
        layer_scale_init_value=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        final_norm=False,
        use_shared_rel_pos_bias=False,
        out_type='raw'),
    multimodal_backbone=dict(
        type='Qformer',
        model_style='bert-base-uncased',
        vision_model_width=1408,
        add_cross_attention=True,
        cross_attention_freq=2,
        num_query_token=32),
    vision_neck=dict(
        type='LinearClsHead',
        in_channels=768,
        num_classes=256,
    ),
    text_neck=dict(
        type='LinearClsHead',
        in_channels=768,
        num_classes=256,
    ),
    multimodal_head=dict(
        type='ITMHead',
        hidden_size=768,
        with_pooler=False,
    ),
    topk=128,
    max_txt_len=35,
)
```

#### Image Caption

```
# 配置文件
model = dict(
    type='Blip2Caption',
    tokenizer=dict(
        type='AutoTokenizer', name_or_path='facebook/opt-2.7b',
        use_fast=False),
    vision_backbone=dict(
        type='BEiTViT',
        # eva-g without the final layer
        arch=dict(
            embed_dims=1408,
            num_layers=39,
            num_heads=16,
            feedforward_channels=6144,
        ),
        img_size=364,
        patch_size=14,
        out_indices=-2,
        layer_scale_init_value=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        frozen_stages=39,
        final_norm=False,
        use_shared_rel_pos_bias=False,
        out_type='raw'),
    text_backbone=dict(
        type='OPTForCausalLM', name_or_path='facebook/opt-2.7b'),
    multimodal_backbone=dict(
        type='Qformer',
        model_style='bert-base-uncased',
        vision_model_width=1408,
        add_cross_attention=True,
        cross_attention_freq=2,
        num_query_token=32),
    vision_neck=dict(
        type='LinearClsHead',
        in_channels=768,
        num_classes=2560,
    ),
    prompt='a photo of',
    max_txt_len=30)
```

```
# 推理总流程
def predict(self,
            images: torch.Tensor,
            data_samples: Optional[list] = None,
            **kwargs) -> List[DataSample]:
    """Predict captions from a batch of inputs.

    Args:
        images (torch.Tensor): The input tensor with shape
            (N, C, ...) in general.
        data_samples (List[DataSample], optional): The annotation
            data of every samples. Defaults to None.
        **kwargs: Other keyword arguments accepted by the ``predict``
            method of :attr:`head`.

    Returns:
        List[DataSample]: Return list of data samples.
    """

    # extract image features
    image_embeds = self.ln_vision_backbone(self.vision_backbone(images)[0])
    image_atts = torch.ones(
        image_embeds.size()[:-1],
        dtype=torch.long,
    ).to(images.device)

    # distill image features to query tokens
    query_tokens = self.query_tokens.expand(image_embeds.size(0), -1, -1)
    query_outputs = self.multimodal_backbone.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    inputs_opt = self.vision_neck([query_outputs.last_hidden_state])
    attns_opt = torch.ones(
        inputs_opt.size()[:-1], dtype=torch.long).to(images.device)

    prompt = [self.prompt] * image_embeds.size(0)

    opt_tokens = self.tokenizer(
        prompt,
        return_tensors='pt',
        padding='longest',
        truncation=True,
        max_length=self.max_txt_len,
    ).to(images.device)
    attention_mask = torch.cat([attns_opt, opt_tokens.attention_mask],
                                dim=1)

    inputs_embeds = (
        self.text_backbone.get_input_embeddings()(opt_tokens.input_ids))
    inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

    # 基于拼接后的特征，使用束搜索（Beam Search）生成文本
    outputs = self.text_backbone.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        do_sample=False,
        top_p=0.9,
        temperature=1.,
        num_beams=5,
        max_new_tokens=self.max_txt_len,
        min_length=1,
        eos_token_id=self.eos_token_id,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=self.num_captions,
    )

    output_text = self.tokenizer.batch_decode(
        outputs, skip_special_tokens=True)
    output_text = [text.strip() for text in output_text]

    out_data_samples = []
    if data_samples is None:
        data_samples = [None for _ in range(len(output_text))]

    for data_sample, decode_token in zip(data_samples, output_text):
        if data_sample is None:
            data_sample = DataSample()
        data_sample.pred_caption = decode_token
        out_data_samples.append(data_sample)

    return out_data_samples
```

#### Visual Question Answering

```
# 配置文件
model = dict(
    type='Blip2VQA',
    tokenizer=dict(
        type='AutoTokenizer', name_or_path='facebook/opt-2.7b',
        use_fast=False),
    vision_backbone=dict(
        type='BEiTViT',
        # eva-g without the final layer
        arch=dict(
            embed_dims=1408,
            num_layers=39,
            num_heads=16,
            feedforward_channels=6144,
        ),
        img_size=364,
        patch_size=14,
        out_indices=-2,
        layer_scale_init_value=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        frozen_stages=39,
        final_norm=False,
        use_shared_rel_pos_bias=False,
        out_type='raw'),
    text_backbone=dict(
        type='OPTForCausalLM', name_or_path='facebook/opt-2.7b'),
    multimodal_backbone=dict(
        type='Qformer',
        model_style='bert-base-uncased',
        vision_model_width=1408,
        add_cross_attention=True,
        cross_attention_freq=2,
        num_query_token=32),
    vision_neck=dict(
        type='LinearClsHead',
        in_channels=768,
        num_classes=2560,
    ),
    prompt='Question: {} Answer:',
    max_txt_len=10)
```

```
# 推理总流程
def predict(self,
            images: torch.Tensor,
            data_samples: Optional[list] = None,
            **kwargs) -> List[DataSample]:
    """Predict captions from a batch of inputs.

    Args:
        images (torch.Tensor): The input tensor with shape
            (N, C, ...) in general.
        data_samples (List[DataSample], optional): The annotation
            data of every samples. Defaults to None.
        **kwargs: Other keyword arguments accepted by the ``predict``
            method of :attr:`head`.

    Returns:
        List[DataSample]: Return list of data samples.
    """
    questions = [d.question for d in data_samples]

    # extract image features from
    image_embeds = self.ln_vision_backbone(self.vision_backbone(images)[0])
    image_atts = torch.ones(
        image_embeds.size()[:-1],
        dtype=torch.long,
    ).to(images.device)

    # distill image features to query tokens
    query_tokens = self.query_tokens.expand(image_embeds.size(0), -1, -1)
    query_outputs = self.multimodal_backbone.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    inputs_opt = self.vision_neck([query_outputs.last_hidden_state])
    attns_opt = torch.ones(
        inputs_opt.size()[:-1], dtype=torch.long).to(images.device)

    prompt = [self.prompt.format(q) for q in questions]

    # use left padding
    # padding_side同capture有所调整
    self.tokenizer.padding_side = 'left'

    opt_tokens = self.tokenizer(
        prompt, return_tensors='pt', padding='longest').to(images.device)
    input_ids = opt_tokens.input_ids
    attention_mask = torch.cat([attns_opt, opt_tokens.attention_mask],
                                dim=1)

    inputs_embeds = self.text_backbone.model.decoder.embed_tokens(
        input_ids)
    inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

    # length_penalty=-1.0，通常用于鼓励生成更长的文本
    outputs = self.text_backbone.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        do_sample=False,
        num_beams=5,
        max_new_tokens=self.max_txt_len,
        min_length=1,
        eos_token_id=self.eos_token_id,
        length_penalty=-1.0,
    )

    output_text = self.tokenizer.batch_decode(
        outputs, skip_special_tokens=True)
    output_text = [text.strip() for text in output_text]

    out_data_samples = []
    for data_sample, decode_token in zip(data_samples, output_text):
        data_sample.pred_answer = decode_token
        out_data_samples.append(data_sample)

    return out_data_samples
```
