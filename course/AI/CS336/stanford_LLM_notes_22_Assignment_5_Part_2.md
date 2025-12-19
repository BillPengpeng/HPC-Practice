本文主要整理Assignment 5 (alignment)的主要内容。

## 4 Supervised Finetuning for MATH

### 内容概括
本节主要介绍了针对MATH数据集的有监督微调方法。内容涵盖了SFT算法的具体步骤（如输入初始化模型、批量采样、计算交叉熵损失和参数更新），并强调微调目标是提升模型的推理能力，而非直接预测答案。文档还提到使用了来自DeepSeek R1的链式思维推理数据，并解释了在实践中SFT常作为强化学习微调的热启动原因，包括数据需求差异和性能增益潜力。

### 要点总结
- **算法核心**：SFT算法通过最小化交叉熵损失来微调模型，输入包括初始策略模型和SFT数据集，步骤涉及批量采样和梯度更新。
- **微调目标**：重点提升模型生成链式思维推理轨迹的能力，而非直接输出答案，使用MATH数据集和DeepSeek提供的推理数据。
- **数据来源**：推理数据来自DeepSeek R1，存储在特定路径（/data/a5-alignment/MATH/sft.jsonl），用于训练模型的推理过程。
- **与RL关系**：SFT常作为RL微调的热启动，因SFT需高质量标注数据（含推理轨迹），而RL仅需答案反馈；RL可进一步优化SFT策略，但当前模型规模下两者效果需分开评估。
- **实践限制**：文档指出所用模型规模较小，SFT与RL结合效果不显著，因此作业中将两者视为独立阶段。

## 4.1 Using HuggingFace Models

### 内容概况

本节主要介绍了一个完整的模型微调工作流程指南，从**加载Hugging Face模型**，到**执行前向传播与计算损失**，再到使用**梯度累积**技术进行训练，最后**保存训练好的模型**。内容侧重于实际操作，提供了详细的理论解释和配套的PyTorch代码示例。

---

### 要点总结

1.  **模型加载与优化**：介绍了如何从本地路径加载模型和分词器，并强调了使用 `bfloat16` 精度和 `flash_attention_2` 来节省显存。
2.  **训练流程**：详细说明了如何执行前向传播以获取logits，并计算预测与真实标签之间的交叉熵损失。
3.  **梯度累积**：这是核心内容，解释了一种在显存有限的情况下模拟更大批量训练的技术。其原理是**累加多个小批量的梯度后再更新模型权重**，而非每处理一个小批量就更新一次。关键在于：
    *   将损失除以累积步数 (`k`)，使梯度被平均。
    *   每累积 `k` 步才执行一次 `optimizer.step()` 和 `optimizer.zero_grad()`。
4.  **有效批量大小**：使用梯度累积后，有效批量大小 = 实际批量大小 × 梯度累积步数 (`k`)。
5.  **模型保存**：训练完成后，需要将模型权重和分词器保存到指定目录，以便后续使用。

---

### 可打印/可执行的代码

#### 1. 加载模型与分词器

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 确保使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从本地目录加载模型和分词器，应用内存优化
model = AutoModelForCausalLM.from_pretrained(
    "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    torch_dtype=torch.bfloat16,  # 使用bfloat16精度节省显存
    attn_implementation="flash_attention_2",  # 使用FlashAttention-2加速并节省显存
).to(device)  # 将模型移至GPU

tokenizer = AutoTokenizer.from_pretrained("/data/a5-alignment/models/Qwen2.5-Math-1.5B")
```

#### 2. 前向传播与损失计算

```python
import torch.nn.functional as F

# 假设 train_batch 是一个包含 input_ids 和 labels 的批次数据
input_ids = train_batch["input_ids"].to(device)
labels = train_batch["labels"].to(device)

# 前向传播，获取logits
outputs = model(input_ids)
logits = outputs.logits  # 模型输出的预测值

# 计算损失（例如交叉熵损失）
# 注意：需要根据logits的维度和labels的维度调整F.cross_entropy的参数
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
```

#### 3. 实现梯度累积的训练循环

```python
# 设置梯度累积步数
gradient_accumulation_steps = 4

# 初始化优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()  # 设置模型为训练模式

for idx, (inputs, labels) in enumerate(data_loader):
    # 将数据移至设备
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # 前向传播
    logits = model(inputs).logits
    # 计算损失，并除以累积步数进行缩放
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1)) / gradient_accumulation_steps
    
    # 反向传播，累积梯度
    loss.backward()
    
    # 每累积 `gradient_accumulation_steps` 个批次，更新一次权重
    if (idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()        # 更新模型参数
        optimizer.zero_grad()   # 清空累积的梯度，为下一轮累积做准备

# 注意：如果数据总量不能被gradient_accumulation_steps整除，最后可能需要处理剩余的梯度
```

#### 4. 保存训练好的模型与分词器

```python
# 指定输出目录，请将 'yourusername' 替换为您的实际用户名
output_dir = "/data/yourusername/my_trained_model"

# 保存模型权重
model.save_pretrained(save_directory=output_dir)
# 保存分词器
tokenizer.save_pretrained(save_directory=output_dir)

print(f"模型与分词器已保存至: {output_dir}")
```

## 4.2 SFT Helper Methods

### Tokenizing prompts and outputs

1.  **核心任务**：实现用于SFT和RL实验的辅助方法。
2.  **关键方法一：标记化提示与输出**
    *   **流程**：对于每个问题-目标输出对 `(q, o)`，需要将问题（prompt）和输出（output/completion/response）**分别进行标记化**，然后将它们**拼接**起来。
    *   **目的**：这样处理后的序列可用于计算模型（SFT模型或RL策略）对该输出的**对数概率**。
3.  **关键方法二：构建响应掩码**
    *   **定义**：`response_mask` 是一个布尔掩码，其作用是**精确标识出序列中属于“响应/输出”的所有标记**。
    *   **规则**：对于响应部分的标记，掩码值为 `True`；对于问题部分和填充的标记，掩码值为 `False`。
    *   **核心用途**：在训练循环中，利用此掩码**确保损失函数仅对模型生成的响应部分进行计算**，而忽略提示和填充部分，这对于模型正确学习生成内容至关重要。

### Problem (tokenize_prompt_and_output): Prompt and output tokenization (2 points)

Deliverable: Implement a method tokenize_prompt_and_output that tokenizes the question and
output separately, concatenates them together, and constructs a response_mask. The following
interface is recommended:
- 完成

### Logging per-token entropies

1.  **核心目的**：在强化学习（RL）训练过程中，**记录每个生成令牌（per-token）的熵**，以此作为关键监控指标，用于判断模型的预测分布是否变得**过度自信**。
2.  **监控意义**：通过观察熵值的变化趋势，可以评估模型训练的动态。熵值持续下降可能意味着模型预测的多样性降低、变得过于确定，这可能有助于提升某些任务的准确性，但也可能导致模型过于保守或失去创造性，是调整训练过程的重要信号。
3.  **数学定义**：熵是度量概率分布不确定性的标准。对于离散概率分布 $ p(x) $，其熵 $ H(p) $ 的数学定义为：
    $$ H(p)=-\sum_{x\in\mathcal{X}}p(x)\log p(x) $$
    其中，$ \mathcal{X} $ 是词汇表，$ p(x) $ 是令牌 $ x $ 的概率。
4.  **计算方法**：在具体实现上，将利用模型（SFT或RL模型）前向传播后输出的**logits**，通过Softmax函数将其转换为概率分布，然后根据上述公式计算**每个令牌位置**的预测熵。

在代码实现中，计算“每个令牌熵”通常遵循以下步骤：
1.  获取模型输出的 logits（形状通常为 `[batch_size, sequence_length, vocab_size]`）。
2.  对 logits 在词汇表维度（`vocab_size`）上应用 Softmax，得到每个令牌的概率分布。
3.  根据熵的定义公式，计算每个序列位置（`sequence_length`）上概率分布的熵，得到形状为 `[batch_size, sequence_length]` 的熵值张量。
4.  在训练过程中记录这些熵值（如计算批次均值、绘制变化曲线），用于监控分析。

### Problem (compute_entropy): Per-token entropy (1 point)

Deliverable: Implement a method compute_entropy that computes the per-token entropy of
next-token predictions.
- 完成

### Getting log-probabilities from a model

1.  **核心目的与重要性**：从模型获取下一个词的（对数）概率，是进行模型微调（SFT）和基于梯度的策略优化（RL）的**基础性操作**，是后续损失计算和梯度更新的前提。

2.  **核心数学定义**：文档通过公式(2)精确定义了如何计算。
    *   **输入**：一个前缀序列 $x$，一个语言模型（参数为 $θ$），以及一个目标词（标签）$y$。
    *   **计算过程**：模型根据前缀 $x$ 输出一个关于整个词表 $V$ 的 logits 向量 $f_θ(x)$。对该向量进行 softmax 操作，得到下一个词的概率分布。最后，取出这个概率分布中对应目标词 $y$ 的那个概率值，并计算其对数。
    *   **公式表达**：$log p_θ(y|x) = log[ softmax( f_θ(x) ) ]_y$

3.  **关键实现建议**：
    *   **数值稳定性**：强调在代码实现中必须使用**数值稳定**的方法来计算 log-softmax，直接使用上述公式的朴素实现可能导致数值溢出或精度问题。
    *   **实用工具**：推荐利用 `torch.nn.functional` 中的现有函数来实现，这些函数通常已内置数值稳定优化。
    *   **扩展功能**：建议在实现该功能时，可以设计一个可选参数，使其能够同时计算并返回每个位置的**标记熵**。这与您之前在处理的计算令牌熵的任务直接相关，熵值来源于同一个概率分布，可作为监控模型校准程度的重要指标。

### Problem (get_response_log_probs): Response log-probs (and entropy) (2 points)
- 完成

### SFT microbatch train step

1.  **核心目标**：在SFT中，通过优化（最小化）**负对数似然损失**来训练模型，使其生成的结果尽可能接近给定的目标输出。
2.  **损失计算关键**：
    *   **计算对象**：计算的是**目标输出标记**在给定输入提示下的对数概率。
    *   **求和范围**：损失是目标输出序列中**所有标记**的对数概率之和。
    *   **屏蔽操作**：在计算损失时，必须**忽略**（屏蔽）两类标记：
        *   **提示（Prompt）部分的标记**：因为损失只应针对模型需要生成的部分。
        *   **填充（Padding）标记**：用于将批次内序列长度对齐的无意义标记。
3.  **代码实现**：将专门实现一个**辅助函数**来完成上述损失计算逻辑。这确保了计算的一致性和代码的复用性。
4.  **后续应用**：这个辅助函数不仅用于SFT阶段，在之后的**强化学习（RL）** 阶段也会被使用。这表明了SFT和RL在优化目标上可能存在关联（例如，RL阶段可能同样需要基于模型输出的概率来构建目标或奖励）。

### Problem (masked_normalize): Masked normalize (1 point)

Deliverable: Implement a method masked_normalize that sums over tensor elements and
normalizes by a constant while respecting a boolean mask.
- 完成

### SFT microbatch train step

而不能部分解释了在进行**监督微调**时的一个关键训练步骤——**“微批次训练步骤”**。

1.  **切分数据**：将一个大批次（`train minibatch`，比如1000条数据）切分成多个小包，每个小包就是一个“微批次”（`microbatch`，比如50条数据）。
2.  **循环处理**：**顺序**处理每一个微批次：
    - 做一次前向传播（计算损失）。
    - 做一次反向传播（计算梯度）。
    - **但先不执行`optimizer.step()`更新权重！** 只是把当前微批次计算出的梯度**累加**到总梯度上。
3.  **统一更新**：当处理完指定数量（`gradient_accumulation_steps`，比如20个）的微批次后，我们才调用`optimizer.step()`，用累积的总梯度来更新一次模型参数，然后清空（`optimizer.zero_grad()`）梯度，准备下一轮累积。

```python
# 假设 gradient_accumulation_steps = 4
# 一个 minibatch 被分成了 4 个 microbatch

optimizer.zero_grad() # 在累积开始前清空梯度
total_loss = 0

for step in range(gradient_accumulation_steps):
    # 1. 获取一个微批次的数据
    microbatch_data = get_microbatch(data, step)
    
    # 2. 执行“微批次训练步骤” -> 这就是图片要求您实现的函数！
    loss = train_sft_microbatch(model, microbatch_data, optimizer)
    
    # 3. 损失和梯度会自动累加
    total_loss += loss.item()
    
    # 注意：这里还没有 optimizer.step()!

# 4. 所有微批次处理完后，用累积的梯度更新一次模型
optimizer.step()
# 5. 清空梯度，为下一个累积周期做准备
optimizer.zero_grad()

print(f"更新一次参数，平均损失: {total_loss / gradient_accumulation_steps}")
```

### Problem (sft_microbatch_train_step): Microbatch train step (3 points)

Deliverable: Implement a single micro-batch update for SFT, including cross-entropy loss, summing
with a mask, and gradient scaling.
- 完成

### Logging generations in-the-loop

本节强调了在进行模型（此处特指SFT或RL模型）训练时，**实施“循环内生成日志记录”** 这一良好实践的必要性及其核心记录项。

1.  **输入提示**：用于生成的具体问题或指令。这是评估的基准。
2.  **模型生成的响应**：模型针对输入提示所产生的实际输出。这是评估的直接对象。
3.  **真实答案/标准答案**：用于评估生成响应正确性的参考答案。
4.  **奖励信息**：一个结构化的评估结果，通常包括：
    *   **格式奖励**：响应是否符合指定格式（如是否包含 `\boxed{}`）。
    *   **答案奖励**：最终答案是否与标准答案匹配。
    *   **总奖励**：格式与答案奖励的综合（如总和），是RL训练中的关键信号。
5.  **响应的平均标记熵**：衡量模型在生成响应中每个标记（token）时的不确定性/置信度。熵值降低可能意味着模型变得“过度自信”。
6.  **响应长度分析**：
    *   **平均响应长度**：所有生成响应的平均长度。
    *   **正确响应的平均长度**：答案正确的那些响应的平均长度。
    *   **错误响应的平均长度**：答案错误的那些响应的平均长度。这有助于发现模型是否在“胡诌”时会产生更长的文本。

### Problem (log_generations): Logging generations (1 point)

Deliverable: Implement a function log_generations that can be used to log generations from your
model.
- 完成
- temperature=1.0和 top_p=1.0配置，pytorch运行和vllm运行 结果会出现不一致现象
    - vllm: We need to calculate the number of ducks that are sold every day and multiply it by the price per egg.
First, we subtract the number of eggs she eats for breakfast and the number she uses to bake muffins from the total number of eggs laid per day. Then, we multiply the number of eggs sold by the price per egg to get the total earnings. </think>
    - pytorch: **结果会随机变化**

## 4.3 SFT Experiment

### 内容概括

1. **实验目标**：在MATH数据集上对Qwen 2.5 Math 1.5B基础模型进行监督微调
2. **数据格式**：使用JSONL格式的数据，包含格式化的提示和带有思维链推理的目标响应
3. **硬件配置**：使用2个GPU，一个用于策略模型训练，另一个用于vLLM实例进行周期性评估
4. **技术实现**：提供了初始化vLLM、加载策略权重到vLLM实例的完整代码
5. **实验监控**：建议使用wandb记录训练和验证指标，便于后续分析

### 要点总结

#### 1. 实验设置
- **任务**：监督微调（SFT）Qwen 2.5 Math 1.5B模型
- **数据集**：MATH数据集，格式化为包含prompt和response的JSONL文件
- **硬件要求**：2个GPU（一个用于训练，一个用于评估）
- **评估机制**：定期在验证集上评估模型性能

#### 2. 关键技术组件
- **vLLM集成**：使用vLLM进行高效推理
- **模型加载**：支持从HuggingFace加载预训练模型
- **设备管理**：将策略模型和评估模型放在不同的GPU上
- **随机性控制**：设置随机种子确保可重复性

#### 3. 监控与日志
- **wandb集成**：记录训练和评估指标
- **指标分离**：区分训练步数和评估步数
- **可视化**：便于追踪模型训练进展

### 完整可执行代码

以下是整合所有图片中代码的完整实现：

```python
# 第一部分：导入必要的库
import torch
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel
from unittest.mock import patch
import wandb

# 第二部分：初始化vLLM实例
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    """
    启动推理过程，使用vLLM在与策略不同的GPU上加载模型。
    
    Args:
        model_id: 模型ID或路径
        device: 设备，如"cuda:0"或"cuda:1"
        seed: 随机种子
        gpu_memory_utilization: GPU内存利用率，默认为0.85
        
    Returns:
        vLLM实例
    """
    vllm_set_random_seed(seed)
    
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    #   22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.float16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

# 第三部分：将策略模型加载到vLLM实例
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """
    将策略模型加载到vLLM实例中。
    
    Args:
        policy: 预训练模型（策略）
        llm: 已初始化的vLLM实例
        
    Note:
        Copied from https://github.com/huggingface/trl/blob/
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# 第四部分：初始化WandB指标
def setup_wandb_metrics():
    """
    设置wandb指标，区分训练和验证步骤。
    
    这对于后续的强化学习实验也很有用。
    """
    # 设置wandb指标
    wandb.define_metric("train_step")  # 训练步骤的x轴
    wandb.define_metric("eval_step")    # 评估步骤的x轴
    
    # 以"train/"开头的所有指标都与train_step关联
    wandb.define_metric("train/*", step_metric="train_step")
    
    # 以"eval/"开头的所有指标都与eval_step关联
    wandb.define_metric("eval/*", step_metric="eval_step")

# 第五部分：完整的SFT训练示例
def run_sft_experiment():
    """
    完整的SFT实验流程
    """
    # 1. 设置设备和随机种子
    seed = 42
    torch.manual_seed(seed)
    
    # 2. 初始化wandb
    wandb.init(project="math-sft-experiment")
    setup_wandb_metrics()
    
    # 3. 初始化vLLM实例用于评估（在GPU 1上）
    print("正在初始化vLLM评估实例...")
    eval_llm = init_vllm(
        model_id="Qwen/Qwen2.5-Math-1.5B",  # 或本地路径
        device="cuda:1",  # 使用第二个GPU
        seed=seed,
        gpu_memory_utilization=0.85
    )
    
    # 4. 加载策略模型（在GPU 0上）
    print("正在加载策略模型...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        torch_dtype=torch.float16,
        device_map="cuda:0"  # 使用第一个GPU
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    
    # 5. 将策略模型权重加载到vLLM实例
    print("正在同步策略模型到vLLM实例...")
    load_policy_into_vllm_instance(policy_model, eval_llm)
    
    # 6. 加载训练数据
    print("正在加载MATH数据集...")
    import json
    from pathlib import Path
    
    data_path = Path("/data/a5-alignment/MATH/sft.jsonl")
    training_data = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            training_data.append({
                'prompt': example['prompt'],
                'response': example['response']
            })
    
    print(f"已加载 {len(training_data)} 个训练样本")
    
    # 7. 训练循环示例
    print("开始训练...")
    train_step = 0
    eval_step = 0
    
    for epoch in range(10):  # 假设训练10个epoch
        for batch in training_data:  # 简化示例，实际需要批处理
            # 训练步骤
            # ... 这里添加实际的训练代码 ...
            
            # 记录训练指标
            wandb.log({
                "train_step": train_step,
                "train/loss": 0.1,  # 示例损失
                "train/accuracy": 0.8,  # 示例准确率
            })
            train_step += 1
            
            # 定期评估
            if train_step % 100 == 0:
                # 在验证集上评估
                eval_results = evaluate_model(eval_llm, tokenizer)
                
                # 记录评估指标
                wandb.log({
                    "eval_step": eval_step,
                    "eval/accuracy": eval_results['accuracy'],
                    "eval/loss": eval_results['loss'],
                })
                eval_step += 1
    
    print("训练完成!")
    
    # 8. 清理
    wandb.finish()
    return policy_model, eval_llm

# 第六部分：辅助函数 - 模型评估
def evaluate_model(llm: LLM, tokenizer, val_data=None):
    """
    在验证集上评估模型
    
    Args:
        llm: vLLM实例
        tokenizer: 分词器
        val_data: 验证数据，如果为None则使用示例数据
        
    Returns:
        评估结果字典
    """
    if val_data is None:
        # 示例验证数据
        val_data = [
            {"prompt": "问题: 2+2=?", "response": "答案是4"},
            {"prompt": "问题: 3*4=?", "response": "答案是12"},
        ]
    
    correct = 0
    total = len(val_data)
    
    for example in val_data:
        prompt = example['prompt']
        expected_response = example['response']
        
        # 使用vLLM生成
        outputs = llm.generate(prompt, sampling_params={"max_tokens": 100})
        generated_response = outputs[0].outputs[0].text
        
        # 简化的评估逻辑
        if generated_response.strip() == expected_response.strip():
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'loss': 1.0 - accuracy  # 示例损失
    }

# 第七部分：主程序入口
if __name__ == "__main__":
    # 运行SFT实验
    policy_model, eval_llm = run_sft_experiment()
    
    # 保存模型
    policy_model.save_pretrained("./trained_qwen_math_1.5b")
    
    print("模型已保存到 ./trained_qwen_math_1.5b")
```

### 关键配置说明

#### 1. 数据格式
```json
{
  "prompt": "数学问题描述...",
  "response": "思考过程... 最终答案: \boxed{答案}"
}
```

#### 2. 硬件配置
- GPU 0: 策略模型训练
- GPU 1: vLLM评估实例
- 内存利用率: 85% (可调整)

#### 3. 训练监控
- 训练指标: 以 `train/` 开头，关联 `train_step`
- 评估指标: 以 `eval/` 开头，关联 `eval_step`
- 评估频率: 每100个训练步评估一次

### 使用说明

1. **安装依赖**:
```bash
pip install vllm transformers wandb torch
```

2. **运行实验**:
```bash
python sft_experiment.py
```

3. **监控训练**:
- 访问 wandb 面板查看训练曲线
- 观察训练损失和验证准确率
- 调整超参数以获得最佳性能

```bash
# https://github.com/Louisym/Stanford-CS336-spring25/blob/main/assignment5-alignment/scripts/sft_experiment.py#L232
```

### Problem (sft_experiment): Run SFT on the MATH dataset (2 points) (2 H100 hrs)

1. Run SFT on the reasoning SFT examples (provided in /data/a5-alignment/MATH/sft.jsonl)
using the Qwen 2.5 Math 1.5B base model, varying the number of unique examples for SFT in the range {128, 256, 512, 1024}, 
along with using the full dataset. Tune the learning rate and
batch size to achieve at least 15% validation accuracy when using the full dataset.
- 采用gsm8k体验流程
    - Qwen2.5-Math-1.5B_bs1_asteps64_niters125_nsamples128_gpu2  (Eval: iter: 120 correct_num: 1070 error_num: 249)
      - {'{"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0}': 1030, '{"answer_reward": 0.0, "format_reward": 0.0, "reward": 0.0}': 114, '{"answer_reward": 0.0, "format_reward": 1.0, "reward": 0.0}': 175}
    - Qwen2.5-Math-1.5B_bs1_asteps64_niters125_nsamples256_gpu1 
      - iter96 {'{"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0}': 902, '{"answer_reward": 0.0, "format_reward": 0.0, "reward": 0.0}': 352, '{"answer_reward": 0.0, "format_reward": 1.0, "reward": 0.0}': 65}
    - Qwen2.5-Math-1.5B_bs1_asteps64_niters125_nsamples512_gpu1 
      - iter96 {'{"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0}': 1011, '{"answer_reward": 0.0, "format_reward": 1.0, "reward": 0.0}': 206, '{"answer_reward": 0.0, "format_reward": 0.0, "reward": 0.0}': 102}
    - Qwen2.5-Math-1.5B_bs1_asteps64_niters125_nsamples1024_gpu1 
      - iter48 {'{"answer_reward": 0.0, "format_reward": 0.0, "reward": 0.0}': 1185, '{"answer_reward": 0.0, "format_reward": 1.0, "reward": 0.0}': 25, '{"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0}': 109}
      - iter12 {'{"answer_reward": 0.0, "format_reward": 0.0, "reward": 0.0}': 1245, '{"answer_reward": 0.0, "format_reward": 1.0, "reward": 0.0}': 55, '{"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0}': 19}
    - Qwen2.5-Math-1.5B_bs1_asteps64_niters125_nsamples10240_gpu1 全部数据
      - iter108 {'{"answer_reward": 0.0, "format_reward": 0.0, "reward": 0.0}': 613, '{"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0}': 590, '{"answer_reward": 0.0, "format_reward": 1.0, "reward": 0.0}': 116}
      - iter96 {'{"answer_reward": 0.0, "format_reward": 0.0, "reward": 0.0}': 623, '{"answer_reward": 0.0, "format_reward": 1.0, "reward": 0.0}': 120, '{"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0}': 576}
      
      
2. Filter the reasoning SFT examples to only include examples that produce the correct answer. Run
SFT on the (full) filtered dataset and report the size of the filtered dataset and the validation
accuracy you achieve.

- 采用gsm8k体验流程，结果不正确的cot序号如下，在0-127之外，对上述结果无影响
```python
1082 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
1367 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
2738 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
3296 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
4976 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
5755 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
6969 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
6982 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
7031 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
7081 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
7082 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
7094 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
7098 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
7109 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
7113 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
7121 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
7124 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
7137 {'format_reward': 1.0, 'answer_reward': 0.0, 'reward': 0.0}
```
-  Qwen2.5-Math-1.5B_bs1_asteps64_niters125_nsamples10240_gpu1_2nd
   - iter120 {'{"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0}': 598, '{"answer_reward": 0.0, "format_reward": 0.0, "reward": 0.0}': 625, '{"answer_reward": 0.0, "format_reward": 1.0, "reward": 0.0}': 96}
   - iter108 {'{"answer_reward": 0.0, "format_reward": 0.0, "reward": 0.0}': 604, '{"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0}': 614, '{"answer_reward": 0.0, "format_reward": 1.0, "reward": 0.0}': 101}

- Qwen2.5-Math-1.5B_bs1_asteps64_niters125_nsamples10240_gpu1_3rd 延长训练时长
   - iter800 {'{"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0}': 762, '{"answer_reward": 0.0, "format_reward": 0.0, "reward": 0.0}': 522, '{"answer_reward": 0.0, "format_reward": 1.0, "reward": 0.0}': 35}
   - iter700 {'{"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0}': 750, '{"answer_reward": 0.0, "format_reward": 0.0, "reward": 0.0}': 533, '{"answer_reward": 0.0, "format_reward": 1.0, "reward": 0.0}': 36}