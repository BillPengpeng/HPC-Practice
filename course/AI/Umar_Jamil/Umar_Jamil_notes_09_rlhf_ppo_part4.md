本文主要整理rlhf_ppo的主要内容。

## 11.3 - batched_forward_pass源码分析

这是**批量前向传播的核心实现**，用于高效计算语言模型在RLHF训练中所需的各种输出。以下是详细分析：

### 函数概览

`batched_forward_pass`函数的核心功能是：
1. **批量分割处理**：将大批量数据分割为小批量，避免内存溢出
2. **多任务计算**：一次前向传播同时获取对数概率、价值预测等
3. **掩码生成**：精确识别哪些位置对应响应token（用于后续奖励计算）

### 参数详解

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `PreTrainedModelWrapper` | 包装过的模型，包含策略和价值头 |
| `queries` | `torch.Tensor` | 编码后的查询（提示），形状`(batch_size, query_length)` |
| `responses` | `torch.Tensor` | 编码后的响应，形状`(batch_size, response_length)` |
| `model_inputs` | `dict` | 模型输入字典（attention_mask等） |
| `return_logits` | `bool` | 是否返回logits（消耗内存） |
| `response_masks` | `Optional[torch.Tensor]` | 可选的响应内部掩码 |

### 核心设计：内存优化策略

#### 1. 小批量处理
```python
bs = len(queries)  # 原始批量大小
fbs = self.config.mini_batch_size  # 小批量大小

for i in range(math.ceil(bs / fbs)):
    # 分割批次
    input_kwargs = {key: value[i*fbs:(i+1)*fbs] for key, value in model_inputs.items()}
```
- **问题**：大语言模型前向传播消耗大量显存
- **解决方案**：将大批量拆分为可管理的小批量
- **典型设置**：`mini_batch_size=1-8`，取决于GPU内存

#### 2. 选择性返回logits
```python
if return_logits:
    all_logits.append(logits)
else:
    del logits  # 及时释放内存
```
- **logits形状**：`(batch, seq_len, vocab_size)`，非常大
- **内存优化**：仅在需要时保留logits（如计算熵时）

### 核心计算步骤

#### 1. 模型前向传播
```python
logits, _, values = model(**input_kwargs)
```
模型输出：
- `logits`：未归一化的词表分数，形状`(batch, seq_len, vocab_size)`
- `values`：价值头预测，形状`(batch, seq_len)`

#### 2. 对数概率计算
```python
logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
```
**关键偏移**：
- 输入：`logits[:, :-1, :]` → 除最后一个token外的所有logits
- 目标：`input_ids[:, 1:]` → 除第一个token外的所有token
- **原因**：预测下一个token，标准的语言建模目标

#### 3. 掩码生成（最复杂部分）

##### 基础掩码
```python
masks = torch.zeros_like(attention_mask)
masks[:, :-1] = attention_mask[:, 1:]  # 指示哪些位置有logprobs
```
- 初始：与attention_mask同形状的全0张量
- 填充：将`attention_mask[:, 1:]`的值赋给`masks[:, :-1]`
- **含义**：标记哪些位置是有效的下一个token预测

##### 响应位置识别
```python
# 编码器-解码器模型
if self.is_encoder_decoder:
    start = 1  # 解码从索引1开始
    end = attention_mask[j, :].sum() - 1  # 有效序列长度-1

# 仅解码器模型（如GPT）
else:
    start = len(query_batch[j]) - 1  # 查询结束后开始
    if attention_mask[j, 0] == 0:  # 左填充偏移
        start += attention_mask[j, :].nonzero()[0]
    end = start + len(response_batch[j])  # 响应结束位置
```

**可视化示例**：
```
查询: [BOS, "Hello", "world", "?"]
响应: ["I'm", "good", EOS]
完整序列: [BOS, "Hello", "world", "?", "I'm", "good", EOS, PAD, PAD]

注意力掩码: [1, 1, 1, 1, 1, 1, 1, 0, 0]
基础掩码:  [1, 1, 1, 1, 1, 1, 0, 0, 0]  # 移除了最后一个位置
响应掩码:  [0, 0, 0, 0, 1, 1, 0, 0, 0]  # 只标记响应部分
```

##### 掩码应用
```python
masks[j, :start] = 0  # 掩码查询部分
masks[j, end:] = 0    # 掩码填充部分
```
- 只保留响应部分的token
- 查询token和填充token的掩码设为0

### 返回结果处理

```python
return (
    torch.cat(all_logprobs),  # 所有小批次的对数概率
    torch.cat(all_logits)[:, :-1] if return_logits else None,  # 移除了最后一个token
    torch.cat(all_values)[:, :-1],  # 移除了最后一个token
    torch.cat(all_masks)[:, :-1],  # 移除了最后一个token
)
```

**维度对齐**：
- 所有返回张量的形状：`(batch_size, seq_len-1)`
- 移除了最后一个位置，因为`logprobs`只计算到`seq_len-1`

### 与PPO训练流程的整合

```python
def gather_training_data(self, batch):
    """收集PPO训练所需数据"""
    
    # 1. 获取当前策略的输出
    with torch.no_grad():
        current_logprobs, _, current_values, masks = self.batched_forward_pass(
            model=self.policy_model,
            queries=batch["queries"],
            responses=batch["responses"],
            model_inputs=batch["model_inputs"],
            return_logits=False
        )
    
    # 2. 获取参考模型的输出
    with torch.no_grad():
        ref_logprobs, _, _, _ = self.batched_forward_pass(
            model=self.ref_model,  # 冻结的参考模型
            queries=batch["queries"],
            responses=batch["responses"],
            model_inputs=batch["model_inputs"],
            return_logits=False
        )
    
    # 3. 获取奖励模型的评分
    with torch.no_grad():
        scores = self.reward_model(
            queries=batch["queries"],
            responses=batch["responses"]
        )
    
    # 4. 计算每个token的奖励
    rewards, non_score_rewards, kls = self.compute_rewards(
        scores=scores,
        logprobs=current_logprobs,
        ref_logprobs=ref_logprobs,
        masks=masks
    )
    
    return {
        "queries": batch["queries"],
        "responses": batch["responses"],
        "logprobs": current_logprobs,
        "values": current_values,
        "ref_logprobs": ref_logprobs,
        "rewards": rewards,
        "masks": masks
    }
```

## 11.4 - train_minibatch源码分析

这是**PPO小批量训练的核心实现**，展示了如何在一个小批量上执行一次完整的PPO参数更新。以下是详细分析：

### 函数概览

`train_minibatch`函数执行**单个PPO小批量的训练步骤**，包括：
1. 计算损失（策略损失 + 价值损失）
2. 反向传播
3. 梯度裁剪
4. 参数更新
5. 梯度清零

### 参数详解

| 参数 | 形状 | 说明 |
|------|------|------|
| `old_logprobs` | `(mini_batch_size, response_length)` | 旧策略下各动作的对数概率 |
| `values` | `(mini_batch_size, response_length)` | 旧策略下价值网络的预测 |
| `logprobs` | `(mini_batch_size, response_length)` | 新策略下各动作的对数概率 |
| `logits` | `(mini_batch_size, response_length, vocab_size)` | 新策略的logits（用于计算熵） |
| `vpreds` | `(mini_batch_size, response_length)` | 新策略下价值网络的预测 |
| `mask` | `(mini_batch_size, response_length)` | 掩码，标识有效token位置 |
| `advantages` | `(mini_batch_size, response_length)` | 优势函数估计 |
| `returns` | `(mini_batch_size, response_length)` | 回报（用于价值网络训练） |

### 训练步骤详解

#### 1. 设置训练模式
```python
self.model.train()
```
- 启用dropout、batch normalization等的训练模式
- 确保模型在训练状态下计算梯度和统计数据

#### 2. 计算损失
```python
loss_p, loss_v, train_stats = self.loss(
    old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
)
```
调用之前分析的`loss`函数，返回：
- `loss_p`：策略损失（Policy Loss）
- `loss_v`：价值损失（Value Loss）
- `train_stats`：训练统计信息字典

#### 3. 组合总损失
```python
loss = loss_p + loss_v
```
- 总损失 = 策略损失 + 价值损失
- **注意**：这里没有包含熵正则化项，可能在`loss`函数内部已处理

#### 4. 反向传播
```python
self.accelerator.backward(loss)
```
使用Hugging Face Accelerate进行**自动混合精度和分布式训练**：
```python
# Accelerate自动处理：
# 1. 混合精度（自动类型转换）
# 2. 梯度累积
# 3. 分布式训练梯度同步
# 4. 梯度缩放（防止下溢）
```

#### 5. 梯度裁剪
```python
if self.config.max_grad_norm is not None:
    if self.accelerator.sync_gradients:
        self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
```
**梯度裁剪的条件**：
1. 配置了最大梯度范数（通常1.0）
2. 当前是梯度累积的同步步骤

**梯度累积机制**：
```python
# 假设gradient_accumulation_steps=4
for step in range(gradient_accumulation_steps):
    # 前向传播、计算损失
    loss.backward()  # 梯度累积
    
    if (step + 1) % gradient_accumulation_steps == 0:
        # 同步步骤：裁剪并更新
        accelerator.clip_grad_norm_(...)
        optimizer.step()
        optimizer.zero_grad()
```

#### 6. 参数更新
```python
self.optimizer.step()
```
- 执行优化器更新（如Adam）
- 基于累积的梯度更新模型参数

#### 7. 梯度清零
```python
self.optimizer.zero_grad()
```
- 清空优化器中的梯度
- **注意**：Accelerate处理梯度累积，这里每次迭代都清零

## 11.5 - step源码分析

这是**完整的PPO训练步骤实现**，整合了之前分析的所有组件，是RLHF训练的核心入口函数。以下是详细分析：

### 函数概览

`step`函数是PPO训练的**顶层调度器**，它：
1. 接收一批查询、响应和奖励分数
2. 计算所有必要中间量（对数概率、优势函数等）
3. 执行多轮PPO优化
4. 收集训练统计信息
5. 更新相关参数（如KL系数）


### PPO训练阶段详解

#### 阶段1：数据准备与计算

##### 1. 输入验证
```python
queries, responses, scores, response_masks = self._step_safety_checker(
    bs, queries, responses, scores, response_masks
)
```
- 检查输入形状、类型一致性
- 确保批量大小匹配配置

##### 2. 奖励标准化（可选）
```python
# 奖励标准化和裁剪
if self.config.use_score_scaling:
    scores_mean, scores_std = self.running.update(scores)
    score_scaling_factor = self.running.std.to(...) + torch.finfo(scores.dtype).eps
    scores = (scores - self.running.mean.to(...)) / score_scaling_factor

if self.config.score_clip is not None:
    scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip)
```
- **奖励标准化**：使奖励均值为0，标准差为1
- **奖励裁剪**：防止异常奖励值

##### 3. 模型输入准备
```python
model_inputs = self.prepare_model_inputs(queries, responses)
```
- 拼接查询和响应
- 生成注意力掩码
- 处理填充以保证批量处理

##### 4. 前向传播计算
```python
with torch.no_grad():
    # 当前策略
    all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
        self.model, queries, responses, model_inputs, response_masks=response_masks
    )
    
    # 参考模型
    ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
        self.model if self.is_peft_model else self.ref_model,
        queries, responses, model_inputs
    )
```
- **无梯度计算**：仅用于数据收集
- **双重前向传播**：分别计算当前策略和参考模型

##### 5. 奖励计算
```python
rewards, non_score_reward, kls = self.compute_rewards(
    scores, all_logprobs, ref_logprobs, masks
)
```
- 结合奖励模型分数和KL惩罚
- 生成每个token的奖励信号

##### 6. 优势函数计算
```python
values, advantages, returns = self.compute_advantages(values, rewards, masks)
```
- 使用GAE计算优势函数
- 计算回报（用于价值网络训练）

#### 阶段2：PPO优化

##### 1. 数据存储
```python
batch_dict = {
    "queries": queries,
    "responses": responses,
    "logprobs": all_logprobs.to(torch.float32),
    "values": values.to(torch.float32),
    "masks": masks,
    "advantages": advantages,
    "returns": returns,
}
batch_dict.update(model_inputs)
```
- 存储所有训练数据
- 转换为float32以确保精度

##### 2. 多轮PPO优化
```python
for _ in range(self.config.ppo_epochs):
    if early_stop:
        break
    b_inds = np.random.permutation(bs)  # 打乱数据顺序
    for backward_batch_start in range(0, bs, self.config.backward_batch_size):
        backward_batch_end = backward_batch_start + self.config.backward_batch_size
        backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]
        
        for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
            mini_batch_end = mini_batch_start + self.config.mini_batch_size
            mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
            
            # 构建小批量
            mini_batch_dict = {...}
            
            # 训练步骤
            with self.accelerator.accumulate(self.model):
                # 计算新策略输出
                logprobs, logits, vpreds, _ = self.batched_forward_pass(
                    self.model,
                    mini_batch_dict["queries"],
                    mini_batch_dict["responses"],
                    model_inputs,
                    return_logits=True,
                )
                
                # 训练小批量
                train_stats = self.train_minibatch(...)
```

**分层批处理策略**：
1. **PPO Epochs**：对同一批数据重复训练多次（通常2-4次）
2. **Backward Batch**：梯度累积的大批次
3. **Mini Batch**：实际训练的小批次

##### 3. 提前停止
```python
if self.config.early_stopping:
    policykl = train_stats["policy/policykl"]
    early_stop = self._early_stop(policykl)
    if early_stop:
        break
```
- 当KL散度过大时提前停止
- 防止模型偏离参考模型太远

#### 阶段3：统计与更新

##### 1. 统计信息收集
```python
train_stats = stack_dicts(all_stats)
stats = self.record_step_stats(
    scores=scores,
    logprobs=all_logprobs,
    ref_logprobs=ref_logprobs,
    non_score_reward=non_score_reward,
    train_stats=train_stats,
    kl_coef=self.kl_ctl.value,
    masks=masks,
    queries=queries,
    responses=responses,
    kls=kls,
)
```

##### 2. KL控制系数更新
```python
self.kl_ctl.update(
    stats["objective/kl"],
    self.config.batch_size * self.accelerator.num_processes,
)
```
- 自适应调整KL惩罚强度
- 保持KL散度在目标范围内

##### 3. 学习率更新
```python
if self.lr_scheduler is not None:
    self.lr_scheduler.step()
```

### 关键配置参数

| 参数 | 说明 | 典型值 |
|------|------|------|
| `batch_size` | 每步处理的样本数 | 32-256 |
| `mini_batch_size` | 每次优化的小批量大小 | 1-8 |
| `backward_batch_size` | 梯度累积的批次大小 | 16-64 |
| `ppo_epochs` | 对同批数据的优化轮数 | 2-4 |
| `kl_penalty` | KL惩罚类型 | "kl"/"full" |
| `early_stopping` | 是否提前停止 | True/False |
| `score_clip` | 奖励裁剪阈值 | 5.0 |
| `use_score_scaling` | 是否标准化奖励 | True |


### 训练监控与日志

#### 1. 训练统计
```python
# 记录的关键指标
stats_keys = [
    "objective/kl",            # KL散度
    "objective/kl_dist",       # KL分布
    "objective/score",         # 奖励分数
    "ppo/policy/advantages_mean",  # 优势均值
    "ppo/policy/ratio",        # 重要性采样比率
    "ppo/policy/clipfrac",     # 裁剪比例
    "ppo/returns/mean",        # 回报均值
    "ppo/val/error",           # 价值误差
]
```

#### 2. 时序统计
```python
timing = {
    "time/ppo/forward_pass": 0.1,
    "time/ppo/compute_rewards": 0.05,
    "time/ppo/compute_advantages": 0.02,
    "time/ppo/optimize_step": 0.5,
    "time/ppo/total": 0.67,
}
```

### 完整训练流程示例

```python
# 简化版训练循环
for iteration in range(total_iterations):
    
    # 1. 收集响应
    queries, responses, rewards = collect_responses(policy_model, prompt_dataset)
    
    # 2. 计算奖励
    with torch.no_grad():
        scores = reward_model(queries, responses)
    
    # 3. PPO优化步骤
    stats = ppo_trainer.step(queries, responses, scores)
    
    # 4. 日志记录
    if iteration % log_interval == 0:
        log_stats(stats)
    
    # 5. 模型检查点
    if iteration % checkpoint_interval == 0:
        save_checkpoint(policy_model, iteration)
```

## 11.6 - PPO的第二阶段（优化阶段）不重新计算rewards和advantages

### PPO的设计哲学

#### 1. **在线策略 vs 伪离线策略**
```python
# 传统策略梯度（如REINFORCE）是严格的在线策略：
# 每次更新必须重新采样
for update in range(num_updates):
    # 必须重新采样！
    trajectories = collect_new_trajectories(current_policy)
    compute_advantages(trajectories)
    update_policy()

# PPO是"伪离线策略"：
# 用旧数据多次更新
trajectories = collect_trajectories(old_policy)  # 只采样一次
advantages = compute_advantages(trajectories)    # 只计算一次

for epoch in range(ppo_epochs):  # 多次复用
    for mini_batch in split_data(trajectories):
        update_policy(mini_batch)  # 使用固定的advantages
```

#### 2. **重要性采样的使用**
PPO通过**重要性采样**来复用旧数据：
```python
# 重要性采样比率
ratio = π_new(a|s) / π_old(a|s)

# 在更新中使用这个比率来校正分布差异
loss = E_old[ratio * A_old]
```
这意味着我们可以用旧策略(`π_old`)的数据来评估新策略(`π_new`)，**只要新旧策略差异不大**。

### 不重新计算的具体原因

#### 1. **计算效率**
- 每次重新计算`advantages`需要：
  1. 重新运行模型前向传播
  2. 重新计算奖励（调用奖励模型）
  3. 重新运行GAE计算
- 这会使训练速度**减慢3-5倍**

#### 2. **奖励模型的成本**
```python
# 奖励模型通常很大（与策略模型相当甚至更大）
reward_model = load_model("reward-model-7b")  # 70亿参数

# 每次调用都很昂贵
with torch.no_grad():
    scores = reward_model(queries, responses)  # 高延迟、高显存
```
重新计算奖励意味着频繁调用这个昂贵模型。

#### 3. **优势估计的稳定性**
优势估计`A(s,a)`衡量的是**动作相对于平均水平的优劣**：
- 如果策略变化不大，这个相对优劣关系**基本保持不变**
- 重新计算会在训练中引入额外噪声

### PPO的"信任区域"保障

PPO通过多种机制确保策略更新是"安全的"，从而能够复用数据：

#### 1. **裁剪机制**
```python
# 重要性采样比率被裁剪
clipped_ratio = torch.clamp(ratio, 1-ε, 1+ε)  # ε通常=0.1~0.2

# 使用裁剪后的比率
loss = min(ratio * A, clipped_ratio * A)
```
这确保了新旧策略不会差异太大。

#### 2. **KL惩罚**
```python
# 在奖励中添加KL散度惩罚
reward = rm_score - β * KL(π_new || π_ref)
```
这防止了策略过度偏离参考模型。

#### 3. **早期停止**
```python
# 监控KL散度
if kl_divergence > threshold:
    break  # 提前停止本轮优化
```
