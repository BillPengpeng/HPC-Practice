本文主要整理Assignment 5 (alignment)的主要内容。

## 8 GRPO Experiments

### Problem (grpo_learning_rate): Tune the learning rate (2 points) (6 H100 hrs)
Starting with the suggested hyperparameters above, perform a sweep over the learning rates and
report the final validation answer rewards (or note divergence if the optimizer diverges).

- 采用gsm8k体验流程
```python
learning_rate: float = 1e-5 Eval n_grpo_idx: 149 correct_num: 1069 error_num: 250
learning_rate: float = 1e-4 Eval n_grpo_idx: 59 correct_num: 1060 error_num: 259  loss跑飞
learning_rate: float = 5e-5 Eval n_grpo_idx: 59 correct_num: 1097 error_num: 222  loss跑飞
learning_rate: float = 3e-5 Eval n_grpo_idx: 179 correct_num: 1138 error_num: 181
```

### Problem (grpo_baselines): Effect of baselining (2 points) (2 H100 hrs)
Train a policy with reinforce_with_baseline and with no_baseline.

- 采用gsm8k体验流程
```python
learning_rate: float = 3e-5 + no_baseline Eval n_grpo_idx: 139 correct_num: 1062 error_num: 257
```

### Length normalization

#### 1. **核心问题**
- 在GRPO训练中，如何**聚合每个token的损失**是一个关键设计选择
- 不同的聚合方法会导致**不同的梯度，从而影响策略动作的信用分配**

#### 2. **两种归一化方法对比**

| 方法 | 计算方式 | 特点 | 对梯度的影响 |
|------|---------|------|------------|
| **masked_mean** | 对未掩码的token取平均 | 每个序列除以实际有效token数 | 梯度大小与序列长度成反比 |
| **masked_normalize** | 对未掩码的token求和，再除以常数 | 使用固定归一化因子（如最大生成长度） | 梯度大小固定，与序列长度无关 |

#### 3. **理论分析**
- 损失计算：`per_token_loss = -advantages * per_token_probability_ratios`
- advantages形状：`(batch_size, 1)`，在序列长度维度广播
- 示例：批次大小2，第一个响应4个token，第二个响应7个token
- 归一化选择会影响梯度的相对大小，从而改变优化方向

#### 4. **梯度计算分析**
- **masked_mean**：梯度 = 1 / 实际有效token数
  - 第一个序列：每个token梯度 = 1/4 = 0.25
  - 第二个序列：每个token梯度 = 1/7 ≈ 0.1429
- **masked_normalize**：梯度 = 1 / 常数归一化因子
  - 所有token梯度 = 1/7 ≈ 0.1429

### Problem (think_about_length_normalization): Think about length normalization (1 point)

Deliverable: Compare the two approaches (without running experiments yet). What are the pros
and cons of each approach? Are there any specific settings or examples where one approach seems
better?

#### masked_mean 方法
**优点**：
- **公平性**：每个序列的损失被其实际长度归一化，避免了长序列因其token数量多而主导梯度更新的问题。
- **适应性**：自动适应不同长度的序列，使模型平等看待长短序列的每个token。
- **稳定性**：在序列长度差异大的数据集中，有助于稳定训练，防止长序列产生过大的梯度。

**缺点**：
- **梯度稀释**：对于短序列，每个token的梯度较大（因为除以较小的数），可能导致更新不稳定；对于长序列，每个token的梯度较小，更新缓慢。
- **信用分配扭曲**：在强化学习中，如果整个序列的回报是稀疏的（只有最终奖励），那么平均分配可能不合理，因为并非每个token都对最终奖励有同等贡献。

#### masked_normalize 方法
**优点**：
- **一致性**：所有序列使用相同的归一化因子，梯度尺度一致，便于调参和优化。
- **控制影响**：可以通过调整常数归一化因子（如设置最大生成长度）来控制长序列的总影响，避免其损失过大。
- **简单可控**：计算简单，无需动态计算每个序列的实际长度。

**缺点**：
- **长度偏差**：如果常数归一化因子设置不当，可能使短序列的损失被过度放大（若因子远大于实际长度），或使长序列的损失被过度压缩（若因子远小于实际长度）。
- **灵活性差**：无法自适应不同长度的序列，需要根据数据集特点谨慎选择归一化因子。

### Problem (grpo_length_normalization): Effect of length normalization (2 points) (2 H100 hrs)

Deliverable: Compare normalization with masked_mean and masked_normalize with an end-to-
end GRPO training run. Report the validation answer reward curves. Comment on the findings,
including any other metrics that have a noticeable trend.

- 采用gsm8k体验流程
```python
长度取response_mask.shape[1]
learning_rate: float = 3e-5 + baseline_masked_norm  Eval n_grpo_idx: 139 correct_num: 1098 error_num: 221
```

### Problem (grpo_group_standard_deviation): Effect of standard deviation normalization (2 points) (2 H100 hrs)

Deliverable: Compare the performance of use_std_normalization == True and use_std_ ⌋
normalization == False. Report the validation answer reward curves. Comment on the findings,
including any other metrics that have a noticeable trend.

- 采用gsm8k体验流程
```python
learning_rate: float = 3e-5 + reinforce_with_baseline_masked_mean_normFalse Eval n_grpo_idx: 179 correct_num: 1148 error_num: 171
```