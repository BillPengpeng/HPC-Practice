本文主要整理Assignment 3 (scaling): Scaling Laws的主要内容。

## 1. Assignment Overview

### 内容概况

本部分的核心主题是让学习者通过实践来理解和应用**语言模型的缩放定律**。学生需要扮演一个模型训练负责人的角色，在固定的巨大计算预算下，通过权衡模型大小和训练数据量，找到实现最低训练损失的最佳方案。

### 要点总结

1.  **核心任务**：
    *   在固定的计算预算（1e19 FLOPs）内，训练出性能最优（即训练损失最低）的语言模型。
    *   核心挑战是进行**模型大小与训练数据量（token数量）之间的权衡**。

2.  **实践方法**：
    *   **使用缩放定律**：学生需要构建缩放定律，来预估在给定计算预算下的最优模型大小及超参数。
    *   **调用“训练API”**：无需亲自训练模型，而是通过向一个模拟的API接口提交模型超参数和期望的计算量，来获取对应的最终训练损失值。

3.  **资源与限制**：
    *   用于拟合缩放定律的计算预算为 **2e18 FLOPs**（占总预算的20%）。这意味着需要高效地利用有限的API查询次数来探索超参数空间。

4.  **作业材料**：
    *   所有相关代码和文档都存放在GitHub仓库中：`github.com/stanford-cs336/assignment3-scaling`。学生需要克隆该仓库以开始作业。

## 2. Scaling Laws Review

### 内容概况

本部分是课程中关于**缩放定律** 的回顾环节。它引用了Hoffmann等人（2022）的《Chinchilla》论文，旨在向学生介绍一个核心问题：在固定的计算预算下，如何选择最佳的超参数组合（如模型大小、训练数据量）来实现最低的训练损失。幻灯片强调了从“小规模实验”可靠地“外推”到“大规模”训练是其中的主要挑战，并提示学生可以参考其他相关研究来完善自己的方法。

---

### 要点总结

1.  **核心问题**：
    *   核心问题是：**给定一个固定的计算预算C**，用于训练大型语言模型，如何选择**模型参数量、训练token数量**等超参数，才能实现**最低的训练损失**。

2.  **主要挑战**：
    *   核心挑战在于**如何从小规模的实验数据中，可靠地推断出在大规模训练时的模型性能**。这是一个关键的预测和规划问题。

3.  **方法来源**：
    *   本次回顾主要基于具有影响力的 **《Chinchilla》论文（Hoffmann 等人, 2022）** 中提出的方法。该论文的核心结论是，在相同计算预算下，训练更多“小模型”但使用更多数据，通常比训练一个“大模型”但数据不足更有效。

4.  **扩展阅读参考**：
    *   **Kaplan 等人 (2020)**：OpenAI 早期关于缩放定律的开创性研究。
    *   **Yang 等人 (2022)**：可能提供了其他视角或更精细化的缩放定律建模方法。

## 2.1 Scaling Laws from IsoFLOPs profiles

### 内容概况

本部分详细解释了Hoffmann等人在2022年提出的核心方法：如何通过**IsoFLOPs** 分析来推导Transformer语言模型的**缩放定律**。该方法的核心是在**固定计算预算C** 的前提下，通过系统性地改变模型大小N和数据量D，找出实现最低训练损失的最佳配置，并据此拟合出可预测更大规模模型行为的幂律关系。

---

### 要点总结

1.  **核心概念：IsoFLOPs方法**
    *   在**相同的计算预算C** 下（C ≈ 6ND），训练一系列**不同大小（N）** 的模型。模型越大，其能用于训练的数据量（D）就越少（因为 D = C / (6N)）。
    *   目标是观察在固定计算量下，模型大小N如何影响最终的训练损失L。

2.  **关键发现：损失与模型大小的二次关系**
    *   经验表明，在固定计算量C下，最终训练损失L与模型大小N之间存在**类似抛物线的二次关系**。
    *   **直观解释**：
        *   **模型太小（欠拟合）**：参数不足，无法有效学习数据，损失高。
        *   **模型太大（训练不足）**：在有限计算量C内，模型太大导致训练步数过少，无法充分训练，损失也高。
    *   因此，对于任意给定的计算预算C，都存在一个能**最小化训练损失的“最优点”模型大小N_opt**。

3.  **最终目标：拟合缩放定律以进行预测**
    *   该方法的核心步骤是，为多个不同的计算预算C_i分别找到其对应的最优模型大小N_opt(C_i)和最优数据量D_opt(C_i)。
    *   然后，利用这些（C_i, N_opt）数据点**拟合一条幂律函数**（形式如 N_opt ∝ C^a），用以预测在**未来更大的计算预算**下，计算最优的模型和数据集规模应该是多少。

### Problem (chinchilla_isoflops): 5 points

Write a script to reproduce the IsoFLOPs method describe above for fitting scaling laws using
the final training loss from a set of training runs. For this problem, use the (synthetic) data from
training runs given in the file data/isoflops_curves.json. This file contains a JSON array, where
each element is an object describing a training run.

For fitting the scaling laws, the scipy package (and scipy.optimize.curve_fit in particular)
might be useful, but you’re welcome to use any curve fitting method you’d like. While Hoffmann et al.
[2022] fits a quadratic function to each IsoFLOP profile to find its minimum, we instead recommend
you simply take the run with the lowest training loss for each compute budget as the minimum.

1. Show your extrapolated compute-optimal model size, together with the ⟨Ci , Nopt (Ci )⟩ points you
obtained. What is your predicted optimal model size for a budget of 1023 FLOPs? What about
for 1024 FLOPs?

2. Show your extrapolated compute-optimal dataset size, together with the ⟨Ci , Dopt (Ci )⟩ data
points from the training runs. What is your predicted optimal dataset size for budgets of 1023
and 1024 FLOPs?

```python
拟合结果: N_opt = 2.58e+01 * C^0.404
flops:100000000000000000000000 => N_opt:50022256960.803764
flops:1000000000000000000000000 => N_opt:126757793201.25563
拟合结果: D_opt = 6.34e-03 * C^0.597
flops:100000000000000000000000 => D_opt:337015838537.39307
flops:1000000000000000000000000 => D_opt:1331744834343.454
拟合结果: Loss_opt = 1.32e+02 * N^-0.153
拟合结果: Loss_opt = 1.07e+02 * D^-0.139
```

## 3 Constructing Scaling Laws

### 内容概况

本部分明确了本阶段的核心任务：通过调用训练API进行实验，收集数据，从而构建出可靠的缩放定律。文本重点说明了**实验目标、方法建议以及一个关键的资源限制**。

---

### 要点总结

1.  **核心任务与目标**：
    *   **最终目标**：构建缩放定律，以**精准预测**在 **1e19 FLOPs** 的计算预算下，能够实现**最低训练损失**的**最优模型大小**。
    *   **主要产出**：预测结果需要包含最优模型大小及其对应的预期训练损失。

2.  **操作方法**：
    *   **使用训练API**：通过向提供的训练API提交不同的实验配置（如模型大小、数据量、超参数），来获取该配置下的训练损失结果。
    *   **分析超参数**：文本建议，为了给预测出的最优模型设置合适的超参数，学生需要**先在小规模实验上分析超参数（如学习率、批大小）对训练损失的影响规律**。

3.  **关键限制与警告**：
    *   **严格的资源预算**：用于构建缩放定律的总计算预算被限定为 **2e18 FLOPs**。一旦所有实验累计消耗的FLOPs超过此限额，API将**拒绝后续的所有请求**。
    *   **规划的重要性**：文本特别强调，学生必须**在开始实验前仔细规划**实验方案，避免因预算耗尽而无法完成目标。

### Problem (scaling_laws): 50 points

Construct a scaling law to accurately predict the optimal model size, its hyperparameters, and the
associated training loss for a FLOPs budget of 1e19. To construct your scaling laws, you will use our
training API to query the final training loss for various experimental configurations (§3.1); you may
not query more than 2e18 FLOPs worth of experiments for fitting your scaling law. This is hard cap
that will be enforced by the API.

- 无API，未实现

## 3.1 Training API

### 内容概况
本部分系统性地介绍了如何通过HTTP接口查询语言模型训练的缩放定律实验结果，包括API的基本信息、访问权限、端点功能、参数规范、返回格式以及具体的代码调用示例。

---

### 要点总结

#### 1. **API 基本信息与访问控制**
- **用途**：用于查询特定训练配置下的最终损失值，以支持缩放定律的实验研究。
- **认证**：使用学期初提供的SSH公钥作为API密钥（需去除换行符）。
- **网络限制**：必须处于斯坦福内网（可能需要VPN），基础文档可访问 `http://hyperturing.stanford.edu:8000/docs` 进行验证。

#### 2. **核心端点功能详解**
- **GET /loss**（核心实验接口）：
  - **输入参数**：包括模型结构（`d_model`、`num_layers`、`num_heads`）、训练超参（`batch_size`、`learning_rate`）、计算规模（`train_flops`）及API密钥。
  - **参数范围**：各参数有严格区间限制（如`d_model`∈[64,1024]），`train_flops`需从预定义的离散值中选择（如1e13至1e18）。
  - **返回值**：JSON对象包含训练损失`loss`和累计消耗的FLOPs总量`total_flops_used`。
  - **重要特性**：重复查询相同配置不会额外消耗FLOPs预算。

- **GET /total_flops_used**（预算查询接口）：
  - **功能**：返回当前API密钥已使用的总FLOPs量，用于监控预算消耗。
  - **异常处理**：无效API密钥返回422错误。

- **GET /previous_runs**（历史记录接口）：
  - **功能**：返回该API密钥下所有已查询实验的详细配置及结果列表。

#### 3. **关键限制与错误处理**
- **资源预算硬约束**：总FLOPs使用量受限（如作业中规定的2e18 FLOPs），超额后API将拒绝请求。
- **参数有效性校验**：超参数超出规定范围时，返回404状态码并提示具体错误（如“d_model must be in range [64, 1024]”）。
- **密钥有效性校验**：无效API密钥返回422错误（如“Invalid API key provided”）。

#### 4. **代码示例与实践指导**
- 提供完整的Python调用示例（使用`requests`库），演示了：
  - **正常请求流程**：构造参数字典，发送GET请求，解析返回的损失和FLOPs用量。
  - **异常场景处理**：包括参数越界、密钥无效等情况的响应示例。
- 强调实验前需**精心规划参数组合**，避免因无效请求或预算耗尽影响实验进度。

### 总结
这组材料为缩放定律实验提供了完整的API工具链，明确了数据查询、预算监控和错误处理的方法。**核心要求是：在有限计算预算下，通过系统化的API调用采集数据，拟合出可靠的缩放定律，最终预测大规模训练的最优配置。** 开发者需严格遵守参数范围，高效利用API资源，并善用历史查询功能优化实验策略。

## 3.2 Training Run Details

### 内容概况
本节详细说明了用于生成缩放定律数据的**具体模型架构和训练配置**。它基于课程第一次作业的Transformer模型，但明确指出了一系列关键改动，旨在为学生提供一个清晰、可复现的实验基准。文档还提供了对应的参考代码文件（`cs336_scaling/model.py`）。

---

### 要点总结

#### 1. **模型架构核心改动（与assignment 1模型对比）**
*   **位置编码**：使用**绝对位置嵌入**，而不是旋转位置嵌入。
*   **归一化层**：使用**层归一化**，而不是RMSNorm。
*   **前馈网络**：采用标准结构（线性层 → GELU激活 → 线性层），隐藏维度为 `4 * d_model`；而不是SwiGLU结构（涉及3个线性层）。
*   **Dropout应用**：明确使用了注意力Dropout和残差Dropout。
*   **输入输出嵌入**：未进行权重绑定。

#### 2. **训练配置与超参数**
*   **数据**：使用SlimPajama数据集。
*   **分词器**：字节对编码分词器，词表大小32K（基于SlimPajama训练）。
*   **上下文长度**：512。
*   **Dropout率**：0.1。
*   **优化器**：AdamW（权重衰减=0.01，梯度裁剪=1.0）。
*   **学习率调度**：采用余弦衰减，将初始学习率降低10倍，无学习率预热。

#### 3. **实践指导意义**
*   此描述定义了**训练API背后模型的具体实现**，学生需要基于此架构来理解和分析API返回的损失结果。
*   提供的代码文件（`cs336_scaling/model.py`）是理解模型细节的权威参考。
