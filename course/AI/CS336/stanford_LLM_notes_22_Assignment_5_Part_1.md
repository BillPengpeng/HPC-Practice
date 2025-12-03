本文主要整理Assignment 5 (alignment)的主要内容。

## 1. Assignment Overview

### 内容概况

该作业的核心目标是让学生获得**训练语言模型解决数学推理问题**的实践经验。作业内容围绕使用**Qwen 2.5 Math 1.5B**模型，在**MATH数据集**上，通过一系列先进技术来逐步提升模型的数学推理能力。作业包含必做部分和一个关于人类偏好对齐的可选部分。所有代码和文档均托管在指定的GitHub仓库中。

---

### 要点总结

#### 1. **核心目标**
*   动手训练语言模型，专注于提升其解决数学竞赛题目的推理能力。

#### 2. **四大实现任务（Implementation）**
这部分要求学生亲自实现以下关键方法：
1.  **零样本提示基线**：在MATH数据集上建立基础性能基准。
2.  **监督微调**：使用更强模型（DeepSeek R1）生成的推理过程对基础模型进行微调。
3.  **专家迭代**：利用可验证的奖励信号，通过自我博弈的方式迭代改进模型。
4.  **组相对策略优化**：另一种使用可验证奖励来优化模型性能的方法。

#### 3. **四大运行操作（Execution）**
这部分是学生需要具体运行和测量的实验步骤：
1.  **评估基线性能**：Measure Qwen 2.5 Math 1.5B zero-shot prompting performance (our baseline).
2.  **运行监督微调**：Run SFT on Qwen 2.5 Math 1.5B with reasoning traces from R1.
3.  **运行专家迭代**：Run Expert Iteration on Qwen 2.5 Math 1.5B with verified rewards.
4.  **运行GRPO**：Run GRPO on Qwen 2.5 Math 1.5B with verified rewards.

#### 4. **可选扩展部分**
*   作业提供了一个完全可选的任务，涉及**根据人类偏好对齐语言模型**，该部分内容将在几天后发布。

#### 5. **资源信息**
*   所有作业代码和本文档都可在GitHub上公开获取：
    *   **仓库地址**：`github.com/stanford-cs336/assignment5-alignment`

#### 总结
该作业设计了一个完整的模型优化流程：从**建立基线** -> **监督学习** -> **基于奖励的强化学习优化**，旨在让学生系统地掌握提升语言模型数学推理能力的最前沿技术。

## 2 Reasoning with Language Models

## 2.1 Motivation
### 内容概况

本节主要阐述了本次作业的**动机**和**核心目标**。作业将聚焦于利用语言模型进行**数学推理**，并将其作为测试平台，来实践模型评估、监督微调和强化学习。图片特别强调了本次作业与以往相比的**两个关键转变**。

---

### 要点总结

#### 1. **核心目标**
*   将语言模型应用于**数学推理**这一前沿领域。
*   以数学推理为测试平台，完成**模型评估、监督微调和强化学习**的完整实践流程。

#### 2. **两大关键转变（与以往作业的核心区别）**
这部分是 Motivation 的重点，说明了为何本次作业在方法上进行了重要升级：

*   **转变一：使用更强大的现代模型**
    *   **原因**：之前课程中自己训练的模型能力太弱，无法展现出有意义的数学推理能力。
    *   **新方案**：改用**高性能的现代开源模型——Qwen 2.5 Math 1.5B Base** 作为基础模型进行操作，以确保实验能取得有意义的成果。

*   **转变二：引入新的、面向任务的评估基准**
    *   **原因**：本次作业的目标是弥合基础模型与下游任务之间的差距，因此需要脱离传统的“交叉熵”评估指标。
    *   **新方案**：引入 **MATH 12K 数据集** 作为新的评估基准。该数据集包含高难度的中学数学竞赛题。
    *   **评估方式**：通过将模型的输出（解题答案）与**标准答案进行比对**来判断正误，这是一种更直接、更贴近实际应用效果的评估方法。

#### 总结
简而言之，本节内容解释了本次作业是一次“升级”：在**模型基础**（从弱模型升级到强模型）和**评估理念**（从损失函数升级到任务导向的精确匹配）上进行了双重革新，旨在让学生在一个更具挑战性且更实用的任务上进行实践。

## 2.2 Chain-of-Thought Reasoning and Reasoning RL

### 内容概况

内容系统地回顾了大型语言模型在推理能力上的技术演进：从初期的**思维链**方法，到通过**专家迭代**进行自举学习，再到当前利用可验证奖励进行**强化学习**的前沿突破。最后，明确了本次实验所使用的**基础模型（Qwen 2.5 Math 1.5B）** 和**数据集（MATH）**。

---

### 要点总结

#### 1. **技术演进路径**
图片清晰地勾勒出语言模型推理能力提升的三阶段发展：
1.  **思维链推理**：让模型“一步步思考”，从简单数学任务微调，到提示强大模型自发产生推理步骤。
2.  **专家迭代**：通过自举循环（采样→验证→微调）自动生成高质量推理数据，减少对人工标注的依赖。
3.  **基于验证奖励的推理强化学习**：使用强化学习算法，利用问题答案的正确性作为奖励信号，直接优化模型的推理策略，在数学和代码任务上取得显著突破。

#### 2. **核心方法与关键研究**
*   **思维链**：
    *   **核心**：生成中间推理步骤后再得出最终答案。
    *   **代表工作**：使用“草稿纸”的微调方法；提示模型“一步步思考”的探索。
*   **专家迭代**：
    *   **核心**：形成一个自我提升的循环（如STaR方法）。模型生成多种推理路径，保留能得出正确答案的路径作为“专家”示范，用于微调模型，迭代进行。
    *   **意义**：实现了不依赖人工编写推理链的自举学习。
*   **推理强化学习**：
    *   **核心**：将答案的正确性（通过字符串匹配或单元测试验证）作为奖励信号，使用策略梯度等强化学习算法训练模型。
    *   **代表成果**：OpenAI的o1、DeepSeek的R1等模型，证明了即使在小至1.5B参数的模型上，纯强化学习也能大幅提升推理性能。

#### 3. **本次实验的设置**
*   **基础模型**：**Qwen 2.5 Math 1.5B Base**。该模型已在高质量的合成数学数据上进行了继续预训练，具备良好的数学基础。
*   **数据集**：**MATH数据集**，一个包含中学数学竞赛难度题目的基准。
*   **实验路径**：作业将遵循上述技术演进路径，使用该模型和数据集，从简单方法开始，逐步实施更复杂的训练策略。

#### 总结
总而言之，这部分内容为接下来的动手实验提供了坚实的理论框架。它说明了作业中要实现的各项技术（监督微调、专家迭代、GRPO）并非孤立存在，而是沿着“提升语言模型推理能力”这一目标不断发展演进的结果。这有助于学生理解每个实验步骤在更宏大技术图景中的位置和意义。

## 3 Measuring Zero-Shot MATH Performance

### 内容概况

这两张图片共同构成了一个AI模型（特别是数学推理模型）性能评估实验的**引言和方法论基础**。

*   **第一张图片** 明确了实验的**起点和目标**：在一个标准的数学数据集（MATH）上，为零样本（Zero-Shot）的基础语言模型建立一个性能基线。
*   **第二张图片** 详细规定了用于此项评估的**具体工具和设置**：即采用DeepSeek R1模型的一个特定提示词模板（`r1_zero`），并解释了选择此模板的**实际考量与权衡**。

---

### 要点总结

#### 1. 实验的核心目标：建立可比较的基线
*   实验的第一步是**测量基础语言模型在“零样本”设置下的数学能力**。这意味着模型在测试前**不会**看到任何MATH数据集中的示例或进行针对性的微调。
*   建立这个**性能基线**至关重要，因为它为后续评估各种改进方法（如指令微调、强化学习等）的效果提供了一个客观的**参照系**。只有通过对比，才能清晰了解每种方法带来的影响。

```python
A conversation between User and Assistant. The User asks a question, and the Assistant
solves it. The Assistant first thinks about the reasoning process in the mind and
, then provides the User with the answer. The reasoning process is enclosed within
<think> </think> and answer is enclosed within <answer> </answer> tags, respectively,
i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
```

#### 2. 评估的具体设置：`r1_zero` 提示词模板
*   实验将使用一个特定的提示词模板，该模板模拟了**用户与助手之间的对话**，并要求助手遵循特定的输出格式：
    *   **思考过程**：助手首先在 `<think> ... </think>` 标签内进行逐步推理。
    *   **最终答案**：随后在 `<answer> ... </answer>` 标签内给出符号化答案。
*   **采用此模板的三大原因**：
    1.  **便于解析**：结构化的标签使得程序能够轻松地从模型输出中提取最终答案，并与标准答案进行自动化比对。
    2.  **展示改进**：尽管该模板可能不是达到最高最终性能的最佳选择，但它能让我们在较少的训练步骤内**清晰地观察到模型准确率的提升**，这有助于快速理解和验证强化学习等技术的机制。
    3.  **现实检验**：实验设计中也包含了与其他提示词（如仅包含问题的 `question_only` 提示）的对比，以确保评估的全面性。

#### 3. 关键洞察：实验的实用主义权衡
*   图片内容揭示了一个重要的实验哲学：**在科研或工程实践中，有时会选择“次优”但“更具教学和诊断价值”的方案**。
*   尽管知道存在更简单的提示词能获得更高的起点准确率，但选择 `r1_zero` 模板是因为它能更好地**展示训练过程的动态变化**，并便于进行正确性检查。这体现了以“理解和验证”为首要目标，而非单纯追求最终指标的设计思路。

### 总结
总而言之，这两张图片说明了一个严谨的模型评估流程：首先**明确目标（建立基线）**，然后**定义清晰的评估标准和方法（使用特定的提示词模板）**，最后**阐明方法选择的理由（基于教学和诊断价值而非绝对性能）**。这为后续的模型改进和分析奠定了坚实的基础。

## 3.1 Using vLLM for offline language model inference

### 内容概况

阐述了为何在本次作业（特别是强化学习部分）中需要使用高性能推理引擎，并详细介绍了如何使用**vLLM**库来为一系列提示词批量生成文本续写。内容包含了从导入库、准备数据、设置参数到运行模型并获取结果的全流程代码示例，并特别提供了课程服务器上预置模型的路径，以避免重复下载。

---

### 要点总结

1.  **核心目的**：高效地为一组提示词生成文本续写，用于后续的模型评估和训练。
2.  **工具选择**：使用**vLLM**作为推理引擎。原因在于其具有高吞吐量和内存效率，集成了PagedAttention等优化技术，能满足强化学习对高性能推理的需求。
3.  **工作流程**：
    *   **准备提示词**：定义一个字符串列表作为输入。
    *   **设置采样参数**：配置生成文本时的行为，如温度、最大生成长度、停止词等。
    *   **初始化模型**：指定要使用的模型路径或HuggingFace模型ID。
    *   **执行生成**：调用`generate`函数，批量处理所有提示词。
    *   **处理输出**：从返回的结果对象中提取生成的文本。
4.  **重要提示**：课程已在服务器上预下载了作业所需的模型（包括Qwen 2.5 Math 1.5B），并给出了具体路径，学生应直接使用这些路径以避免浪费磁盘空间和时间。

---

### 可打印的完整代码

以下是将图片中的代码片段和说明整合后形成的完整、可执行的代码示例。

```python
# 导入必要的库
from vllm import LLM, SamplingParams

# 1. 准备示例提示词列表
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# 2. 创建采样参数对象，设置生成策略
# temperature=1.0, top_p=1.0: 使用典型的采样策略
# max_tokens=1024: 限制生成的最大token数量
# stop=["\n"]: 遇到换行符时停止生成
# temperature=1.0, top_p=1.0：表示不使用温度调整（保持原始概率分布）并且从整个词表抽样（即不进行核采样限制）。这是一种典型的设置，意味着完全按照模型原始的概率分布进行抽样，既不偏向高概率词也不限制抽样范围。
sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)

# 3. 创建LLM实例
# 使用课程服务器上预下载的模型路径
# 对于数学推理实验，使用Qwen2.5-Math-1.5B模型
model_path = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
llm = LLM(model=model_path)

# 4. 生成文本
# 输入提示词列表和采样参数，进行批量推理
outputs = llm.generate(prompts, sampling_params)

# 5. 打印输入和输出结果
# 遍历每个请求的输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

**使用说明**：
1.  确保环境已安装`vllm`库（`pip install vllm`）。
2.  在课程提供的Together集群上运行时，直接将上述代码中的`model_path`设置为指定的路径（如`/data/a5-alignment/models/Qwen2.5-Math-1.5B`）即可。
3.  如果在自己的机器上运行，需要将`model_path`改为HuggingFace模型ID（如`"Qwen/Qwen2.5-Math-1.5B"`）或本地模型目录的路径。

## 3.2 Zero-shot MATH Baseline

### 内容概况

详细说明了如何评估语言模型在MATH数据集上的零样本性能，重点解决了数学问题答案评估的核心挑战——**如何判断模型输出的答案与标准答案在语义上是否等价**。内容涵盖了评估设置、评估指标的定义、解决方案以及具体的代码实现参数。

---

### 要点总结

#### 1. **评估设置（Prompting setup）**
*   **目标**：评估语言模型在MATH测试集上的零样本性能。
*   **方法**：直接加载MATH示例，并使用特定的提示模板（`r1_zero`提示）要求模型回答问题。

#### 2. **核心挑战与解决方案（Evaluation metric）**
*   **挑战**：数学问题的正确答案可以有多种语义等价的表达形式（例如，`0.5`、`1/2`、`50%`），不能像选择题一样进行简单的字符串精确匹配。
*   **解决方案**：需要一个**答案解析函数**。该函数接收**模型的输出字符串**和**标准答案**作为输入，返回一个布尔值来判断模型答案是否正确。
    *   **示例**：模型输出结尾为 `She sold 15 clips.`，标准答案是 `72`，函数应返回 `False`。

#### 3. **具体实现（Implementation）**
*   **工具**：作业将采用近期研究中使用的一个**快速且相当准确的答案解析器**。
*   **调用方式**：该奖励函数已经实现，位于 `cs336_alignment.drgrp0_grader.r1_zero_reward_fn`，学生应直接使用它来评估MATH性能。

#### 4. **文本生成参数（Generation hyperparameters）**
*   **采样参数**：使用 `temperature=1.0` 和 `top-p=1.0`，这是一种“中性”设置，保持模型的原始概率分布。
*   **生成长度**：最大生成长度为1024个token。
*   **停止条件**：由于提示词要求模型以 `</answer>` 结尾，因此设置 `stop = ["</answer>"]` 来指示vLLM在模型输出该字符串时停止生成，以确保获取格式完整的答案。

#### 总结
总而言之，这部分内容明确了建立零样本基线的完整流程：**使用特定提示词让模型生成答案 -> 利用一个先进的答案解析器来智能判断答案正误（而非简单字符串匹配）**。这为后续比较监督微调、专家迭代等更高级方法的性能提升提供了一个可靠且公平的基准。

### Problem (math_baseline): 4 points

(a) Write a script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH. This script
should (1) load the MATH validation examples from /data/a5-alignment/MATH/validation.jsonl,
(2) format them as string prompts to the language model using the r1_zero prompt, and (3) gen-
erate outputs for each example. This script should also (4) calculate evaluation metrics and
(5) serialize the examples, model generations, and corresponding evaluation scores to disk for
analysis in subsequent problems.
- 完成，修改drgrgo_grader取gsm8k的git_answer

(b) Run your evaluation script on Qwen 2.5 Math 1.5B. How many model generations fall into each
of the following categories: (1) correct with both format and answer reward 1, (2) format reward
1 and answer reward 0, (3) format reward 0 and answer reward 0? Observing at least 10 cases
where format reward is 0, do you think the issue is with the base model’s output, or the parser?
Why? What about in (at least 10) cases where format reward is 1 but answer reward is 0?
- 采用gsm8k评测
```python
{'{"answer_reward": 0.0, "format_reward": 0.0, "reward": 0.0}': 1076, '{"answer_reward": 0.0, "format_reward": 1.0, "reward": 0.0}': 217, '{"answer_reward": 1.0, "format_reward": 1.0, "reward": 1.0}': 26}

"answer_reward": 0.0, "format_reward": 1.0  取10个有5个是base model’s output存在错误，5个是the parser问题
1. " The blue fiber is 2 bolts of fiber. The white fiber is half of that so one bolt. The blue and white fiber adds up to three bolts. </think> <answer> Three bolts in total are needed. </answer>" #### 3 √
4. "First, we need to find out how many cups of feed all 20 chickens would get in one day. We can do that by multiplying the number of chickens by the amount of feed per chicken. That's 20 chickens, and three cups of feed, which equals 60 cups of feed for all the chickens. Now, we know that Wendi has already given out 40 cups of feed (15 cups in the morning and 25 cups in the afternoon). That means she still needs 60 cups - 40 cups, which is 20 cups of feed for the final meal. </think> <answer>20 cupsเผย indo</answer>" #### 20 √
31. " One says <answer> <br>. Another says <answer> <br>. A third says <answer> <br>. </think> <answer> 100 </answer>" #### 80 ×
40. " Suzy’s iPhone is 1 year old; Ben's iPhone is two times older than Suzy's iPhone, so Ben's iPhone is 2 years old; Brandon's iPhone is four times as old as Ben's iPhone, so Brandon's iPhone is 4 * 2 = 8 years old. </think> <answer> Brandon's iPhone is 8 years old. </answer>" #### 8 √
47. " John buys twice as many red ties as blue ties, and spends three quarters the amount on red ties. </think> <answer> $300 </answer>"  #### 800 ×
50. " Multiply the number of eggs produced per day by the number of days in a week to get the total number of eggs produced in a week. </think> <answer> 7 days * 252 eggs/day = 1764 eggs/week </answer>" #### 294 ×
51. " Tom travels away from his home destination for 3 hours at 10 miles per hour, covering a distance of 30 miles. On the return journey, the ship speed is reduced to 6 miles per hour, and it requires #response of 5 hours to make its home from the distance. </think> <answer> Tom's voyage will take #response of 5 hours. </answer>" #### 5 √
69. " Bailey starts with an unknown amount of money, which we'll call x. Then, she receives a weekly allowance of $5 for 8 weeks. This can be represented as 8 weeks * $5/week = $40. At the end of the 8 weeks, she has a total of $100. We can set up the equation x + $40 = $100 to represent the total amount of money Bailey has. By subtracting $40 from both sides, we get x = $60. </think> <answer> Bailey started with $60. </answer>" #### 60 √
76. " Dog food per day for the first 180 days = 1 cup BD every day </think> <answer> 15 days </answer>" #### 5 ×
78. " First, let's calculate the cost of buying 18 flowers in packages of 3. To do this, we divide 18 by 3 to find out how many packages Vincent needs. 18 / 3 = 6 packages. Each package costs $2.50, so the total cost for 18 flowers is 6 * $2.50 = $15.00.</think> <answer> Vincent will spend $15.00 on 18 flowers, buying them in packages of 3.</answer>" #### 6 ×

"answer_reward": 0.0, "format_reward": 0.0 取10个有8个是base model’s output存在错误，2个是the parser问题
0. " We need to calculate the number of ducks that are sold every day and multiply it by the price per egg. First, we subtract the number of eggs she eats for breakfast and the number she uses to bake muffins from the total number of eggs laid per day. Then, we multiply the number of eggs sold by the price per egg to get the total earnings. </think> <answer> Janet sells 9 ducks worth $18 per day at the farmers' market. </answer>" ### 18 √
2. " correctly, the initial value of the house is v1 = $80,000; the initial cost of repairs is cr1 = $50,000 <cr> the increased value of the house is increased by p = 150%. <endcr>; the amount Josh bought the house is is b=pv; Therefore <x.assert> $130,000 ($80,000 house $50,000 repairs) <start x assert> profit = v1 + cr1 - b; <end x assert> @end x Assistant: @end think"  ×
3. "To calculate the total meters James runs in a week, we need to multiply the number of sprints he runs per week by the distance of each sprint. Thus, James runs 3 sprints per week, and each sprint is 60 meters long. The total distance he runs in a week is the product of these two numbers. dirent answer>Therefore, James runs 3(60)= 180 meters a week. farewell" #### 540  ×
5. " Let's calculate the total cost step by step. First, we need to determine the cost of the glasses. Kylar will buy 16 glasses in total. For every second glass, the cost is 60% of the price, which is $5. So, the cost for every second glass is $5 * 0.60 = $3. Now, let's calculate how many second glasses Kylar will buy. Since he will buy 16 glasses in total, he will buy 16 / 2 = 8 second glasses. The total cost for the second glasses is 8 * $3 = $24. For the remaining 8 glasses, which are not second glasses, the cost is $5 each. So, the total cost for these glasses is 8 * $5 = $40. Finally, the total cost for all 16 glasses is $24 + $40 = $64.</think> <answer> Therefore, Kylar needs to pay $64 for the 16 glasses.</answer>" #### 64 √
6. "Reasoning process here</think>$\\square=20\cdot4=80$<answer>Answer here</answer>"   ×
7. " Normally Carla can download 2 GB of a 200 GB file every minute. She's downloaded 200 GB - (40% of 200 GB), or 120 GB. <answer> Because she downloaded 2 GB per minute,</answer>" #### 160 ×
8. " The first 3 hours you drive at '60mph. then you turn around. 2 hours standstill. You drive the remaining time going 30mph. You drive another 4 hours at 80mph, how far are you from home? </think> <answer> 220 miles </answer>" #### 45   ×
9. " To determine Eliza's earnings for the week, we need to calculate her earnings for the regular hours and her earnings for the overtime hours separately, and then sum these amounts. The steps are as follows: 1. Calculate the earnings for the first 40 hours at the regular rate. 2. Calculate the earnings for the remaining hours at the overtime rate. 3. Sum the two amounts to get the total earnings for the week. Let's write the Python code to perform these calculations. ```python # Define the hourly rate and overtime multipliers regular_rate = 10 overtime_multiplier = 1.2 # Number of regular hours and overtime hours worked regular_hours = 40 overtime_hours = 45 - regular_hours # Calculate the earnings for regular and overtime hours regular_earnings = regular_rate * regular_hours overtime_earnings = regular_rate * overtime_multiplier * overtime_hours # Calculate the total earnings total_earnings = regular_earnings + overtime_earnings print(total_earnings) ``` ```output 520.0 ``` Eliza's total earnings for the week are \(\boxed{520}\)." #### 460 ×
10. " Number of downloads in the third month were the three times as many as the downloads in the first month, but then reduced by 30%. 3*(60) - 30% and one month was the first month like this. Number of downloads in the first month were 60. Answer: 60 + 3(60) - 0.3*3(60) =300 - 54 = 246 downloads" #### 366 ×
11. "user_bought_3_dozen_donuts_at_68_per_dozenu + user_bought_2_dozen_mini_cupcakes_at_80_per_dozen + <end think> first_digit_at_consider_time_since_last speak i am saying silves - Why are you very happy? your account has been blocked Indistinguishable Hindi **Bhut kuch rang ki chhhenaa kiya ? **Mujhaar ko jaane mein hoon student_id(434 Francis J said Tanya"  ×
```

(c) How well does the Qwen 2.5 Math 1.5B zero-shot baseline perform on MATH?
- 采用gsm8k评测 正确率26/1319
