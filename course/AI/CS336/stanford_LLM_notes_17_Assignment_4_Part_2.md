本文主要整理Assignment 4 (data): Filtering Language Modeling Data的主要内容。

## 3. Deduplication

### 内容概况

本部分主要探讨**网络数据中存在的重复内容问题及其处理方案**。章节指出网络中存在大量重复内容，包括完全相同的页面（如404错误页、存档页）和更常见的**页面部分内容重复**（如网站模板、导航栏、页脚等）。该章节将首先解决**精确重复**问题，再进一步处理**近似重复**情况。

---

### 要点总结

1. **重复内容的两大类型**
   - **完全重复页面**：如存档页面、工具生成的默认页面（如404错误页）
   - **部分内容重复**：更常见的情况，页面既有独特内容，也包含大量重复元素

2. **以Stack Overflow为例说明部分重复**
   - **独特内容**：问题、答案、评论等核心信息
   - **重复部分**：页眉、菜单选项、页脚等模板化内容

3. **去重工作的两个阶段**
   - **第一阶段**：处理**精确重复**（完全相同的内容）
   - **第二阶段**：处理**近似重复**（部分相似或修改后的内容）

4. **问题的重要性**
   - 重复内容会**污染训练数据**，降低语言模型训练效率
   - 去除重复是构建高质量数据集的**关键预处理步骤**

## 3.1 Exact line deduplication

### 内容概况
本部分介绍了**精确行去重**的基本方法，这是一种通过消除语料库中完全重复的文本行来减少数据冗余的技术。该方法特别适用于去除网页模板中的重复内容（如页眉、菜单选项等），从而保留每个文档的核心独特内容。

---

### 要点总结

1. **核心目标**  
   - 仅保留语料库中**唯一出现的文本行**，消除大量冗余内容（如网页模板、导航栏等）。
   - 通过去重提取页面的核心信息（如 StackOverflow 的问答内容）。

2. **实施步骤**  
   - **第一次遍历**：统计语料库中每一行的出现次数。
   - **第二次遍历**：仅保留文档中**出现次数为 1** 的独特行。

3. **关键技术优化**  
   - 使用**行的哈希值（而非原始文本）**作为计数器的键，将存储开销从变长字符串转换为固定长度的哈希值，显著节省内存空间。

4. **应用场景**  
   - 适用于处理结构化的重复内容（如网页模板、错误页面、标准化页脚等）。
   - 是后续处理**近似重复**问题的基础步骤。

### Problem (exact_deduplication): 3 points

Write a function that takes a list of paths to input files and performs exact line deduplication on
them. It should first count the frequency of each line in the corpus, using a hash to reduce memory,
and then rewrite each file by only keeping its unique lines.

- 完成

## 3.2 MinHash + LSH document deduplication

### 内容概况

本部分系统性地介绍了**基于MinHash和局部敏感哈希(LSH)的文档模糊去重技术**，主要解决传统精确去重无法处理的**内容相似但非完全相同的文档重复问题**（如软件许可证模板、新闻报道等）。

### 要点总结

1. **技术背景与问题定义**
- **精确去重的局限性**：只能处理完全相同的文档，无法应对模板化内容（如软件许可证）的细微差异
- **模糊去重需求**：需要识别内容高度相似但不完全相同的文档对
- **Jaccard相似度**：采用集合交集与并集的比例作为相似性度量标准

2. **MinHash核心技术**
- **内存优化**：用固定长度的签名替代庞大的n-gram集合，大幅减少内存占用
- **相似性保持**：MinHash签名能够近似保持文档间的Jaccard相似度
- **计算原理**：使用k个哈希函数，每个函数生成文档n-gram的最小哈希值作为签名元素

3. **LSH效率优化**
- **分块策略**：将k维签名划分为b个band，每个band包含r个minhash（k = b×r）
- **候选对筛选**：只有至少一个band完全匹配的文档对才进入精细比较
- **精度-召回权衡**：增加band数提高召回率但降低精度，减少band数则相反

4. **完整处理流程**
1. **签名生成**：为每个文档计算MinHash签名
2. **LSH分桶**：通过band哈希将相似文档聚集到同一桶中
3. **候选对验证**：计算候选对的真实Jaccard相似度，过滤超过阈值的对
4. **聚类去重**：将相互关联的重复文档聚为一类，每类只保留一个代表文档

5. **实际应用价值**
- **大规模处理能力**：特别适合Common Crawl等海量网络文档集合
- **计算效率**：避免O(n²)的成对比较，实现近似线性的时间复杂度
- **质量保证**：在保证去重效果的同时，显著提升处理效率

### Problem (minhash_deduplication): 8 points

Write a function that takes a list of paths to input files and performs fuzzy document deduplication
with minhash and LSH. In particular, your function should compute minhash signatures for each
document in the provided list of paths, use LSH with the provided number of bands to identify candidate
duplicates, and then compute the true ngram Jaccard similarity between candidate duplicates and
remove those that exceed a given threshold. To improve recall (following Penedo et al., 2023), normalize
the text before computing minhash signatures and/or comparing Jaccard similarity by lowercasing,
removing punctuation, normalizing whitespaces, and removing accents, and applying NFD unicode
normalization.

- 完成

## 4. Leaderboard: filter data for language modeling

### 内容概况

要求使用多种技术手段高效处理约 375GB 的压缩文本数据，最终目标是训练 GPT-2 模型并在 Paloma 基准测试上取得良好表现。

### 要点总结

1. 项目目标
- **主要任务**：过滤 Common Crawl 的 WET 文件，生成高质量的语言模型训练数据
- **评估标准**：在 Paloma 基准的 C4 100 个领域子集上最小化验证困惑度
- **约束条件**：不能修改模型架构或训练过程，只能通过数据筛选优化性能

2. 数据资源
- **输入数据**：5000 个 WET 文件（约 375GB 压缩文本），路径为 `/data/CC/CC*.warc.wet.gz`
- **验证数据**：Paloma 的 C4 100 领域验证集，位于 `/data/paloma/tokenized_paloma_c4_100_domains_validation.bin`
- **数据限制**：可以使用验证数据构建过滤器，但不能直接复制到训练数据中

3. 技术要求
- **处理规模**：需要处理大规模数据，推荐使用多进程/分布式计算
- **推荐工具**：`concurrent.futures`、`multiprocessing`、`submitit`（用于 Slurm 集群）
- **数据处理库**：`FastWARC`（读取 WARC 文件）、`tldextract`（域名提取）

4. 执行环境
- 支持单机多进程和分布式集群（Slurm）两种执行模式
- 提供了完整的代码示例和配置参数

### 识别出的代码

#### 代码片段 1：加载验证数据
```python
import numpy as np

data = np.fromfile(
    "/data/paloma/tokenized_paloma_c4_100_domains_validation.bin",
    dtype=np.uint16
)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.decode(data[0:2000]))
```

#### 代码片段 2：单机多进程处理
```python
import concurrent.futures
import os
from tqdm import tqdm

def process_single_wet_file(input_path: str, output_path: str):
    # TODO: read input path, process the input, and write the output to output_path
    return output_path

# Set up the executor
num_cpus = len(os.sched_getaffinity(0))
executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
wet_filepaths = ["a.warc.wet.gz", "b.warc.wet.gz", "c.warc.wet.gz"]
output_directory_path = "/path/to/output_directory/"

futures = []
for wet_filepath in wet_filepaths:
    # For each warc.wet.gz filepath, submit a job to the executor and get a future back
    wet_filename = str(pathlib.Path(wet_filepath).name)
    future = executor.submit(
        process_single_wet_file,
        wet_filepath,
        os.path.join(output_directory_path, wet_filename)
    )
    futures.append(future)
```

#### 代码片段 3：Slurm 集群分布式处理
```python
import os
import submitit
from tqdm import tqdm

def process_single_wet_file(input_path: str, output_path: str):
    # TODO: read input path, process the input, and write the output to output_path
    return output_path

# Set up the submitit executor
executor = submitit.AutoExecutor(folder="slurm_logs")
max_simultaneous_jobs = 16
wet_filepaths = ["a.warc.wet.gz", "b.warc.wet.gz", "c.warc.wet.gz"]
output_directory_path = "/path/to/output_directory/"

# Configure parameters of each job launched by submitit
executor.update_parameters(
    slurm_array_parallelism=max_simultaneous_jobs,
    timeout_min=15,
    mem_gb=2,
    cpus_per_task=2,
    slurm_account="student",
    slurm_partition="a4-cpu",
    slurm_qos="a4-cpu-qos",
)

futures = []
# Use executor.batch() context manager to group all of the jobs in a Slurm array
with executor.batch():
    for wet_filepath in wet_filepaths:
        # For each WARC filepath, submit a job to the executor and get a future back
        wet_filename = str(pathlib.Path(wet_filepath).name)
        future = executor.submit(
            process_single_wet_file,
            wet_filepath,
            os.path.join(output_directory_path, wet_filename)
        )
        # Store the futures
        futures.append(future)

# Use tqdm to display progress
for future in tqdm(
    submitit.helpers.as_completed(futures),
    total=len(wet_filepaths),
):
    output_file = future.result()
    print(f"Output file written: {output_file}")
```

#### 代码片段 4：推荐的数据处理库导入
```python
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract
```

### 项目技术架构分析

这个项目展示了现代大规模语言模型数据预处理的典型工作流：

1. **数据获取**：从 Common Crawl 获取原始网页数据
2. **并行处理**：使用多进程或分布式计算处理海量数据
3. **质量过滤**：基于领域知识和验证数据构建过滤策略
4. **格式转换**：将过滤后的数据转换为模型训练所需的格式
5. **模型训练**：使用处理后的数据训练 GPT-2 模型
6. **评估验证**：在标准基准上评估模型性能

这种架构特别适合处理 TB 级别的网络爬虫数据，是构建高质量语言模型数据集的标准实践。

### Problem (filter_data): 6 points

(a) Write a script to filter language modeling data from a collection of Common Crawl WET files
(located at /data/CC/CC*.warc.wet.gz on the Together cluster). You are free to apply any
of the primitives we’ve implemented in earlier parts of the assignment, and you’re also free to
explore other filters and methods for generating data (e.g., filtering based on n-gram language
model perplexity). Your goal is to produce data that, when trained on, minimizes the perplexity
on the C4 100 domains subset of the Paloma benchmark.
(b) How long does it take to filter the 5,000 WET files? How long would it take to filter the entire
Common Crawl dump (100,000 WETs)?
- 无完整数据，采用CC-MAIN-20250417135010-20250417165010-00065体验流程

### Problem (inspect_filtered_data): 4 points

(a) Take five random examples from your filtered dataset. Comment on their quality and whether
or not they’d be suitable for language modeling, especially given that our goal is to minimize
perplexity on the C4 100 domains benchmark.

(b) Take five CC WETs that were removed and/or modified by your filtering script. What part of
your filtering process removed or modified these documents, and do you think that their removal
and/or modification was justified?

(c) If your analysis above motivates further changes to your data pipeline, feel free to make those
changes before training your model. Report any changes and/or iterations of data that you
experimented with.
- 无完整数据，采用CC-MAIN-20250417135010-20250417165010-00065体验流程

## 4. tokenize_data

### 1. 核心处理流程
- **标记化转换**：使用 GPT-2 分词器将文本转换为整数 ID 序列
- **结束符添加**：每个文档后必须添加 `<|endoftext|>` 结束标记
- **并行处理**：利用多进程池加速大规模文本处理
- **二进制存储**：将结果保存为 uint16 格式的 NumPy 二进制文件

### 2. 技术要点
- **分词器选择**：Hugging Face Transformers 库的 AutoTokenizer
- **并行优化**：multiprocessing.Pool 实现多进程并行处理
- **内存优化**：使用 uint16 数据类型节省存储空间
- **进度监控**：tqdm 提供实时处理进度显示

### 3. 文件处理规范
- 输入：过滤后的纯文本数据（每行一个文档）
- 输出：二进制格式的令牌 ID 序列文件
- 格式：NumPy array 直接写入二进制文件

### Problem (tokenize_data): 2 points

Write a script to tokenize and serialize your filtered data. Make sure to serialize following the
example code above, with ids_array.tofile(output_path), where ids_array is a np.uint16 numpy
array of integer IDs. This ensures compatibility with the provided training script.
How many tokens are in your filtered dataset?

- 无完整数据，采用CC-MAIN-20250417135010-20250417165010-00065体验流程

## 4. train_model

### 要点总结

#### 训练阶段要点
1. **模型配置**：使用 GPT-2 small 架构，训练 200K 迭代步数
2. **数据要求**：需要使用预处理后的标记化数据（tokenized data）
3. **硬件配置**：2个GPU，数据并行，每设备批量大小128
4. **时间预估**：完整训练约需7小时
5. **监控配置**：需要设置 Weights & Biases 进行训练监控
6. **约束条件**：只能优化数据质量，不能修改模型架构或训练过程

#### 工程实践要点
1. **配置文件管理**：通过 YAML 文件管理训练参数和数据路径
2. **检查点机制**：支持保存验证损失最低的模型检查点
3. **分布式训练**：使用 torchrun 进行多GPU训练
4. **依赖管理**：使用 uv 工具管理Python环境
5. **测试便利性**：提供快速验证和样本生成功能

### 识别打印的代码

1. 训练配置代码
```yaml
# 文件路径: cs336-basics/configs/experiment/your_data.yaml
paths:
  train_bin: "path/to/your/tokenized/training/data"  # 需要设置的实际路径
training:
  wandb_entity: "your_wandb_entity"    # 需要设置
  wandb_project: "your_wandb_project"  # 需要设置
```

2. 模型训练命令
```bash
# 基础训练命令
uv run torchrun --standalone --nproc_per_node=2 scripts/train.py --config-name=experiment/your_data

# 启用检查点保存的训练命令
uv run torchrun --standalone --nproc_per_node=2 \
scripts/train.py --config-name=experiment/your_data \
+training.save_checkpoints=True
```

3. 文本生成命令
```bash
# 从训练好的模型生成文本样本
uv run python scripts/generate_with_gpt2_tok.py \
--model_path cs336-basics/output/your_data/step_N
```

4. 关键Python脚本路径
```python
# 训练脚本
cs336-basics/scripts/train.py

# 超参数配置
cs336-basics/cs336_basics/train_config.py

# 文本生成脚本
scripts/generate_with_gpt2_tok.py
```

### 完整工作流程示例

```bash
# 步骤1: 配置训练参数
# 编辑 cs336-basics/configs/experiment/your_data.yaml
# 设置 paths.train_bin, training.wandb_entity, training.wandb_project

# 步骤2: 启动训练（基础版本）
uv run torchrun --standalone --nproc_per_node=2 scripts/train.py --config-name=experiment/your_data

# 步骤3: 或者启动带检查点保存的训练
uv run torchrun --standalone --nproc_per_node=2 \
scripts/train.py --config-name=experiment/your_data \
+training.save_checkpoints=True

# 步骤4: 训练完成后生成文本样本
uv run python scripts/generate_with_gpt2_tok.py \
--model_path cs336-basics/output/your_data/step_200000
```

### 注意事项
1. 路径中的 `step_N` 需要替换为实际训练步数（如 step_200000）
2. 所有命令需要在项目根目录下执行
3. 需要提前安装 uv、torch 等相关依赖
4. 训练前确保已准备好标记化的训练数据


### Problem (train_model): 2 points

Train a language model (GPT-2 small-shaped) on your tokenized dataset. Periodically mea-
sure the validation loss on C4 100 domains (this is already enabled by default in the config at
cs336-basics/cs336_basics/train_config.py). What is the best validation loss that your model
achieves? Submit this value to the leaderboard.

- 无完整数据，采用CC-MAIN-20250417135010-20250417165010-00065体验流程