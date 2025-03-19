本文主要整理Bert及衍生模型的Embeddings、Tokenizer。

## 1. Embeddings

### 1.1 BertEmbeddings

#### **1. 初始化阶段 (`__init__`)**
**嵌入层定义**
```python
def __init__(self, config):
    super().__init__()
    # 词嵌入（WordPiece）
    self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    # 绝对位置嵌入
    self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    # 段落类型嵌入（Segment Embeddings）
    self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
    
    # 后处理层
    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 预注册缓冲区（非持久化，不保存到模型文件）
    self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
    self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)
```

**关键组件说明**
| **组件**                | **作用**                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| **`word_embeddings`**    | 将输入 Token ID 映射为词向量（`vocab_size` → `hidden_size`）            |
| **`position_embeddings`**| 为每个位置生成绝对位置编码（`max_position_embeddings` 为最大序列长度） |
| **`token_type_embeddings`** | 区分句子 A/B 的段标记（如问答任务中问题和上下文的区分）              |
| **`LayerNorm`**           | 对嵌入结果归一化，稳定训练过程                                       |
| **`dropout`**             | 随机失活防止过拟合                                                  |

#### **2. 前向传播 (`forward`)**
**输入参数**
- **`input_ids`**：形状 `(batch_size, seq_len)` 的 Token ID 序列
- **`token_type_ids`**：形状 `(batch_size, seq_len)` 的段标记（0 表示句子 A，1 表示句子 B）
- **`position_ids`**：显式指定的位置 ID（通常自动生成）
- **`inputs_embeds`**：直接提供词向量（绕过 `word_embeddings`）
- **`past_key_values_length`**：历史缓存长度（用于生成任务处理长序列）

**处理流程**
**a. 确定输入形状**
```python
if input_ids is not None:
    input_shape = input_ids.size()  # (batch_size, seq_len)
else:
    input_shape = inputs_embeds.size()[:-1]  # (batch_size, seq_len)
seq_length = input_shape[1]
```

**b. 生成位置 ID**
```python
if position_ids is None:
    # 从缓存位置后截取当前序列长度
    position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
```
- **示例**：  
  若 `past_key_values_length=5`，当前 `seq_length=3`，则位置 ID 为 `[5,6,7]`

**c. 生成段标记 ID**
```python
if token_type_ids is None:
    # 使用预注册的全零缓冲区扩展至 batch 维度
    buffered_token_type_ids = self.token_type_ids[:, :seq_length]
    token_type_ids = buffered_token_type_ids.expand(input_shape[0], seq_length)
```
- **作用**：  
  若未提供 `token_type_ids`，默认所有 Token 属于句子 A（全零）

**d. 计算各嵌入分量**
```python
# 词嵌入（直接输入或通过 ID 映射）
inputs_embeds = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
# 段嵌入
token_type_embeddings = self.token_type_embeddings(token_type_ids)
# 绝对位置嵌入
position_embeddings = self.position_embeddings(position_ids)
```

**e. 融合嵌入 + 后处理**
```python
embeddings = inputs_embeds + token_type_embeddings + position_embeddings
embeddings = self.LayerNorm(embeddings)
embeddings = self.dropout(embeddings)
```
词嵌入 + 段嵌入 + 位置嵌入 → LayerNorm → Dropout

#### **3. 关键设计细节**
**a. 位置编码的灵活性**
- **绝对位置编码**：  
  当前实现为绝对位置编码（每个位置独立学习），通过 `position_embedding_type` 可扩展支持相对位置编码（需修改此处逻辑）。
- **动态位置适应**：  
  通过 `past_key_values_length` 支持自回归生成任务中位置偏移（如 GPT 的缓存机制）。

**b. 段标记的自动化处理**
- **默认全零**：  
  当输入为单句或用户未提供段标记时，自动填充为 0，简化接口调用。
- **缓冲区加速**：  
  预注册 `token_type_ids` 缓冲区避免重复生成全零张量，提升效率。

**c. 嵌入融合方式**
- **逐元素相加**：  
  词、位置、段嵌入直接相加，通过后续 LayerNorm 调整分布。
- **对比其他方法**：  
  某些模型（如 BERT 变体）可能选择拼接（Concatenation）或其他融合方式，但相加更节省计算资源。

#### **4. 输入输出示例**
**输入**
```python
input_ids = [[101, 2054, 2003, 1996, 4248, 102]]  # "hello world" 的 Token ID
token_type_ids = [[0, 0, 0, 0, 0, 0]]             # 单句子标记
```

**处理步骤**
1. **词嵌入**：将每个 ID 映射为 768 维向量。
2. **段嵌入**：全零，无附加信息。
3. **位置嵌入**：位置 0~5 的编码向量。
4. **相加 + LayerNorm**：融合后归一化。
5. **Dropout**：随机置零部分神经元。

**输出**
- 形状：`(1, 6, 768)`，即每个 Token 的最终嵌入表示。


#### **5. 与标准 Transformer 嵌入层的区别**
| **特性**           | **标准 Transformer**               | **BERT Embeddings**               |
|--------------------|------------------------------------|-----------------------------------|
| **位置编码**       | 正弦/余弦公式计算                  | 可学习的绝对位置嵌入               |
| **段标记**         | 无                                 | 支持句子对任务（问答、NSP）        |
| **归一化位置**     | 无                                 | 嵌入相加后立即 LayerNorm           |
| **处理流程**       | 仅词嵌入 + 位置编码                | 词 + 段 + 位置 → LayerNorm → Dropout |


#### **6. 总结**
`BertEmbeddings` 是 BERT 模型处理输入的**第一层**，核心功能为：
1. **三重嵌入融合**：联合编码词汇、位置和段落信息。
2. **归一化与正则化**：通过 LayerNorm 和 Dropout 提升训练稳定性。
3. **灵活输入支持**：自动处理缺失的段标记，适配生成任务的位置偏移。

### 1.2 RobertaEmbeddings

#### **1. 与 BertEmbeddings 的总体对比**
`RobertaEmbeddings` 继承自 BERT 的嵌入层结构，但针对 RoBERTa 的预训练任务特点进行了 **两处关键调整**：
1. **位置嵌入的填充索引处理**：显式指定 `padding_idx`，优化填充位置的位置编码。
2. **动态位置 ID 生成**：根据输入自动计算位置 ID，适配无 NSP 任务的长序列。

#### **2. 关键代码差异解析**

**a. 位置嵌入层初始化**
```python
# BertEmbeddings
self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

# RobertaEmbeddings
self.position_embeddings = nn.Embedding(
    config.max_position_embeddings, 
    config.hidden_size, 
    padding_idx=self.padding_idx  # 新增
)
```
- **差异点**：RoBERTa 的位置嵌入层指定 `padding_idx`，使填充符的位置向量保持固定（通常初始化为零），避免梯度更新干扰。
- **作用**：提升模型对填充符的鲁棒性，尤其在处理变长序列时。

**b. 位置 ID 生成逻辑**
```python
# BertEmbeddings
if position_ids is None:
    position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

# RobertaEmbeddings
if position_ids is None:
    if input_ids is not None:
        position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
    else:
        position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
```
- **动态生成**：根据输入 ID 或嵌入向量动态计算位置 ID，而非依赖预注册的缓冲区。
- **create_position_ids_from_input_ids**：  
  ```python
  def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
      mask = input_ids.ne(padding_idx).int()
      position_ids = torch.cumsum(mask, dim=1).int() * mask + padding_idx
      return position_ids[:, past_key_values_length:]
  ```
  - **逻辑**：非填充符位置从 1 开始递增，填充符位置设为 `padding_idx`。
  - **示例**：  
    `input_ids = [[1, 2, 0, 0]]` → `position_ids = [[1, 2, 0, 0]]`（假设 `padding_idx=0`）

**c. 输入嵌入时的位置 ID 生成**
```python
def create_position_ids_from_inputs_embeds(self, inputs_embeds):
    sequence_length = inputs_embeds.size(1)
    position_ids = torch.arange(
        self.padding_idx + 1,  # 起始位置为 padding_idx + 1
        sequence_length + self.padding_idx + 1,
        device=inputs_embeds.device
    )
    return position_ids.unsqueeze(0).expand(inputs_embeds.size()[:-1])
```
- **设计意图**：当直接提供 `inputs_embeds`（无法推断填充位置）时，生成连续位置 ID，起始位置跳过 `padding_idx`。
- **示例**：  
  `sequence_length=3`, `padding_idx=0` → `position_ids = [1, 2, 3]`

#### **3. 改进动机与效果**
**a. 更精确的填充处理**
- **BERT 的缺陷**：预注册的 `position_ids` 无法区分填充位置，可能导致填充符参与位置编码计算。
- **RoBERTa 的优化**：动态生成位置 ID，确保填充符位置固定为 `padding_idx`，其对应位置嵌入不参与有效计算。

**b. 适配无 NSP 任务**
- **预训练任务差异**：RoBERTa 移除了 NSP 任务，专注于更长序列的 MLM。
- **位置编码增强**：动态生成机制支持超过 `max_position_embeddings` 的序列（通过偏移 `past_key_values_length`）。

**c. 嵌入输入的兼容性**
- **直接处理嵌入**：当输入为预计算嵌入时（如跨模型迁移），仍能生成合理的位置 ID，提升模块灵活性。

#### **4. 与 BERT 的对比总结**
| **特性**               | **BertEmbeddings**                  | **RobertaEmbeddings**                  |
|------------------------|--------------------------------------|-----------------------------------------|
| **位置嵌入填充处理**    | 无显式处理                          | 指定 `padding_idx`，填充位固定编码       |
| **位置 ID 生成**        | 预注册缓冲区截取                    | 动态计算，区分填充位置                   |
| **长序列支持**          | 受限于 `max_position_embeddings`    | 通过 `past_key_values_length` 扩展       |
| **输入嵌入适配**        | 无特殊处理                          | 生成连续位置 ID，跳过 `padding_idx`      |
| **预训练任务适配**      | 适配 NSP + MLM                      | 专注 MLM，优化长序列处理                 |

#### **5. 示例场景**
**输入处理**
```python
# 输入序列含填充
input_ids = [[1, 2, 0, 0]]  # 假设 pad_token_id=0

# BERT 位置 IDs
position_ids = [[0, 1, 2, 3]]  # 填充位置仍计算

# RoBERTa 位置 IDs
position_ids = [[1, 2, 0, 0]]  # 填充位置设为 padding_idx=0
```

**效果分析**
- **BERT**：填充位置参与位置编码计算，可能引入噪声。
- **RoBERTa**：填充位置使用独立编码，减少无效位置对模型的影响。

#### **6. 总结**
`RobertaEmbeddings` 通过 **动态位置 ID 生成** 和 **显式填充索引处理**，解决了 BERT 在处理填充符和长序列时的局限性。这些改进使 RoBERTa 在以下方面表现更优：
1. **填充鲁棒性**：精准隔离填充符的影响。
2. **长序列兼容性**：支持超过预设最大长度的序列。
3. **训练效率**：减少无效位置的计算开销。


### 1.3 AlbertEmbeddings

#### **1. 初始化方法 (`__init__`)**
**嵌入层定义**
```python
self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
```
- **关键参数**：
  - `embedding_size`：ALBERT 的嵌入维度（通常小于隐藏层维度 `hidden_size`，以节省参数）。
  - `max_position_embeddings`：最大序列长度（如 512）。
  - `type_vocab_size`：段落类型数（通常为 2，区分句子 A/B）。

#### **2. 前向传播 (`forward`)**
**输入参数**
- **`input_ids`**：形状 `(batch_size, seq_len)` 的 Token ID 序列。
- **`token_type_ids`**：段落类型 ID（如 `0` 表示句子 A，`1` 表示句子 B）。
- **`position_ids`**：显式指定位置 ID，支持自定义位置编码。
- **`past_key_values_length`**：历史缓存长度（用于生成任务的位置偏移）。

**处理流程**
1. **确定输入形状**：
   ```python
   if input_ids is not None:
       input_shape = input_ids.size()  # (batch_size, seq_len)
   else:
       input_shape = inputs_embeds.size()[:-1]  # (batch_size, seq_len)
   seq_length = input_shape[1]
   ```

2. **生成位置 ID**：
   ```python
   if position_ids is None:
       position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
   ```
   - **动态调整**：根据 `past_key_values_length` 截取位置 ID，适配自回归生成任务。

3. **生成段落类型 ID**：
   ```python
   if token_type_ids is None:
       # 从缓冲区扩展全零张量 (batch_size, seq_len)
       token_type_ids = buffered_token_type_ids_expanded
   ```

4. **计算各嵌入分量**：
   - **词嵌入**：`inputs_embeds = self.word_embeddings(input_ids)`
   - **段落类型嵌入**：`token_type_embeddings = self.token_type_embeddings(token_type_ids)`
   - **位置嵌入**（若为绝对编码）：
     ```python
     if self.position_embedding_type == "absolute":
         position_embeddings = self.position_embeddings(position_ids)
     ```

5. **融合嵌入与后处理**：
   ```python
   embeddings = inputs_embeds + token_type_embeddings + position_embeddings
   embeddings = self.LayerNorm(embeddings)
   embeddings = self.dropout(embeddings)
   ```

#### **3. 关键设计解析**
**a. 参数高效性**
- **嵌入维度压缩**：ALBERT 通过 **因式分解嵌入参数**，将 `embedding_size`（如 128）与后续层的 `hidden_size`（如 768）解耦，大幅减少参数量（从 `vocab_size * hidden_size` 降至 `vocab_size * embedding_size + embedding_size * hidden_size`）。

**b. 位置编码兼容性**
- **绝对位置编码**：默认使用可学习的位置嵌入，与 BERT 一致。
- **相对位置编码**：通过 `position_embedding_type` 可扩展支持，但当前代码仅实现绝对编码。

**c. 动态位置偏移**
- **`past_key_values_length`**：在生成任务中，已生成的 Token 位置 ID 从历史长度开始递增，避免重复计算。

**d. 段落类型默认处理**
- **全零默认值**：当输入为单句或未提供 `token_type_ids` 时，自动填充为 0，简化接口调用。

#### **4. 与 BertEmbeddings 的对比**
| **特性**               | **AlbertEmbeddings**                  | **BertEmbeddings**                  |
|------------------------|---------------------------------------|--------------------------------------|
| **嵌入维度**           | `embedding_size`（较小）              | `hidden_size`（与隐藏层一致）        |
| **参数共享**           | 跨层共享嵌入参数                       | 独立嵌入参数                         |
| **位置编码类型**       | 仅实现绝对编码（可扩展）                | 支持绝对与相对编码                   |
| **缓冲区注册**         | 显式注册 `position_ids`/`token_type_ids` | 类似，但依赖配置参数                 |

#### **5. 示例计算流程**
**输入**：
- `input_ids`: `[[101, 2054, 2003, 102]]` （假设 `[CLS]`, `"hello"`, `"world"`, `[SEP]`）
- `token_type_ids`: `[[0, 0, 0, 0]]` （单句）

**处理步骤**：
1. **词嵌入**：将每个 ID 映射为 128 维向量。
2. **位置嵌入**：位置 0~3 的 128 维向量。
3. **段落嵌入**：全 0 的 128 维向量。
4. **相加**：`embeddings = word_emb + position_emb + token_type_emb`
5. **LayerNorm**：归一化处理。
6. **Dropout**：随机置零部分神经元。

**输出形状**：`(1, 4, 128)`


### 1.4 ElectraEmbeddings

#### **1. 初始化方法 (`__init__`)**
**嵌入层定义**
```python
self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
```
- **关键参数**：
  - `embedding_size`：嵌入维度（通常小于隐藏层维度 `hidden_size`，与 ALBERT 类似）。
  - `max_position_embeddings`：最大序列长度（如 512）。
  - `type_vocab_size`：段落类型数（通常为 2）。

#### **2. 前向传播 (`forward`)**
**输入参数**
- **`input_ids`**：形状 `(batch_size, seq_len)` 的 Token ID 序列。
- **`token_type_ids`**：段落类型 ID（如 `0` 表示句子 A，`1` 表示句子 B）。
- **`position_ids`**：显式指定位置 ID，支持自定义位置编码。
- **`past_key_values_length`**：历史缓存长度（用于生成任务的位置偏移）。

**处理流程**
1. **确定输入形状**：
   ```python
   if input_ids is not None:
       input_shape = input_ids.size()  # (batch_size, seq_len)
   else:
       input_shape = inputs_embeds.size()[:-1]
   seq_length = input_shape[1]
   ```

2. **生成位置 ID**：
   ```python
   if position_ids is None:
       position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
   ```
   - **动态调整**：根据 `past_key_values_length` 截取位置 ID，适配自回归生成任务。

3. **生成段落类型 ID**：
   ```python
   if token_type_ids is None:
       # 从缓冲区扩展全零张量 (batch_size, seq_len)
       token_type_ids = buffered_token_type_ids_expanded
   ```

4. **计算各嵌入分量**：
   - **词嵌入**：`inputs_embeds = self.word_embeddings(input_ids)`
   - **段落类型嵌入**：`token_type_embeddings = self.token_type_embeddings(token_type_ids)`
   - **位置嵌入**（若为绝对编码）：
     ```python
     if self.position_embedding_type == "absolute":
         position_embeddings = self.position_embeddings(position_ids)
     ```

5. **融合嵌入与后处理**：
   ```python
   embeddings = inputs_embeds + token_type_embeddings + position_embeddings
   embeddings = self.LayerNorm(embeddings)
   embeddings = self.dropout(embeddings)
   ```

#### **3. 关键设计解析**
**a. 参数高效性**
- **嵌入维度压缩**：ELECTRA 与 ALBERT 类似，通过 **因式分解嵌入参数**，将 `embedding_size`（如 128）与后续层的 `hidden_size`（如 768）解耦，显著减少参数量（从 `vocab_size * hidden_size` 降至 `vocab_size * embedding_size + embedding_size * hidden_size`）。

**b. 位置编码兼容性**
- **绝对位置编码**：默认使用可学习的位置嵌入，与 BERT 一致。
- **相对位置编码**：通过 `position_embedding_type` 可扩展支持，但当前代码仅实现绝对编码。

**c. 动态位置偏移**
- **`past_key_values_length`**：在生成任务中，已生成的 Token 位置 ID 从历史长度开始递增，避免重复计算。

#### **4. 与 BERT/ALBERT/RoBERTa 的对比**
| **特性**               | **ElectraEmbeddings**              | **BERT**               | **ALBERT**              | **RoBERTa**             |
|------------------------|------------------------------------|------------------------|-------------------------|-------------------------|
| **嵌入维度**           | `embedding_size`（较小）           | `hidden_size`          | `embedding_size`        | `hidden_size`           |
| **参数共享**           | 与判别器共享生成器嵌入              | 无                     | 跨层共享嵌入参数         | 无                      |
| **位置编码类型**       | 绝对编码（可扩展）                 | 绝对编码               | 绝对编码                | 动态生成位置 ID          |
| **预训练任务适配**     | 替换令牌检测（RTD）                | MLM + NSP              | MLM + SOP               | MLM（无 NSP）           |

#### **5. ELECTRA 嵌入层的独特优势**
**a. 生成器-判别器参数共享**
ELECTRA 的生成器和判别器 **共享词嵌入矩阵**（仅在隐藏层维度不同），这使得：
- **训练效率提升**：减少总参数量，加速收敛。
- **知识迁移**：生成器的语义信息直接传递给判别器，提升替换令牌检测任务的准确性。

**b. 因式分解嵌入的兼容性**
- **参数效率**：小嵌入维度减少内存占用，尤其适合大规模词表场景。
- **模型轻量化**：在保持性能的同时降低计算开销，适配资源受限环境。

**c. 替换检测任务的适应性**
- **细粒度语义捕捉**：因式分解迫使模型在低维嵌入中编码更多语义信息，有助于判别器识别生成器替换的 Token。

#### **6. 示例计算流程**
**输入**：
- `input_ids`: `[[1, 2, 0, 0]]` （假设 `pad_token_id=0`）
- `token_type_ids`: `[[0, 0, 0, 0]]` （单句）

**处理步骤**：
1. **词嵌入**：将每个 ID 映射为 128 维向量（填充符对应零向量）。
2. **位置嵌入**：位置 0~3 的 128 维向量。
3. **段落嵌入**：全 0 的 128 维向量。
4. **相加**：`embeddings = word_emb + position_emb + token_type_emb`
5. **LayerNorm**：归一化处理。
6. **Dropout**：随机置零部分神经元。

**输出形状**：`(1, 4, 128)`

## 2. Tokenizer

### 2.1 BertTokenizer

#### **1. 类定义与继承关系**
```python
class BertTokenizer(PreTrainedTokenizer):
```
- **基类**：`PreTrainedTokenizer`，Hugging Face 所有预训练分词器的抽象基类，提供标准化接口（如 `encode()`、`decode()`）。
- **分词类型**：基于 **WordPiece** 算法，支持子词切分。

#### **2. 初始化参数解析**
**关键参数**
| **参数名**               | **默认值**     | **作用**                                                                 |
|--------------------------|---------------|-------------------------------------------------------------------------|
| `vocab_file`             | 必填           | WordPiece 词表文件路径（每行一个 Token）                                 |
| `do_lower_case`          | `True`        | 是否将输入文本转为小写                                                   |
| `do_basic_tokenize`      | `True`        | 是否在 WordPiece 前进行基础分词（按空格、标点切分）                       |
| `never_split`            | `None`        | 指定永不切分的 Token 集合（如特殊标记）                                   |
| `unk_token`              | `"[UNK]"`     | 未知 Token 的替换符                                                      |
| `sep_token`              | `"[SEP]"`     | 分隔符，用于分割句子对或序列结束                                          |
| `pad_token`              | `"[PAD]"`     | 填充符，用于批次处理不同长度序列                                          |
| `cls_token`              | `"[CLS]"`     | 分类符，位于序列开头用于汇总信息                                          |
| `mask_token`             | `"[MASK]"`    | 掩码符，用于掩码语言模型训练                                              |
| `tokenize_chinese_chars` | `True`        | 是否将中文字符逐个切分（针对中文优化，日语需关闭）                        |
| `strip_accents`          | `None`        | 是否去除重音符号（如 `é` → `e`），默认与 `do_lower_case` 一致             |
| `clean_up_tokenization_spaces` | `True` | 解码时是否清理多余空格（如合并子词产生的 `" ##"`）                         |

#### **3. 核心组件初始化**
**a. 词表加载**
```python
self.vocab = load_vocab(vocab_file)  # 加载词表：token → id
self.ids_to_tokens = OrderedDict(...)  # 反向映射：id → token
```

**b. 分词器实例化**
- **基础分词器**（可选）：
  ```python
  if do_basic_tokenize:
      self.basic_tokenizer = BasicTokenizer(...)  # 处理空格、标点、中文等
  ```
- **WordPiece 分词器**：
  ```python
  self.wordpiece_tokenizer = WordpieceTokenizer(...)  # 子词切分
  ```

#### **4. 分词流程 (`_tokenize`)**
**步骤分解**
1. **基础分词**（若启用）：
   - 按空格、标点分割文本，处理中文逐字切分。
   - 保留 `never_split` 指定的 Token（如特殊标记）不被切分。
2. **WordPiece 切分**：
   - 将基础分词后的每个 Token 进一步拆分为子词。
   - 子词前缀添加 `##` 表示非起始位置（如 `"word"` → `["w", "##ord"]`）。

**代码逻辑**
```python
def _tokenize(self, text):
    split_tokens = []
    if self.do_basic_tokenize:
        # 基础分词
        for token in self.basic_tokenizer.tokenize(...):
            if token in never_split:
                split_tokens.append(token)
            else:
                # WordPiece 切分
                split_tokens += self.wordpiece_tokenizer.tokenize(token)
    else:
        # 直接 WordPiece 切分
        split_tokens = self.wordpiece_tokenizer.tokenize(text)
    return split_tokens
```

#### **5. 关键方法解析**
**a. 特殊标记插入 (`build_inputs_with_special_tokens`)**
- **单序列**：`[CLS] A [SEP]`
- **双序列**：`[CLS] A [SEP] B [SEP]`
```python
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    if token_ids_1 is None:
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
    return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]
```

**b. Token类型ID生成 (`create_token_type_ids_from_sequences`)**
- **单序列**：全 `0`
- **双序列**：`0` 表示第一句，`1` 表示第二句
```python
def create_token_type_ids_from_sequences(...):
    if token_ids_1 is None:
        return [0] * (len(cls + token_ids_0 + sep))
    return [0]*(len(cls + token_ids_0 + sep)) + [1]*(len(token_ids_1 + sep))
```

**c. 特殊标记掩码 (`get_special_tokens_mask`)**
- **掩码格式**：`1` 表示特殊标记，`0` 表示普通 Token
```python
def get_special_tokens_mask(...):
    if token_ids_1:
        return [1, 0,..., 1, 0,..., 1]
    return [1, 0,..., 1]
```

#### **6. 编码与解码流程**
**a. 编码（文本 → ID）**
1. **分词**：`text → tokens`
2. **转ID**：`tokens → token_ids`
3. **添加特殊标记**：插入 `[CLS]`、`[SEP]`
4. **生成掩码**：`attention_mask`、`token_type_ids`

**b. 解码（ID → 文本）**
```python
def convert_tokens_to_string(self, tokens):
    return " ".join(tokens).replace(" ##", "").strip()
```
- **处理子词**：合并 `"hello"` 和 `"##world"` 为 `"helloworld"`。

#### **7. 词表管理**
**保存词表 (`save_vocabulary`)**
```python
def save_vocabulary(...):
    with open(vocab_file, "w") as writer:
        for token, index in sorted(vocab.items()):
            writer.write(token + "\n")
```
- **按ID排序**：确保保存顺序与加载时一致。
- **索引检查**：若索引不连续发出警告。

#### **8. 设计亮点总结**
- **灵活的分词流程**：支持基础分词与纯 WordPiece 模式，适应不同语言需求。
- **特殊标记自动化**：自动处理 `[CLS]`、`[SEP]` 插入及类型ID生成。
- **高效子词合并**：通过替换 `" ##"` 快速还原原始词片段。
- **多语言支持**：通过 `tokenize_chinese_chars` 和 `strip_accents` 适配不同语言特性。

#### **9. 使用示例**
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
text = "Hello world! How are you?"

# 编码
inputs = tokenizer(text, return_tensors="pt")
# inputs包含:
# - input_ids: [101, 7592, 2088, 999, 2129, 2024, 2017, 1029, 102]
# - token_type_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0]
# - attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1]

# 解码
decoded = tokenizer.decode(inputs["input_ids"][0])
# "hello world! how are you?"
```

### 2.2 BasicTokenizer

#### **1. 功能概述**
`BasicTokenizer` 是 BERT 分词流程中的 **基础预处理模块**，负责以下任务：
1. **文本清洗**：去除无效字符、控制字符，标准化空白符。
2. **中文字符处理**：在中文字符周围添加空格以便后续切分。
3. **大小写转换**：可选将文本转为小写。
4. **重音去除**：可选去除重音符号（如 `é → e`）。
5. **标点分割**：将标点符号与相邻字符切分开。
6. **空格分词**：按空白符初步分割文本为词片段。

#### **2. 核心方法解析**
**`tokenize(text, never_split)`**
**输入**：原始文本字符串  
**输出**：初步分词的 Token 列表  
**流程**：
1. **文本清洗**：
   ```python
   text = self._clean_text(text)  # 去除无效字符、控制符，标准化空格
   ```
2. **中文字符处理**（可选）：
   ```python
   if self.tokenize_chinese_chars:
       text = self._tokenize_chinese_chars(text)  # 中文字符周围加空格
   ```
3. **Unicode 标准化**：
   ```python
   text = unicodedata.normalize("NFC", text)  # 合并兼容字符
   ```
4. **空格初步分词**：
   ```python
   orig_tokens = whitespace_tokenize(text)  # 按空白符分割
   ```
5. **逐词处理**：
   - **大小写转换**（若启用）
   - **去除重音**（若启用）
   - **按标点分割**（若启用）
   ```python
   for token in orig_tokens:
       if token not in never_split:
           # 大小写转换与去重音
           token = self._process_case_and_accents(token)
           # 按标点分割
           split_tokens.extend(self._run_split_on_punc(token, never_split))
       else:
           split_tokens.append(token)
   ```
6. **合并结果**：
   ```python
   output_tokens = whitespace_tokenize(" ".join(split_tokens))  # 重新按空格分割
   ```

#### **3. 关键子方法详解**
**`_clean_text(text)`**
- **功能**：清理无效字符与控制符，标准化空白符。
- **处理逻辑**：
  - 过滤 Unicode 控制字符（`_is_control`）。
  - 替换无效字符（`U+FFFD` 替换符）。
  - 将连续空白符转换为单个空格。

**`_tokenize_chinese_chars(text)`**
- **功能**：在中文字符周围插入空格，确保后续分词独立处理。
- **示例**：
  ```python
  Input: "你好世界" → Output: " 你  好  世  界 "
  ```

**`_run_strip_accents(text)`**
- **实现**：
  ```python
  text = unicodedata.normalize("NFD", text)  # 分解重音
  output = [c for c in text if unicodedata.category(c) != 'Mn']  # 过滤组合符号
  ```
- **效果**：`"café" → "cafe"`

**`_run_split_on_punc(text, never_split)`**
- **功能**：将标点与相邻字符切分。
- **示例**：
  ```python
  Input: "hello,world!" → Output: ["hello", ",", "world", "!"]
  ```
- **逻辑**：遍历字符，遇到标点时创建新 Token。

**`_is_chinese_char(cp)`**
- **功能**：判断 Unicode 码点是否为中文字符（CJK 统一表意文字区）。
- **码点范围**：
  - 基本区：`0x4E00-0x9FFF`
  - 扩展A-G区、兼容区等。

#### **4. 参数控制与设计选择**
| **参数**               | **默认值** | **影响**                                                                 |
|------------------------|------------|-------------------------------------------------------------------------|
| `do_lower_case`         | `True`     | 统一为小写，适配英文模型，但可能损害多语言大小写敏感性。                 |
| `tokenize_chinese_chars`| `True`     | 中文字符独立切分，优化中文处理，但日语可能需要禁用。                     |
| `strip_accents`         | `None`     | 默认与 `do_lower_case` 同步，去除变音符号可能影响法语、西班牙语等语言。 |
| `do_split_on_punc`      | `True`     | 分割标点，有助于后续 WordPiece 处理，但可能破坏某些缩略词（如 `Mr.`）。 |

#### **5. 处理流程示例**
**输入文本**：`"Hello! 你好，世界！ Don't worry."`

**处理步骤**：
1. **清洗**：无变化。
2. **中文字符处理** → `"Hello!  你  好 ， 世  界  ！ Don't worry."`
3. **Unicode 标准化**：无变化。
4. **空格分词** → `["Hello!", "你", "好", "，", "世", "界", "！", "Don't", "worry."]`
5. **逐词处理**（假设 `do_lower_case=True`）：
   - `"Hello!"` → 转小写 → `"hello!"` → 按标点分割 → `["hello", "!"]`
   - `"Don't"` → 转小写 → `"don't"` → 按标点分割 → `["don", "'", "t"]`
6. **合并结果** → `["hello", "!", "你", "好", "，", "世", "界", "！", "don", "'", "t", "worry", "."]`

#### **6. 注意事项与局限性**
- **中文与日语冲突**：中文字符处理可能误伤日语汉字，需手动关闭 `tokenize_chinese_chars`。
- **重音去除风险**：可能改变词义（如法语 `"péché"`（罪）与 `"peche"`（钓鱼））。
- **标点分割问题**：无法处理复杂缩写（如 `"U.S.A."` 应保持整体，但会被分割为 `["U", ".", "S", ".", "A", "."]`）。

#### **7. 总结**
`BasicTokenizer` 是 BERT 分词流程的 **前置处理器**，其核心贡献在于：
1. **文本标准化**：统一字符表示，清理噪声。
2. **语言适配**：通过参数控制中文字符、重音处理。
3. **结构化解构**：将标点、空格等显式分离，为后续 WordPiece 分词提供清晰输入。
通过灵活的配置选项，该模块能够适应多语言场景，同时为深度学习模型提供稳定、规范化的文本输入。

### 2.3 WordpieceTokenizer

#### **1. 类定义与初始化**
```python
class WordpieceTokenizer:
    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab        # WordPiece词表（子词到ID的映射）
        self.unk_token = unk_token  # 未知标记（如"[UNK]"）
        self.max_input_chars_per_word = max_input_chars_per_word  # 单词最大长度阈值
```
- **关键参数**：
  - **`max_input_chars_per_word`**：超过该长度的单词直接标记为 `unk_token`，避免处理过长输入。


#### **2. 分词流程 (`tokenize`)**
**输入与预处理**
- **输入**：经过 `BasicTokenizer` 处理后的词（如 `["hello", "world"]`）。
- **处理逻辑**：
  1. **按空格分割**：`whitespace_tokenize(text)` 确保每个词独立处理。
  2. **逐词处理**：对每个词进行子词切分。

**单个词处理步骤**
1. **长度检查**：
   ```python
   if len(chars) > self.max_input_chars_per_word:
       output_tokens.append(self.unk_token)
       continue
   ```
   - 超过阈值直接标记为未知。

2. **贪婪最长匹配**：
   - **初始化**：`start = 0`，从词首开始。
   - **循环查找子词**：
     ```python
     while start < len(chars):
         end = len(chars)  # 从词尾向前搜索
         while start < end:
             substr = chars[start:end]
             if start > 0:
                 substr = "##" + substr  # 非起始子词添加前缀
             if substr in self.vocab:
                 cur_substr = substr
                 break
             end -= 1  # 缩短子词长度
         if cur_substr is None:
             is_bad = True  # 未找到有效子词
             break
         sub_tokens.append(cur_substr)
         start = end  # 移动起始位置
     ```
   - **示例**：  
     `"unaffable"` → `["un", "##aff", "##able"]`

3. **结果处理**：
   - **成功拆分**：将子词列表加入输出。
   - **失败处理**：添加 `unk_token`。

#### **3. 关键算法解析**
**贪婪最长匹配策略**
- **方向**：从右向左扫描，优先匹配最长可能的子词。
- **前缀处理**：非起始子词添加 `##` 标记（如 `##able`）。
- **终止条件**：无法找到有效子词时，标记整个词为未知。

**复杂度优化**
- **最大长度限制**：避免处理超长词，减少计算开销。
- **快速失败**：一旦发现无法切分，立即跳出循环。

#### **4. 示例演算**
**输入词**：`"unaffable"`  
**词表包含**：`["un", "##aff", "##able"]`  
**处理步骤**：
1. `start=0`, 查找最长有效子词：
   - `"unaffable"`（不在词表）
   - ... 缩短至 `"un"`（在词表）
2. `sub_tokens = ["un"]`, `start=2`
3. 剩余部分 `"affable"`，添加 `##` 前缀：
   - `"##affable"` → 不在词表
   - 缩短至 `"##aff"`（在词表）
4. `sub_tokens = ["un", "##aff"]`, `start=5`
5. 剩余部分 `"able"`，添加 `##`：
   - `"##able"`（在词表）
6. **最终输出**：`["un", "##aff", "##able"]`

#### **5. 设计局限与应对**
| **局限**                  | **应对策略**                               |
|---------------------------|-------------------------------------------|
| **未登录词处理**           | 标记为 `unk_token`，依赖词表覆盖度         |
| **贪婪匹配局部最优**       | 无法保证全局最优，但计算效率高             |
| **前缀标记依赖**           | 需词表中子词明确标记起始与非起始位置       |

#### **6. 与 BPE 算法的对比**
| **特性**         | **WordPiece**                     | **BPE**                          |
|------------------|-----------------------------------|----------------------------------|
| **合并策略**     | 基于概率（如语言模型）选择合并对   | 基于频率选择合并对                |
| **前缀标记**     | 使用 `##` 表示非起始子词           | 无特殊标记                       |
| **分词方向**     | 从右向左最长匹配                   | 从左到右逐步合并                  |
| **应用场景**     | BERT、ALBERT                     | GPT、RoBERTa                    |

#### **7. 总结**
`WordpieceTokenizer` 是 BERT 分词流程的 **核心子词处理器**，其核心机制包括：
1. **贪婪最长匹配**：高效拆分词为子词单元。
2. **前缀标记系统**：区分子词位置（起始或中间）。
3. **异常处理**：超长词与未登录词统一标记为未知。
该设计平衡了计算效率与分词准确性，成为预训练语言模型处理未登录词的关键组件。


### 2.4 RobertaTokenizer

#### **1. 类定义与继承关系**
```python
class RobertaTokenizer(PreTrainedTokenizer):
```
- **基类**：`PreTrainedTokenizer`，提供标准化的分词器接口（如编码、解码、保存词表等）。
- **分词算法**：基于 **字节级 BPE（Byte-Pair-Encoding）**，与 GPT-2 相同，但针对 RoBERTa 任务优化。

#### **2. 关键参数解析**
| **参数**               | **默认值**     | **作用**                                                                 |
|------------------------|---------------|-------------------------------------------------------------------------|
| `vocab_file`           | 必填           | BPE 词表文件（JSON 格式，子词到 ID 的映射）                              |
| `merges_file`          | 必填           | BPE 合并规则文件（记录字符对的合并顺序）                                  |
| `errors`               | `"replace"`    | 字节解码错误处理策略（如替换无效字符为 `�`）                              |
| `bos_token`            | `"<s>"`        | 序列开始标记，但实际分类任务使用 `cls_token`（即 `<s>`）                  |
| `eos_token`            | `"</s>"`       | 序列结束标记，同时作为分隔符（`sep_token`）                               |
| `add_prefix_space`     | `False`        | 是否在输入文本前添加空格，解决首单词分词不一致问题                        |

#### **3. 初始化流程 (`__init__`)**
**a. 特殊 Token 处理**
- **掩码 Token 配置**：
  ```python
  mask_token = AddedToken(..., lstrip=True)  # 保留左侧空格，避免与单词粘连
  ```
  - **设计意图**：确保掩码标记像普通单词一样处理，兼容预训练模式。

**b. 加载词表与合并规则**
```python
with open(vocab_file) as f:
    self.encoder = json.load(f)  # 子词到 ID 的映射
self.decoder = {v: k for k, v in self.encoder.items()}  # ID 到子词的反向映射

with open(merges_file) as f:
    bpe_merges = f.read().split("\n")[1:-1]  # 跳过版本行和空行
self.bpe_ranks = dict(zip([tuple(merge.split()) for merge in bpe_merges], range(len(bpe_merges))))  # 合并规则优先级
```

**c. BPE 正则表达式**
```python
self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```
- **功能**：将文本拆分为待处理的基础单元（如完整单词、数字、标点、空格等）。
- **示例**：  
  `"Don't worry!"` → `["Don", "'t", " worry", "!"]`

#### **4. 核心方法解析**
**a. BPE 分词 (`bpe`)**
```python
def bpe(self, token):
    if token in cache:  # 缓存加速
        return cache[token]
    word = tuple(token)  # 字符级拆分
    pairs = get_pairs(word)  # 获取相邻字符对
    
    while True:
        bigram = min(pairs, key=lambda p: bpe_ranks.get(p, float('inf')))  # 选择优先级最高的合并对
        if bigram not in bpe_ranks:  # 无更多可合并项
            break
        # 合并字符对并更新 word
        # ...
    return " ".join(word)  # 返回 BPE 结果
```
- **合并策略**：贪心选择词表中优先级最高（出现最频繁）的字符对进行合并，直到无法合并。

**b. 文本分词 (`_tokenize`)**
```python
def _tokenize(self, text):
    bpe_tokens = []
    for token in re.findall(self.pat, text):
        # 将 token 转为 BPE 处理后的子词
        token_bytes = token.encode("utf-8")
        token_unicode = "".join(self.byte_encoder[b] for b in token_bytes)  # 字节到 Unicode 映射
        bpe_result = self.bpe(token_unicode).split(" ")  # BPE 处理并拆分
        bpe_tokens.extend(bpe_result)
    return bpe_tokens
```
- **字节编码**：将每个字符映射为可打印 Unicode 字符，避免控制字符干扰 BPE。
- **示例**：  
  `"hello"` → 字节 `[104, 101, 108, 108, 111]` → Unicode `"h e l l o"` → BPE 合并为 `["hello"]` 或更小子词。

#### **5. 特殊输入处理**
**a. 前缀空格处理 (`prepare_for_tokenization`)**
```python
def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
    if 需要添加前缀空格:
        text = " " + text  # 确保首单词正确分词
    return text
```
- **场景**：当输入文本未分割成单词（`is_split_into_words=False`）且 `add_prefix_space=True` 时，自动添加起始空格。

**b. 特殊标记添加 (`build_inputs_with_special_tokens`)**
```python
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    if token_ids_1 is None:
        return [cls_token_id] + token_ids_0 + [sep_token_id]
    return [cls_token_id] + token_ids_0 + [sep_token_id, sep_token_id] + token_ids_1 + [sep_token_id]
```
- **格式**：
  - 单序列：`<s> A </s>`
  - 双序列：`<s> A </s> </s> B </s>`


#### **6. 编码解码流程**
**a. 编码（文本 → Token ID）**
1. **预处理**：添加前缀空格（若需）、正则拆分基础单元。
2. **BPE 分词**：将每个基础单元拆分为子词。
3. **转换为 ID**：查词表映射，未知词用 `unk_token` 替代。
4. **添加特殊标记**：插入 `<s>` 和 `</s>`。

**b. 解码（Token ID → 文本）**
```python
def convert_tokens_to_string(self, tokens):
    text = "".join(tokens)
    # 将 Unicode 字符映射回字节
    byte_array = bytearray([self.byte_decoder[c] for c in text])
    return byte_array.decode("utf-8", errors=self.errors)
```
- **处理步骤**：合并子词、转换字节、解码为 UTF-8。

#### **7. 与 BERT 分词器的关键差异**
| **特征**               | **RobertaTokenizer**              | **BertTokenizer**               |
|------------------------|-----------------------------------|----------------------------------|
| **分词算法**           | 字节级 BPE                        | WordPiece                        |
| **空格处理**           | 空格视为单词部分，影响分词结果     | 通过基础分词器处理空格            |
| **特殊标记**           | `<s>`, `</s>`                     | `[CLS]`, `[SEP]`                 |
| **Token 类型 ID**      | 未使用（全零）                     | 用于区分句子 A/B                  |
| **预训练任务适配**     | 仅 MLM，无 NSP                    | MLM + NSP                        |

#### **8. 设计亮点总结**
- **字节级 BPE**：直接处理原始字节，避免 Unicode 编码问题，提升多语言支持。
- **空格敏感处理**：通过 `add_prefix_space` 和正则匹配，确保单词分词的上下文一致性。
- **无 Token 类型 ID**：简化输入结构，适应 RoBERTa 无 NSP 任务的预训练设定。
- **高效缓存机制**：减少重复计算，加速分词过程。

#### **9. 使用示例**
```python
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
text = "Hello world! How are you?"

# 编码
inputs = tokenizer(text, return_tensors="pt")
# input_ids: [0, 31414, 232, 328, 2]

# 解码
decoded = tokenizer.decode(inputs["input_ids"][0])
# "<s>Hello world! How are you?</s>"
```

```python
>>> from transformers import RobertaTokenizer

>>> tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]
```

### 2.5 AlbertTokenizer

#### **1. 类定义与继承关系**
```python
@export(backends=("sentencepiece",))
class AlbertTokenizer(PreTrainedTokenizer):
```
- **基类**：`PreTrainedTokenizer`，Hugging Face 标准预训练分词器接口。
- **依赖后端**：`sentencepiece`，表明其基于 Google 的 SentencePiece 库实现子词分词。


#### **2. 初始化参数解析**
**关键参数**
| **参数名**           | **默认值** | **作用**                                                                 |
|----------------------|------------|-------------------------------------------------------------------------|
| `vocab_file`         | 必填       | SentencePiece 模型文件路径（`.spm` 后缀）                               |
| `do_lower_case`      | `True`     | 是否将输入文本转为小写                                                  |
| `remove_space`       | `True`     | 是否清除文本首尾空格并压缩连续空格                                      |
| `keep_accents`       | `False`    | 是否保留重音符号（如 `é` → `e`）                                        |
| `sp_model_kwargs`    | `None`     | 传递给 SentencePiece 的配置，如启用子词正则化 (`enable_sampling`)       |


#### **3. 核心组件初始化**
**a. SentencePiece 模型加载**
```python
self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
self.sp_model.Load(vocab_file)  # 加载.spm模型文件
```
- **功能**：将文本分割为 SentencePiece 子词单元。

**b. 特殊 Token 配置**
- **特殊标记**：`[CLS]`、`[SEP]`、`<unk>`、`<pad>`、`[MASK]`，与 BERT 类似。
- **掩码 Token 处理**：
  ```python
  mask_token = AddedToken(..., normalized=False)  # 保留原始形式，避免被预处理影响
  ```

#### **4. 预处理与分词流程**
**a. 文本预处理 (`preprocess_text`)**
1. **空格处理**：
   ```python
   if self.remove_space: 
       outputs = " ".join(inputs.strip().split())  # 压缩连续空格
   ```
2. **引号替换**：` `` → "`，`'' → "`（统一引号格式）。
3. **重音处理**：
   ```python
   if not self.keep_accents:
       outputs = unicodedata.normalize("NFKD", outputs)  # 分解重音
       outputs = "".join([c for c in outputs if not unicodedata.combining(c)])  # 去除组合符号
   ```
4. **小写转换**（若启用）。

**b. 子词分词 (`_tokenize`)**
```python
text = self.preprocess_text(text)  # 预处理
pieces = self.sp_model.encode(text, out_type=str)  # SentencePiece 分词
```
- **特殊处理**：修正数字与逗号的分割问题（如 `"9,9"` → `["▁9", ",", "9"]`）。
  ```python
  for piece in pieces:
      if 以逗号结尾且前一位是数字:
          手动拆分并调整子词前缀
  ```

#### **5. 编码与解码方法**
**a. Token ↔ ID 转换**
- **Token → ID**：`self.sp_model.PieceToId(token)`
- **ID → Token**：`self.sp_model.IdToPiece(index)`

**b. 解码为字符串 (`convert_tokens_to_string`)**
```python
for token in tokens:
    if token 是特殊标记:
        直接拼接，避免 SentencePiece 解码干扰
    else:
        累积子词，调用 sp_model.decode() 转换
return out_string.strip()
```
- **示例**：子词 `["▁Hello", "##World"]` → `"Hello World"`

#### **6. 输入构造与特殊标记**
**a. 添加特殊标记 (`build_inputs_with_special_tokens`)**
- **单序列**：`[CLS] + tokens + [SEP]`
- **双序列**：`[CLS] + tokens_A + [SEP] + tokens_B + [SEP]`

**b. 生成掩码与段落 ID**
- **特殊标记掩码**：标记 `[CLS]` 和 `[SEP]` 的位置。
- **段落 ID**：第一句全 `0`，第二句全 `1`。

#### **7. 词表管理与序列化**
**a. 保存词表 (`save_vocabulary`)**
- **行为**：复制 `.spm` 文件或序列化 SentencePiece 模型二进制内容。
- **路径处理**：确保保存路径正确且不覆盖原文件。

**b. 序列化兼容性**
```python
def __getstate__(self):  # 序列化时排除 sp_model
def __setstate__(self, d):  # 反序列化时重新加载 sp_model
```
- **原因**：SentencePiece 对象无法直接序列化，需特殊处理。

#### **8. 与 BertTokenizer 的对比**
| **特性**           | **AlbertTokenizer**              | **BertTokenizer**               |
|--------------------|-----------------------------------|----------------------------------|
| **分词算法**       | SentencePiece (Unigram/LSTM)     | WordPiece (贪婪最长匹配)          |
| **子词前缀**       | `▁` 表示词首                     | `##` 表示非词首                   |
| **多语言支持**     | 更优（内置多语言分词）            | 依赖基础分词器配置                 |
| **预训练兼容性**   | 专为 ALBERT 设计                 | 专为 BERT 设计                    |
| **特殊标记处理**   | 类似 BERT，但掩码 Token 保留原始形式 | 掩码 Token 可能被预处理影响       |

#### **9. 设计亮点与注意事项**
1. **SentencePiece 优势**：
   - 支持直接从原始文本训练子词模型，无需预分词。
   - 内置 Unicode 处理，更适合多语言场景。
2. **预处理一致性**：
   - 通过 `preprocess_text` 确保训练与推理时文本处理一致。
3. **数字-标点处理**：
   - 手动修正 SentencePiece 对类似 `"9,9"` 的分词，提升下游任务准确性。
4. **特殊标记独立性**：
   - 在解码时单独处理特殊标记，避免 SentencePiece 的错误合并。

#### **10. 使用示例**
```python
from transformers import AlbertTokenizer

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
text = "Hello! 你好，世界！ Don't worry."

# 编码
inputs = tokenizer(text, return_tensors="pt")
# input_ids: [2, 437, 132, 2117, ..., 3] （含 [CLS] 和 [SEP]）

# 解码
decoded = tokenizer.decode(inputs["input_ids"][0])
# "hello! 你好，世界！ don't worry."
```

#### **11. 总结**
`AlbertTokenizer` 是 ALBERT 模型的核心组件之一，其设计特点包括：
1. **基于 SentencePiece**：灵活高效的子词分词，适配多语言和长尾词。
2. **精细预处理**：空格清理、重音处理、大小写统一，保障输入一致性。
3. **特殊标记安全**：独立处理特殊符号，避免分词干扰。
4. **兼容性设计**：继承 Hugging Face 标准接口，无缝接入预训练模型。
通过结合 SentencePiece 的强大能力与 ALBERT 的模型架构，该分词器在提升模型效率（参数共享）的同时，保持了处理复杂文本的鲁棒性。

### 2.6 ElectraTokenizer
同BertTokenizer。