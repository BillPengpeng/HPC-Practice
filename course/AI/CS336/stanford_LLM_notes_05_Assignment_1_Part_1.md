本文主要整理Assignment 1 (basics): Building a Transformer LM的主要内容。

## 1 - Assignment Overview

### 内容概括

您上传的两张图片均为课程作业说明材料，主要围绕“从零开始实现并训练Transformer语言模型”这一任务展开：

- **第一张图** 是作业概述（Assignment Overview），明确了需要**从零实现**的核心组件（如BPE分词器、Transformer模型、损失函数与优化器、训练循环等），规定了可使用的PyTorch组件范围（禁止使用大部分`torch.nn`和`torch.optim`中的高阶API），并列出了具体任务（如在TinyStories上训练模型、生成文本等）。
  
- **第二张图** 补充了作业的**实际操作细节**，包括：
  - **AI工具使用政策**：允许用ChatGPT等工具咨询底层编程或概念问题，但禁止直接解题，建议禁用IDE中的AI自动补全功能。
  - **代码结构说明**：提供了GitHub仓库地址，说明需在指定目录（`cs336_basics/`）内从零编写代码，并通过`adapters.py`对接测试文件。
  - **提交要求**：需提交PDF报告和代码压缩包至Gradescope，并通过GitHub PR提交至排行榜。
  - **数据集获取**：指定使用TinyStories和OpenWebText数据集，并提供了校内机器和本地下载的获取方式。

---

### 要点总结

1. **核心任务**  
   - 从零实现BPE分词器、Transformer语言模型、交叉熵损失、AdamW优化器及训练循环。
   - 在TinyStories和OpenWebText数据集上训练模型，并生成样本与评估困惑度。

2. **实现限制**  
   - 禁止使用`torch.nn`、`torch.nn.functional`、`torch.optim`中的大部分组件（仅允许`Parameter`、容器类如`Module`、`Optimizer`基类）。
   - 强调“从零实现”原则，不确定时需在Slack询问。

3. **代码与测试**  
   - 代码需在空目录（`cs336_basics/`）内独立编写，通过`adapters.py`接口对接测试文件（`test_*.py`）。
   - 禁止修改测试文件，仅通过适配器调用自定义代码。

4. **AI工具政策**  
   - 允许向LLM提问底层编程或概念问题，但禁止直接生成解决方案。
   - 强烈建议禁用IDE的AI自动补全功能（如Copilot），以深入理解内容。

5. **提交与评估**  
   - 提交材料：PDF报告（书面题答案）和代码压缩包（Gradescope）。
   - 排行榜提交：通过GitHub PR提交困惑度结果至指定仓库。

6. **数据集**  
   - 使用TinyStories和OpenWebText预处理数据集。
   - 校内学生可通过指定路径获取（`/data/`），本地学习需按`README.md`指令下载。

## 2 - Byte-Pair Encoding (BPE) Tokenizer

### 内容概况
该部分要求学生从零开始实现一个基于字节的BPE分词器，用于将任意Unicode字符串转换为字节序列，并通过BPE算法进行子词切分。该分词器将用于后续语言建模任务中，将文本字符串编码为整数标记序列。

---

### 要点总结
1.  **核心任务**  
    - 实现一个**字节级BPE分词器**（基于Sennrich et al., 2016和Wang et al., 2019的论文）。
    - 分词器需在字节序列上操作，支持任意Unicode文本的编码和解码。

2.  **技术细节**  
    - 输入处理：将字符串转换为字节序列（UTF-8编码）。
    - 训练过程：通过迭代合并最高频的字节对来构建词汇表。
    - 功能要求：支持文本编码（字符串→整数标记）和解码（标记→字符串）。

3.  **应用场景**  
    - 为后续语言模型（Transformer）提供预处理功能，将文本转化为模型可处理的整数标记序列。

4.  **实现约束**  
    - 需从零实现BPE算法，禁止直接调用现有分词器库（如Hugging Face Tokenizers）。

5.  **关联性**  
    - 属于作业的第一部分，为后续模型训练提供数据预处理基础。

---

### 补充说明
- **BPE优势**：通过子词切分平衡词汇表大小与未登录词（OOV）问题，适合多语言文本处理。
- **字节级设计**：避免Un字符编码问题，可处理任意文本（如特殊符号、罕见语言字符）。

## 2.1 The Unicode Standard

### 内容概况

本节解释了 Unicode 的核心功能：将字符映射到整数代码点，并提供了在 Python 中如何进行字符和code points之间转换的具体方法。

---

### 要点总结

1.  **核心概念**：
    *   **Unicode** 是一个文本编码标准。
    *   其核心是将字符（Characters）映射到整数代码点（Code Points）。

2.  **规模与范围**：
    *   以 Unicode 16.0 (2024年9月发布) 为例，该标准定义了 **154,998** 个字符，涵盖 **168** 种文字。

3.  **具体示例**：
    *   字符 “s” 的代码点是 **115** (十六进制表示为 **U+0073**)。
    *   字符 “牛” 的代码点是 **29275**。

4.  **编程应用 (Python)**：
    *   使用 `ord()` 函数可以将一个**字符**转换为其**code points**。
    *   使用 `chr()` 函数可以将一个**code points**转换回对应的**字符**。


### Problem (unicode1): Understanding Unicode (1 point)
- (a) What Unicode character does chr(0) return?
   - Deliverable: A one-sentence response. => '\x00'，chr(0)返回的是 Unicode 代码点为 0 的字符，即​​空字符（Null character）​​。

- (b) How does this character’s string representation (__repr__()) differ from its printed representa-
tion? 
   - Deliverable: A one-sentence response. => "'\\x00'"，该字符的字符串表示（使用 repr）会显示为转义序列 \x00，而直接打印（使用 print）则​​无任何可见输出​​。

- (c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
   - Deliverable: A one-sentence response. => 空字符在文本中通常被视为​​不可见且无功能的字符​​，但某些系统或程序可能会将其解释为字符串终止符（如 C 语言中的传统），导致后续内容被截断或忽略。

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"  打印'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string") 打印this is a teststring
```

## 2.2 Unicode Encodings

### 内容概况

本节解释了为什么在训练分词器时不能直接使用Unicode码点，并引入了**Unicode编码**（特别是UTF-8）作为解决方案。核心思想是通过编码将字符转换为字节序列，从而将一个庞大稀疏的词汇表（15万个码点）转换成一个更小、更紧凑的字节词汇表（256个值），为后续训练字节级BPE分词器奠定基础。

---

### 要点总结

1.  **问题 (为何不使用Unicode码点)**：
    *   直接使用Unicode码点（约15.5万个）作为词汇表会导致其**过大**且**稀疏**（很多字符很罕见），不适合用于训练分词器。

2.  **解决方案 (使用Unicode编码)**：
    *   采用**UTF-8编码**（占互联网网页的98%以上）将Unicode字符串转换为**字节序列**。
    *   这一转换过程将码点（范围0-154,997）映射为字节值（范围0-255），词汇表大小从15万+降至固定的**256**。

3.  **技术实现 (Python操作)**：
    *   **编码（字符串 → 字节）**: 使用 `"字符串".encode('utf-8')`。
    *   **获取字节值**: 对返回的 `bytes` 对象使用 `list()` 可得到整数列表。
    *   **解码（字节 → 字符串）**: 使用 `bytes_object.decode('utf-8')`。

4.  **核心优势**：
    *   **解决未登录词(OOV)问题**：字节词汇表是封闭的（0-255），任何文本都能被表示为字节序列，从根本上避免了OOV问题。
    *   **为BPE做准备**：在256个字节的基础上应用BPE算法，可以学习并合并出有效的子词单元。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # Get the byte values for the encoded string (integers from 0 to 255).
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129,
161, 227, 129, 175, 33]
>>> # One byte does not necessarily correspond to one Unicode character!
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

### Problem (unicode2): Unicode Encodings (3 points)
- (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
- Deliverable: A one-to-two sentence response.  => 0-255字节值。UTF-8 是可变长度编码，对于 ASCII 字符只使用一个字节，而 UTF-16 和 UTF-32 使用固定长度（2 或 4 字节），导致存储效率较低且词汇表更大；此外，UTF-8 是互联网主导编码，兼容性更好，训练分词器时能减少处理复杂性。

- (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

- Deliverable: An example input byte string for which decode_utf8_bytes_to_str_wrong pro-
duces incorrect output, with a one-sentence explanation of why the function is incorrect. => 示例输入字节串：b'\xe7\x89\x9b'（即中文字符 "牛" 的 UTF-8 编码）。该函数错误在于它逐个字节解码 UTF-8 字节串，但 UTF-8 字符可能由多个字节组成，单独解码会抛出 UnicodeDecodeError或产生错误字符（如字节 0x89不是有效单独字符）。

- (c) Give a two byte sequence that does not decode to any Unicode character(s).
- Deliverable: An example, with a one-sentence explanation. =>  ### 示例字节序列：b'\xC0\x00'。该序列无效是因为第一个字节 0xC0表示两字节字符，但第二个字节 0x00不是有效的继续字节（必须以 10开头），因此无法解码为任何 Unicode 字符。

### UTF-8规则

UTF-8（Unicode Transformation Format - 8-bit）是一种**可变长度编码**，其核心设计是通过**1到4个字节**表示任意Unicode字符，同时**完全兼容ASCII**。其编码规则如下：

---

#### **UTF-8 编码格式规则**
| Unicode 码点范围 (十六进制) | UTF-8 字节序列 (二进制) | 字节数 |
|---------------------------|------------------------|-------|
| `U+0000` – `U+007F`       | `0xxxxxxx`             | 1字节 |
| `U+0080` – `U+07FF`       | `110xxxxx 10xxxxxx`    | 2字节 |
| `U+0800` – `U+FFFF`       | `1110xxxx 10xxxxxx 10xxxxxx` | 3字节 |
| `U+10000` – `U+10FFFF`    | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` | 4字节 |

---

#### **关键设计解析**
1. **前缀标识位**  
   - **首字节前缀**：标识总字节数（如 `110` 开头表示2字节字符）。  
   - **后续字节前缀**：固定以 `10` 开头，避免与其他字节混淆。

2. **兼容ASCII**  
   - ASCII字符（`U+0000`–`U+007F`）直接以单字节存储（最高位为 `0`），无需修改。

3. **码点填充规则**  
   - Unicode码点按规则拆分后，依次填入 `x` 标记的位置（从高位到低位）。  
   - **示例**：字符 `牛`（`U+725B`，十六进制 `29275`）  
     - 码点范围：`U+0800`–`U+FFFF` → **3字节编码**  
     - 十六进制 `725B` → 二进制 `0111 0010 0101 1011`  
     - 填入模板：`1110xxxx 10xxxxxx 10xxxxxx`  
       - 填充后：`11100111 10001001 10011011`  
       - 十六进制：`E7 89 9B` → 字节序列 `b'\xe7\x89\x9b'`

4. **错误检测机制**  
   - 无效序列示例：`0xC0 0x00`  
     - `0xC0`（二进制 `11000000`）要求后续字节以 `10` 开头，但 `0x00`（`00000000`）无效 → **解码失败**。

---

#### **为什么用UTF-8训练分词器？**
1. **词汇表压缩**  
   - Unicode码点（15万+）→ 压缩为 **256种字节值**（0-255）。
2. **消除未登录词（OOV）**  
   - 任何文本均可拆分为UTF-8字节序列，无OOV问题。
3. **跨语言兼容性**  
   - 统一处理所有语言字符（包括Emoji、罕见符号）。

---

#### **Python操作示例**
```python
text = "牛"
# 编码为UTF-8字节
byte_seq = text.encode("utf-8")  # b'\xe7\x89\x9b'
byte_list = list(byte_seq)       # [231, 137, 155]

# 解码回字符串
decoded_text = byte_seq.decode("utf-8")  # "牛"
```

## 2.3 Subword Tokenization

### 内容概况
本节介绍了**子词分词（Subword Tokenization）** 的核心概念，重点阐述了**字节对编码（BPE）** 算法的原理和动机。它解释了为何需要在**词汇表大小和序列长度**之间进行权衡，并说明了BPE如何通过迭代合并高频字节对来构建分词器，从而在解决**未登录词（OOV）问题**和控制序列长度之间取得最佳平衡。

---

### 要点总结

1.  **核心问题与动机**
    *   **字节分词的缺点**：虽然解决了OOV问题，但会导致序列过长（例如一个单词被拆成多个字节），显著增加计算开销（更长的序列需要更多计算）并加剧长程依赖的建模难度。
    *   **词分词的缺点**：词汇表庞大，且无法处理训练时未出现过的词（OOV问题）。

2.  **解决方案：子词分词**
    *   这是介于词分词和字节分词之间的一种折中方案。
    *   **核心思想**：通过使用一个**比256大但比词表小得多**的词汇表，来更好地“压缩”输入序列。常用序列（如字节串 `b'the'`）可以被合并为一个单独的标记，从而缩短序列总长度。

3.  **BPE算法的工作原理**
    *   它是一种数据压缩算法，被Adapted用于构建分词器词汇表。
    *   **训练过程**：算法**迭代地**查找并**合并（merge）** 训练语料中**最频繁出现**的相邻字节对（pair）。
    *   **结果**：频繁出现的字节序列会被合并成新的、更大的子词单元并加入词汇表。一个词如果出现得足够频繁，最终可能会被表示为一个单一的标记。

4.  **本作业的具体实现：字节级BPE**
    *   我们将实现一个**字节级BPE分词器**。这意味着：
        *   **基础单元**：词汇表中的项最初是256个字节，随后是合并后的字节序列。
        *   **双重优势**：既继承了字节分词**无OOV问题**的优点（任何文本都能用字节表示），又能通过合并获得**更短的序列长度**，提高了计算效率。

5.  **关键术语**
    *   **训练（Training）**：指在数据上运行BPE算法以构建合并规则和词汇表的过程，并非训练神经网络模型。

---

### 总结
这段内容清晰地勾勒出了从词分词到字节分词，再到子词分词（BPE）的技术演进路径，并阐明了BPE算法的设计动机和核心机制，为接下来实际实现BPE分词器提供了理论基础。其最终目标是构建一个既高效又实用的分词器。