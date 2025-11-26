本文主要整理Assignment 4 (data): Filtering Language Modeling Data的主要内容。

## 1. Assignment Overview

### 内容概况

其核心目标是让学生通过实践，学习如何将原始的网络爬虫（Common Crawl）HTML数据，经过一系列处理，转化为高质量、可用于训练语言模型的数据集。

### 要点总结

**1. 作业目标**
- **实践核心**：获得过滤网络爬虫数据以创建语言模型数据的实践经验。
- **最终目的**：通过在不同处理过的数据集上训练语言模型，理解不同的数据处理决策（如过滤、去重）如何影响模型的最终性能。

**2. 需要实现的具体任务**
- **数据转换**：将通用的网络爬虫（Common Crawl）原始HTML内容转换为纯文本。
- **数据过滤**：使用多种方法对提取出的文本进行清洗和过滤，例如：
    - 去除有害内容。
    - 去除个人可识别信息（PII）。
- **数据去重**：对训练数据进行重复项删除。

**3. 代码结构与使用说明**
- **代码仓库**：所有代码和文档均托管在GitHub上。
    - 地址：`github.com/stanford-cs336/assignment4-data`
    - 使用 `git clone` 获取代码，使用 `git pull` 更新。
- **目录结构**：
    - **`cs336-basics/`**：包含一个优化过的语言模型训练代码（基于作业1），支持多GPU分布式训练。学生将用此脚本在自已处理好的数据上训练模型。
    - **`cs336_data/`**：一个空模块，这是学生需要**编写作业代码的主要区域**，用于实现上述的数据处理任务。
    - **`tests/`**：包含必须通过的测试用例。学生需要通过实现 `tests/adapters.py` 中的钩子函数来连接自己的代码和测试。
    - **`README.md`**：包含更详细的目录结构说明和环境配置指南。

## 2. Filtering Common Crawl

### 内容概况

主要阐述了为什么大型语言模型的训练数据需要经过复杂的预处理，并介绍了本次作业的核心任务：**构建一个数据处理流程，将原始的Common Crawl网络爬虫数据转化为可用于训练语言模型的高质量数据集。**

### 要点总结

**1. 核心论点：原始网络数据不能直接使用**
- 尽管大型语言模型主要使用互联网数据进行训练，但研究人员通常不直接自己爬取数据。
- 他们更倾向于使用**Common Crawl**这类公开的、大规模的网络爬虫数据集。Common Crawl是一个非营利组织，提供了超过2500亿个网页、时间跨度达17年的庞大免费语料库。

**2. 主要挑战：从“原始数据”到“可用数据”的鸿沟**
- Common Crawl的原始数据是**HTML格式**的网页文件，而语言模型需要的是**纯文本**。
- 网页数据中存在大量质量问题，直接使用会影响模型性能，包括：
    - **低质量内容**（如垃圾邮件、无意义文本）。
    - **重复或近乎重复的页面**。
    - **有害内容**。
    - **敏感信息**（如个人身份信息PII）。

**3. 作业任务：建立数据处理流程（Pipeline）**
- 本次作业的目标是搭建一个数据处理流水线，对原始Common Crawl数据执行一系列清理和转换步骤。
- 这些步骤旨在解决上述挑战，最终产出干净、高质量的文本训练集。

## 2.1 Looking at the data

### 内容概括

本小节强调了在开始处理数据之前，**先查看并熟悉原始数据**的重要性。接着，它系统性地介绍了Common Crawl数据集提供的三种核心数据格式：**WARC**、**WAT**和**WET**文件，并清晰地说明了每种格式所包含的内容、用途及其相互关系。

---

### 要点总结

**1. 核心建议**
*   在实现任何数据处理逻辑之前，首要步骤是亲自查看原始数据，以获得对数据最直观的认识。

**2. Common Crawl 的三种数据格式**

| 格式名称与全称 | 内容与描述 | 用途与特点 |
| :--- | :--- | :--- |
| **WARC**<br/>(Web ARChive) | **最原始、最完整的数据**。包含：<br/>• 页面ID、URL<br/>• 元数据和HTTP请求详情（如请求时间、服务器IP）<br/>• **页面的原始内容（如HTML代码）** | 数据处理的**起点**，保留了所有原始信息。 |
| **WAT**<br/>(Web Archive Transformation) | **从WARC文件提取的、更高层级的元数据**，以JSON格式存储。例如：<br/>• 页面中的链接列表<br/>• 页面标题 | 用于分析页面结构（如链接关系），而不直接提供主要内容文本。 |
| **WET**<br/>(Web Extracted Text) | **已经从原始HTML中提取出的纯文本**。 | 提供了最接近最终训练所需格式（纯文本）的数据，简化了处理步骤。 |

**3. 关键关系**
这三种格式体现了数据处理的不同阶段和细化程度：
*   **WARC** 是源头，包含所有原始信息。
*   **WAT** 和 **WET** 都是由WARC文件**衍生处理**而来。
*   从**WARC**（原始HTML）到**WET**（纯文本），数据的可读性和可用性递增，但信息的丰富性递减。

### Problem (look_at_cc): 4 points

(a) Download the WARC file above, or find the copy we provide on the cluster. Let’s look at the
first page in this file. This is a gzipped file, and you can browse its contents with:
- $ zcat /data/CC/example.warc.gz | less
- less lets you browse the file using keyboard arrows, Page Up, Page Down. To exit, press “q”.
- Look at the very first web page. What is its URL? Is it still accessible? Can you tell what the
page seems to be about by looking at the raw HTML?

```python
http://0371rykj.com/ipfhsb/34.html
林頻產(chǎn)品
```

(b) Let’s now look at the corresponding WET file:
- $ zcat /data/CC/example.warc.wet.gz | less
- Note that the WET files contain HTTP headers (e.g., Content-Length) that are not part of the
extracted text contents. If you look at the first example, you will see that it contains text that
was extracted from the raw HTML you just saw.
- Notice that much of the extracted text is reminiscent of the HTML structure, and not actually
the page’s main content. Are there parts of the text you see that you think should have been
filtered out by the extractor? Think about the quality of this text as training data: what might
go wrong in training a model on text that looks like this? Conversely, what useful information
can a model potentially extract from this page?

WET文件中的文本并非完美的页面主要内容。它混杂了两种“噪音”：
- 结构性噪音：包含了本应被过滤掉的HTTP头信息（如Content-Length）。
- 内容性噪音：文本提取工具未能完美区分页面核心内容和周边元素，导致残留了大量来自HTML模板的文本（如导航菜单、侧边栏链接、版权声明、JavaScript代码等），而非文章主体。

如果将这种质量的文本直接用于训练语言模型，会引发问题：
- 学习错误模式：模型可能会学习并生成无关的模板文本（如反复输出“Home | About Us | Contact”这样的导航菜单），而非有意义的语言表达。
- 降低训练效率：大量低质量、重复的噪音文本会稀释高质量数据的权重，浪费计算资源，导致模型性能下降。
- 学习有毒内容

潜在的利用价值
- 特定领域的术语：恒溫恒濕試驗(yàn)箱是航空、汽車(chē)、家電、科研等領(lǐng)域必備的測(cè)試設(shè)備，用于測(cè)試和確定電工、電子及其他產(chǎn)品及材料進(jìn)行高溫、低溫、濕熱度或恒定試驗(yàn)的溫度環(huán)境變化后的參數(shù)及性能。
- 基本的语言结构：忽略噪音部分后，剩余的有效句子仍然能帮助模型理解语法和句法。

(c) What makes a good training example is highly contextual. Describe an application domain for
which this example might be useful to have in the training data, and one where it might not be.
- **科技类文本**，评价数据质量的关键在于 “任务需求”

(d) Let’s look at some more examples to get a better sense of what’s in the Common Crawl. Look
through 25 more WET records. For each record, very briefly comment on the document’s language
(if you can identify it), the domain name, what type of page it is, etc. How many examples does
it take until you see what you’d deem a “high-quality” webpage?

**绝大部分是低质量或有毒文本**
- 恒溫恒濕試驗(yàn)箱是航空、汽車(chē)、家電、科研等領(lǐng)域必備的測(cè)試設(shè)備，用于測(cè)試和確定電工、電子及其他產(chǎn)品及材料進(jìn)行高溫、低溫、濕熱度或恒定試驗(yàn)的溫度環(huán)境變化后的參
數(shù)及性能。
- 呵呵，昨天在朋友的帮助下我尝试用了新办法合并视频，成功的感觉挺好的。虽然从等待翻译、落实片源，到制作字幕、压制、合并、上传一系列过程很繁琐，也很费时；虽然做出来的东西不是尽善尽美，但依旧是开心的。真正是累并快乐着。 现在，2003年的tik杰西达邦和Maam K McIntosh合作《两地相思》中文字幕版终于可以和大 ...
- 随着体育赛事的不断增多，越来越多的球迷开始关注各类比赛的结果和动态。在这样的背景下，澳客竞彩比分应运而生，成为球迷们获取赛事信息和比分数据的重要平台。无论是足球、篮球还是其他各类体育项目，用户都可以通过这个平台获取实时的比分更新、赛程安排、数据分析等信息，为他们的观赛体验增添更多乐趣。
- 在当今互联网快速发展的时代，越来越多的人选择在线参与各种活动，而福利彩票作为一种受欢迎的公益活动，吸引了越来越多的参与者。福利彩票论坛为广大用户提供了一个交流、分享和获取最新信息的平台。无论你是对福利彩票玩法感兴趣，还是想了解最新的中奖资讯，亦或是希望结识志同道合的朋友，这里都能满足你的需求。
- 总院:四川省成都市青羊区一环路西二段32号;社区服务门诊:省医院第二门诊部地址:成都市外双楠置信路57号附54号;省医院第三门诊部地址;成都市领事馆路9号附1、2号;省医院第五门诊部地址:成都市府南新区刘一手斜对面;草堂病区:成都市大石西路62号;城东病区:位于成都市龙泉驿区大面镇洪河北路589号(市东三环成渝立交桥旁)
- 一位消化外科專家表示：“電凝切割器的低溫切割功能減少了對周圍組織的熱損傷 ，從而降低術后疼痛 。同時 ，其即時止血的特點降低了術中失血量 ，提高了手術安全性 。”
- 欢迎在线观看由内详等主演在2025年的动漫《CLASSIC★STARS–古典乐★之星》，真不卡影院 电影盒子第一时间为你提供CLASSIC★STARS–古典乐★之星完整版在线免费观看，CLASSIC★STARS–古典乐★之星讲述了这是一部以音乐梦想为主题的励志影片CLASSIC★STARS–古典乐★之星讲述一群热爱古典音乐的年轻人在追逐梦想的道路上克服重重困难最终通过不懈努力和团队合作实现了自我突破并登上国际舞台的故事影片通过紧张刺激的比赛情节和感人至深的人物情感描绘展现了古典音乐的魅力同时也传递了坚持梦想永不放弃的精神让观众在欣赏精彩表演的同时感受到青春与奋斗的力量
- 绍兴市越城区皋埠街道社区卫生服务中心 版权所有 地址：浙江省绍兴市越城区皋埠镇银兴路 联系电话：0575-88756599 备案号： 浙ICP备15013313号 技术支持：台州华顶网络技术有限公司
- 依'電腦網際網路分級辦法'為限制級，限定為年滿18歲且已具有完整行為能力之網友，未滿18 歲謝絕進入瀏覽，且願接受本站內影音內容及各項條款。為防範未滿18歲之未成年網友瀏覽網路上限制級內容的圖文資訊，建議您可進行 網路分級基金會TICRF分級服務的安裝與設定。 (為還給愛護本站的網友一個純淨的聊天環境，本站設有管理員)
- 安全性是每位用户在选择娱乐平台时所关注的重要因素。该平台采用了先进的加密技术，确保用户的个人信息及财务数据得到保护。同时，提供多种安全的支付方式，让用户在充值和提现时无需担心资金安全问题。

## 2.2 HTML to text conversion

### 内容概况

本节详细阐述了在构建语言模型训练数据时，**从HTML网页中提取高质量文本内容所面临的核心挑战**，并介绍了本作业中将采用的**解决方案和工具库（Resiliparse和FastWARC）**。

### 要点总结

**1. 核心挑战：区分“主内容”与“噪音文本”**
- **问题根源**：简单的HTML标签提取（如抓取所有 `<p>` 标签）会包含大量非核心内容。
- **具体例子**：以StackOverflow页面为例，虽然主要信息是问答内容，但提取工具也会同时获取到：
    - 菜单选项
    - 其他StackExchange站点的无关链接
    - 页脚信息
    - 登录/注册链接
- **结论**：如何可靠地将页面的“主内容”从这些“噪音文本”中区分出来，是一个重大挑战。

**2. 解决方案：使用专用工具库**
为了解决上述挑战，作业推荐使用以下工具库构建文本提取流程：

- **Resiliparse 库**：
    - **主要作用**：用于执行实际的**文本提取**工作。
    - **附加优势**：能自动检测并处理原始字节的**文本编码**问题（如UTF-8、GBK等），确保流程对不同类型的网页编码具有鲁棒性。

- **FastWARC 库**：
    - **主要作用**：用于高效地**遍历和读取WARC文件**中的每一条记录。
    - **使用建议**：代码示例中提到了 `ArchiveIterator` 和 `WarcRecordType` 这两个类，它们是读取WARC数据的关键工具。

**核心逻辑关系**：整个处理流程将是先用 **FastWARC** 从WARC文件中读取原始记录，然后使用 **Resiliparse** 对这些记录中的HTML内容进行文本提取和编码转换。

### Problem (extract_text): 3 points

(a) Write a function that extracts text from a byte string containing raw HTML. Use
resiliparse.extract.html2text.extract_plain_text to perform the extraction. This func-
tion needs a string, so you will need to first decode the byte string into a Unicode string. Be
aware that the input byte string might not be encoded in UTF-8, so your function should be able
to detect the encoding in case UTF-8 fails. Resiliparse also offers
resiliparse.parse.encoding.detect_encoding(), which might be useful.
- 完成

(b) Run your text extraction function on a single WARC file. Compare its output to the extracted
text in the corresponding WET file. What differences and/or similarities do you notice? Which
extraction seems better?

- LRHS-101-LH和450×450×500分两行
- 开头末尾<ul id="mescs"><center id="mescs"></center></ul>

## 2.3 Language identification

### 内容概况

本节阐述了在构建训练数据集时，进行**语言识别（Language Identification）** 的必要性，并详细介绍了如何使用 **fastText** 这一工具库及其预训练模型来实现一个语言过滤器。

### 要点总结

**1. 核心问题：网页语言的多样性**
*   互联网上的网页由成千上万种语言编写。
*   然而，在有限的算力预算下，训练一个能有效利用如此多样化数据的大规模多语言模型极具挑战性。
*   因此，多数基于Common Crawl的语言模型训练集**只包含有限几种语言的数据**。这就产生了对网页进行语言筛选的需求。

**2. 解决方案：使用fastText库进行语言识别**
*   **工具介绍**：fastText是一个高效的文本分类库，提供了训练自定义分类器和预训练模型。
*   **预训练模型**：本节指定使用其语言识别模型 `lid.176.bin`，并提供了两个下载路径：
    *   官方网址：`https://fasttext.cc/docs/en/language-identification.html`
    *   课程集群路径：`/data/classifiers/lid.176.bin`
*   **过滤器的工作原理**：过滤器并非简单地判断“是/否”，而是基于模型给出的**置信度分数**来决定是否保留某个页面。分数越高，模型对其预测的语言就越确信。

**3. 作业任务**
*   需要实现一个语言识别过滤器。该过滤器应能调用fastText语言识别分类器，并返回一个表示预测置信度的非负分数。

### Problem (language_identification): 6 points

(a) Write a function that will take a Unicode string and identify the main language that is present
in this string. Your function should return a pair, containing an identifier of the language and a
score between 0 and 1 representing its confidence in that prediction.
- 完成

(b) The behavior of language models at inference time largely depends on the data they were trained
on. As a result, issues in the data filtering pipeline can result in problems downstream. What
issues do you think could arise from problems in the language identification procedure? In a
higher-stakes scenario (such as when deploying a user-facing product), how would you go about
mitigating these issues?

#### 第一部分：语言识别过程可能导致的问题
语言识别是数据过滤的关键步骤，如果出现错误，可能引发以下下游问题：

1. **训练数据污染**：
   - 如果语言识别错误，非目标语言的文本可能被误包含在训练数据中（例如，将英文文本误判为中文）。这会导致模型学习到错误的语言模式，影响模型在目标语言上的生成质量，如产生语法错误、语义混乱或混合语言输出。

2. **模型偏见和性能下降**：
   - 语言识别错误可能使训练数据分布失衡，例如，过度包含某些方言或低资源语言，导致模型对主流语言的支持不足。在推理时，模型可能对某些查询响应不佳，或表现出对特定语言群体的偏见。

3. **用户体验受损**：
   - 在面向用户的产品中（如聊天机器人或翻译工具），语言识别错误可能导致模型无法理解用户输入，或生成不相关、冒犯性的内容。例如，用户用中文提问，但模型可能返回英文响应或其他语言的内容，降低可信度。

4. **安全风险**：
   - 如果语言识别失败，有害内容（如垃圾信息或恶意文本）可能因未被正确过滤而进入训练数据。模型在推理时可能复现这些内容，引发安全或伦理问题。

#### 第二部分：高利害场景下的缓解措施
在高利害场景（如用户产品部署）中，需要采取多层次策略来减轻语言识别问题的影响：

1. **改进语言识别流程**：
   - 使用多个语言识别模型进行投票或集成学习，以提高准确性。例如，结合 fastText 与其他库（如 langid.py）。
   - 设置置信度阈值：只保留高置信度的预测结果（如置信度 >0.9），对低置信度文本进行人工审核或丢弃。
   - 对训练数据进行预处理，包括文本清洗（如移除特殊字符）和长度过滤，以减少噪声。

2. **数据质量监控**：
   - 实施数据验证管道：在训练前，对过滤后的数据进行抽样检查，使用人工评估或自动化工具验证语言标签的准确性。
   - 持续监控数据分布：定期分析训练数据的语言组成，确保与目标语言分布一致。

3. **模型测试和评估**：
   - 在部署前，进行多语言压力测试：使用包含边缘案例的测试集评估模型性能，如混合语言输入或低资源语言查询。
   - 引入红队测试：模拟恶意输入或极端场景，检查模型是否会产生不当输出。

4. **部署后的应急机制**：
   - 设置用户反馈循环：允许用户报告问题（如语言识别错误），并快速迭代模型更新。
   - 实施实时监控：使用日志分析工具检测模型输出的异常模式（如突然出现非目标语言响应），并触发自动回滚或人工干预。

5. **伦理和透明度**：
   - 明确产品的能力边界：在文档中说明模型支持的语言范围，管理用户期望。
   - 确保数据来源的多样性：从多个权威渠道收集数据，减少单一源带来的偏差。

(c) Run your language identification system on text extracted from the WARC files (via your
previously-implemented text extraction function). Manually identify the language in 20 random
examples and compare your labels with the classifier predictions. Report any classifier errors.
What fraction of documents are English? Based on your observations, what would be a suitable
classifier confidence threshold to use in filtering?

- 8  en 0.11044929921627045
- Яндекс.Метрика The journal "AIC: economics, management" Issues About Editorial Board For authors Articles About Login РУС Винокуров Сергей Иннокентьевич || Vinokurov Sergei Innokentevich Irkutsk State en 0.48027339577674866
- Português 简体中文 繁體中文 Deutsch English Español Français 日本語 Latviešu Lietuvių Русский Conferences Call For Speakers Result of Speakers Submissions Yerevan Sydney Sydney Register Sydney Sponsor Vienna Min en 0.6359551548957825
- classifier confidence threshold: 0.7

## 2.4 Personal identifiable information

### 内容概况

本节指出了网络数据中包含大量**个人身份识别信息（PII）** 所带来的隐私与安全问题，并明确了本次作业的一项具体任务：**实现三种PII的屏蔽程序**，以防止语言模型在训练中学到并泄露真实个人的敏感信息。

### 要点总结

**1. 核心问题：PII的存在带来风险**
*   **风险来源**：网络数据（如Common Crawl）天然包含大量可用于定位或识别个人的信息。
*   **PII示例**：常见的PII包括电子邮件地址、电话号码和IP地址。
*   **潜在危害**：如果语言模型在训练数据中接触并学习了这些真实的PII，它可能在面向用户时生成并输出这些信息，从而引发严重的隐私和安全问题。

**2. 解决方案：在训练数据中屏蔽PII**
*   **核心思想**：作为一种常见的预防措施，在构建训练数据集时，主动将这些PII从文本中“屏蔽”或“隐去”。
*   **目的**：这可以降低模型记忆并生成特定个人真实信息的可能性，增强模型的安全性。

**3. 作业任务：实现三种屏蔽程序**
作业要求具体实现以下三类PII的识别与屏蔽功能：
*   **(a) 电子邮件地址**
*   **(b) 电话号码**
*   **(c) IP地址**

### Problem (mask_pii): 3 points

1. Write a function to mask out emails. Your function will take a string as input, and replace all
instances of email addresses with the string "|||EMAIL_ADDRESS|||". To detect email addresses,
you can look up regular expressions that do this reliably.
- 完成

2. Write a function to mask out phone numbers. Your function will take a string as input, and replace
all instances of phone numbers with the string "|||PHONE_NUMBER|||". Doing this reliably can
be extremely challenging, as phone numbers might be written in an extremely diverse set of formats, but you should try to capture at least the most common phone number formats used in the United States, and be robust to minor syntactic deviations.
- 完成

3. Write a function to mask out IP addresses. For this problem, it is enough to focus on IPv4
addresses (4 numbers up to 255 separated by points). Your function will take a string as input,
and replace all instances of IP addresses with the string "|||IP_ADDRESS|||".
- 完成

4. What problems do you think might arise downstream in a language model when these filters are
naïvely applied on the training set? How might you mitigate these issues?

1.  **核心问题（Problems）：**
    *   **过度过滤（Over-filtering）：** 过于严格的过滤会删除大量有价值的文本，导致模型无法学习到语言的多样性和复杂性，使其变得刻板、缺乏创造力。
    *   **放大偏见（Amplifying Bias）：** 如果过滤器本身有偏见（例如，用于识别“高质量”文本的分类器是在特定文化或群体的数据上训练的），那么过滤后的数据集会进一步放大这种偏见，导致模型对某些群体或语言风格表现不佳。

2.  **缓解措施（Mitigation）：**
    *   **谨慎调参（Careful Tuning）：** 不要使用过于激进的过滤阈值。例如，在质量过滤中，可以设置一个相对宽松的阈值，保留更多样化的数据。
    *   **组合使用与人工审核（Combination and Human Review）：** 不要依赖单一过滤器。结合多种方法，并对处于过滤边界的数据进行人工抽样检查，以确保过滤的合理性。
    *   **持续监控（Continuous Monitoring）：** 在模型训练后，持续评估其在不同数据子集上的表现，检查是否存在因过滤不当而产生的偏见或能力缺陷。

5. Run your PII masking functions on text extracted from the WARC files (via your previously-
implemented text extraction function). Look through 20 random examples where a replacement
was made; give some examples of false positives and false negatives.

**false positives**
- Index of /yatake-sound/20210913 [ICO]NameLast modifiedSizeDescription [PARENTDIR]Parent Directory  - [SND]20210913080001yatake.mp32021-09-13 08:55 5.4M [SND]20210913130001yatake.mp32021-09-13 13:55 5. => Index of /yatake-sound/20210913 [ICO]NameLast modifiedSizeDescription [PARENTDIR]Parent Directory  - [SND]|||PHONE_NUMBER|||yatake.mp32021-09-13 08:55 5.4M [SND]|||PHONE_NUMBER|||yatake.mp32021-09-13 en 0.5188405513763428 数字误判为电话号码
- Skip to content Bear(ing) the News \u2013 A Chicago Bears Blog Chicago Bears News and Views as Well as Points of View from Around the NFL April 2025 S M T W T F S 12345 6789101112 13141516171819 2021222324 => Skip to content Bear(ing) the News \u2013 A Chicago Bears Blog Chicago Bears News and Views as Well as Points of View from Around the NFL April 2025 S M T W T F S 12345 |||PHONE_NUMBER||| |||PHONE_NUMBER|| en 0.9875467419624329 数字误判为电话号码

**false negatives**
- Skip to content 061 784 8975 | 064 585 446255/46, MoobanBiggerland, Moo 3, Lamlukka, Lamlukka, Pathumthani, 12150 Thailand. Search: Abo Green Solution Abo Green SolutionAbo Green Solution Home About u => Skip to content |||PHONE_NUMBER||| | |||PHONE_NUMBER|||/46, MoobanBiggerland, Moo 3, Lamlukka, Lamlukka, Pathumthani, 12150 Thailand. Search: Abo Green Solution Abo Green SolutionAbo Green Solution Ho en 0.6911945939064026  特殊电话号码格式 064 585 446255/46

## 2.5 Harmful content

### 内容概况

本节阐述了从网络数据中**过滤有害内容**的必要性，并明确了本次作业的具体任务：**使用由Dolma项目提供的预训练fastText分类器，来识别并过滤两类有害内容**。

### 要点总结

**1. 核心问题：网络数据中含有大量有害内容**
*   **普遍存在**：未经处理的网络数据包含大量我们不希望语言模型在推理时复述的内容。即使是通常无害的网站（如Wikipedia的用户评论区）也可能存在有害信息。
*   **界定困难**：虽然很难为“有害内容”划定一条绝对清晰的界线，但大多数数据过滤流程仍会进行此类过滤。

**2. 解决方案：使用预训练分类器进行识别**
*   **识别方法**：作业摒弃了简单的词表过滤等方法，转而采用更先进的、基于人类标注数据训练的分类器。
*   **聚焦两类有害内容**：
    1.  **NSFW**：指“不适合工作场所”的内容，包括色情、污言秽语或其他可能令人不安的内容。
    2.  **Toxic Speech**：指“粗鲁、不尊重或不合理的语言，很可能导致他人退出讨论”的有毒言论。
*   **指定工具**：使用 **Dolma项目** 基于 **Jigsaw Toxic Comments 数据集** 训练的 **fastText** 预训练模型。

**3. 作业任务与资源**
*   **核心任务**：实现一个函数，该函数接收一个包含页面内容的Unicode字符串，使用提供的分类器进行分析，并返回一个分类标签（例如，“toxic”/“non-toxic”）及其相应的置信度分数。
*   **模型下载路径**：
    *   **NSFW 分类器**：`dolma-artifacts.org/.../jigsaw_fasttext_bigrams_nsfw_final.bin`
    *   **Hate Speech 分类器**：`dolma-artifacts.org/.../jigsaw_fasttext_bigrams_hatespeech_final.bin`
    *   **课程集群路径**：模型也已放置在集群的 `/data/classifiers/` 目录下，方便学生直接使用。

### Problem (harmful_content): 6 points

1. Write a function to detect NSFW content.
- 完成

2. Write a function to detect toxic speech.
- 完成

3. What problems do you think might arise downstream in a language model when these filters are
applied to create the training set? How might you mitigate these issues?

**下游可能出现的问题：**

1.  **模型偏见与失真：** 过度过滤会移除大量带有复杂情感、讽刺或特定文化背景的文本。这可能导致训练出的语言模型表达能力单一、刻板，无法理解或生成 nuanced（有细微差别）的人类语言，甚至**放大数据集中的现有偏见**（例如，模型可能学会将某些群体或话题与负面内容过度关联）。
2.  **分布偏移与泛化能力差：** 训练数据与真实世界数据分布不一致。模型在学习过程中从未见过被过滤掉的“边缘”内容，导致其在遇到真实世界中稍微敏感或复杂的用户查询时，可能表现不佳或产生不可预测的输出。
3.  **过度审查与创造力受限：** 过于严格的过滤器可能会将有价值的文学创作、历史文献或社会议题讨论误判为有害内容并移除，从而限制模型的知识广度和创造力。

**缓解这些问题的措施：**

1.  **谨慎调参，而非简单二元过滤：** 不要简单地丢弃所有被标记的内容。应该**为置信度设置一个合理的阈值**。例如，只过滤掉置信度高于0.95的极端内容，对于置信度在0.5-0.95之间的“模糊”样本，可以考虑保留或进行人工复审。
2.  **人工审核与迭代：** 建立一个人工审核流程，定期抽样检查被过滤器判定为阳性的内容（即被标记的内容），特别是那些低置信度的案例。这有助于发现过滤规则的偏差并进行迭代优化。
3.  **多层次过滤与透明度：** 不要仅仅依赖一个过滤器。可以结合多种技术（如基于规则的过滤、多个不同模型的集成判断）来做出更均衡的决策。同时，记录过滤决策，以便后期分析和审计。

4. Run your harmful content filters on text extracted from the WARC files (via your previously-
implemented text extraction function). Look through 20 random examples and compare the
classifier predictions to your own judgments. Report any classifier errors. What fraction of
documents are harmful? Based on your observations, what would be suitable classifier confidence
threshold(s) to use in filtering?

- 判定为nsfw、toxic可基本过滤

## 2.6 Quality Rules

### 内容概况

由于网络数据中包含大量低质量内容（如付费墙页面、链接错误占位页、非文本主导页面等），直接使用这些数据会影响语言模型训练效果。因此，本节借鉴Gopher论文[Rae et al., 2021]提出的启发式规则，定义了一套简单可解释的质量过滤标准，并明确了本次作业需实现的具体规则。

---

### 要点总结
1. **过滤目标**  
   解决语言过滤和有害内容移除后仍存在的**低质量网页问题**，例如：
   - 付费墙内容页面
   - 损坏链接的占位页面
   - 登录/注册表单页面
   - 以非文本内容（如图片、视频）为主、文本提取后无效的页面

2. **方法依据**  
   采用Gopher论文中提出的**启发式规则**，通过简单可量化的标准快速筛选低质量文本。

3. **作业需实现的过滤规则**  
   需移除符合以下任一条件的文档：
   - **文档词数**少于50或超过10万词；
   - **平均词长**不在3–10个字符范围内；
   - **超过30%的行**以省略号（“…”）结尾；
   - **少于80%的词汇**包含至少一个字母字符（用于过滤无意义符号或乱码）。

4. **扩展参考**  
   完整规则详见Gopher论文的附录A，建议深入阅读以全面了解质量过滤的设计逻辑。

### Problem (gopher_quality_filters): 3 points

(a) Implement (at least) the subset of the Gopher quality filters as described above. For tokenizing
text into words, you might find the NLTK package useful (specifically nltk.word_tokenize),
though you’re not required to use it.
- 完成

(b) Run your rule-based quality filter on text extracted from the WARC files (via your previously-
implemented text extraction function). Look through 20 random examples and compare the filter
predictions to your own judgment. Comment on any cases where the quality filters differ from
your judgments.
- 过滤器有时会过度拒绝包含专业术语或技术文档的文本（因平均词长超过10个字符），同时可能通过一些结构混乱但符合表面规则的营销内容。
- 特别是对于非英语母语者创作的文本，虽然内容有价值，但可能因标点使用习惯（如过多省略号）而被误判。


## 2.7 Quality Classifier

### 内容概况
内容指出文本质量评估是信息检索领域的经典问题，并借鉴搜索引擎（如PageRank算法）和OpenAI构建GPT-2训练集（WebText）的思路——即通过高质量来源（如Reddit高赞链接、维基百科外链）作为正样本，训练一个fastText分类器来对Common Crawl海量页面进行质量评分和过滤。

---

### 要点总结

**1. 质量评估的背景与挑战**
*   **核心问题**：简单的启发式规则（Gopher规则）无法完全捕捉内容的语义质量。
*   **经典思路**：借鉴搜索引擎（利用网页链接结构判断质量）和OpenAI（收集Reddit高“karma”评论中的链接）的成功经验，表明**高质量页面通常会相互引用或被权威社区推荐**。
*   **当前目标**：在保证数据规模的前提下提升质量，避免因仅使用受控源（如维基百科）导致数据集过小。

**2. 解决方案：训练质量分类器**
*   **方法**：采用**fastText分类器**，进行监督学习。
    *   **正样本**：来自维基百科页面所引用的外部链接（提供了一份包含4350万个URL的文件 `/data/wiki/enwiki-20240420-extracted_urls.txt.gz` 作为正样本来源）。
    *   **负样本**：从Common Crawl中随机选取的页面。
*   **关键权衡**：通过设置分类器的质量分数阈值，可以在**精确度（Precision）** 和**召回率（Recall）** 之间进行权衡。

**3. 作业任务与实施步骤**
*   **核心任务**：构建一个质量分类器。
*   **关键步骤**：
    1.  **获取正样本**：使用提供的维基百科URL文件，通过 `wget` 命令（图中给出了具体命令示例）抓取网页内容并保存为WARC格式。
    2.  **注意事项**：识别到这些来自维基百科的“高质量”正样本可能仍包含不良内容，因此建议**结合之前已实现的过滤模块**（如语言识别、基础质量规则、有害内容过滤等）对其进行进一步清洗。
    3.  **训练与应用**：用清洗后的正样本和Common Crawl负样本训练分类器，并用于过滤整个数据集。

### 实施流程示意
**收集正样本URL → 使用wget抓取为WARC → 应用已有过滤器清洗 → 与负样本共同训练分类器 → 对Common Crawl进行质量评分过滤**

```python
wget --timeout=5 -i enwiki-20240420-extracted_urls_sample_v1.txt --warc-file=enwiki-20240420-extracted_urls_sample_v1.warc -O /dev/null
```

### Problem (quality_classifier): 15 points

(a) Train a quality classifier that, given text, returns a numeric quality score.

(b) Write a function that labels a page as high or low-quality, and provides a confidence score in the
label.

Read 17M words
Number of words:  9047180
Number of labels: 2
Progress: 100.0% words/sec/thread: 3865308 lr:  0.000000 avg.loss:  0.137391 ETA:   0h 0m 0s
模型已保存至: ./data/fasttext_quality_v1.bin
训练集评估 - 样本数: 1860, 精确率: 0.9817, 召回率: 0.9817

- 完成