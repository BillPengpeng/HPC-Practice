本文主要整理Assignment 2 (systems): Systems and Parallelism的主要内容。

## 1 - Assignment Overview

### 内容概况
这份作业文档描述了一个专注于GPU性能优化的系统编程任务。学生需要在一份已有Transformer模型（基于第一次作业）的基础上，通过剖析性能瓶颈、实现自定义高效内核以及利用分布式训练技术，来提升单GPU训练速度并扩展到多GPU环境。作业提供了详细的代码框架、测试用例和提交说明。

### 要点总结

1.  **核心目标**：
    *   **优化单GPU训练速度**：剖析模型性能，并实现更快的自定义操作（如Flash Attention 2）。
    *   **扩展至多GPU训练**：实现分布式数据并行训练和优化器状态分片，以利用多个GPU。

2.  **需要具体实现的任务**：
    *   **基准测试与性能剖析框架**：用于分析模型在前向传播和反向传播过程中的时间和内存消耗。
    *   **Flash Attention 2 Triton内核**：使用Triton编写自定义的GPU内核，优化自注意力机制，目标是要比朴素的PyTorch实现更快。
    *   **分布式数据并行训练**：实现多GPU训练。
    *   **优化器状态分片**：一种优化技术，用于减少每个GPU在分布式训练中的内存占用。

3.  **代码结构与资源**：
    *   **代码仓库**：所有代码和文档均在GitHub上提供（`github.com/stanford-cs336/assignment2-systems`）。
    *   **关键目录**：
        *   `cs336-basics/`：包含第一次作业的参考解决方案，作为本次作业的基础模型。
        *   `/ (根目录)`：主要工作目录，学生将在此实现新功能。
        *   `tests/`：包含必须通过的测试用例，学生需要实现特定的适配器接口来连接自己的代码。
    *   **环境设置**：README.md文件提供了环境配置的指导。

## 1.1 Profiling and Benchmarking

### 内容概况
主要阐述了在进行任何代码优化之前，必须首先对程序进行性能剖析（Profiling）的重要性。其核心论点是：只有通过剖析找到消耗资源（如时间和内存）的关键瓶颈，优化工作才能有的放矢，从而带来可衡量的端到端性能提升。文档随后简要介绍了将实施的三种性能评估方法。

### 要点总结

1.  **核心原则：先剖析，后优化**
    *   图片开篇即强调，在实施优化前进行剖析是至关重要的一步。这是为了避免“盲目优化”，即花费精力去优化那些对整体性能影响微乎其微的部分。

2.  **剖析的三大目标**
    *   **识别资源消耗点**：明确程序在时间和内存上的主要开销所在。

3.  **三种具体的性能评估路径**
    *   **(a) 端到端基准测试**：使用Python标准库进行简单的整体耗时测量，记录模型前向传播和反向传播的总时间。
    *   **(b) 计算剖析**：使用专业的NVIDIA Nsight Systems工具进行深入分析，了解时间在CPU和GPU的各个操作之间是如何分布的。
    *   **(c) 内存剖析**：专门分析程序的内存使用情况，识别潜在的内存瓶颈或浪费。

4.  **最终目的**
    *   确保优化工作能针对真正的性能瓶颈，最终实现**可测量的端到端改进**。


## 1.1.1 Setup - Importing your Basics Transformer Model

### 内容概况
其主要内容是指导学生如何正确设置环境，以便能够成功导入他们在第一次作业中实现的（或课程提供的）基础Transformer模型，为后续的性能剖析和优化任务做好准备。

### 要点总结

1.  **核心目标**：
    *   确保能够顺利加载第一次作业中完成的Transformer模型，为本次作业的优化工作建立基础。

2.  **实现方法**：
    *   **利用Python包管理**：模型在第一次作业中已被打包成一个Python包（`cs336-basics`）。
    *   **自动路径配置**：作业框架已在其 `pyproject.toml` 配置文件中预先指向了课程官方的实现包路径。
    *   **使用指定命令**：通过 `uv run [command]` 命令运行程序，包管理器`uv`会自动识别并定位到这个本地包。

3.  **关键选项**：
    *   **使用官方实现（默认）**：无需修改，直接使用课程提供的模型。
    *   **使用自己的实现**：如果需要，学生可以修改 `pyproject.toml` 文件，将其指向自己实现的模型包路径。

4.  **验证步骤**：
    *   文档提供了一个简单的测试方法：在命令行中执行 `uv run python` 进入Python交互环境，然后尝试执行 `import cs336_basics`。如果导入成功（不报错），则说明设置正确。之后便可以像 `import cs336_basics.model` 这样导入具体的模块。

```python
[tool.uv.sources]
# cs336-basics = { path = "./cs336-basics", editable = true }  # Change this path to your assignment1-basics repo you want to use your own implementation!
cs336-basics = { path = "../assignment1-basics", editable = true }  # Change this path to your assignment1-basics repo you want to use your own implementation!
```

## 1.1.2 Model Sizing

### 内容概况
该部分明确了本次作业将用于基准测试和性能剖析的一系列不同规模的Transformer模型配置，旨在让学生观察和分析模型性能随规模变化的规律。文档还强烈建议学生使用代码自动生成报告中的表格，以提高效率。

### 要点总结

1.  **基准测试目标**：为了系统地理解模型性能，作业要求对一系列不同规模的模型进行测试和分析（Benchmarking and Profiling）。

2.  **固定参数**：
    *   **词表大小**：10，000
    *   **批量大小**：4
    *   **上下文长度**：可变（Varying context lengths）

3.  **定义的模型规格**：
    表格中定义了从“small”到“2.7B”共五种模型配置，具体参数（`d_model`, `d_ff`, `num_layers`, `num_heads`）如下表所示：

| 模型规模 | d_model | d_ff | num_layers | num_heads |
| :------- | :------ | :--- | :--------- | :-------- |
| small    | 768     | 3072 | 12         | 12        |
| medium   | 1024    | 4096 | 24         | 16        |
| large    | 1280    | 5120 | 36         | 20        |
| xl       | 1600    | 6400 | 48         | 25        |
| 2.7B     | 2560    | 10240 | 32        | 32        |

4.  **重要建议**：
    *   **自动化生成表格**：作业强调会需要大量表格来呈现结果，强烈建议学生使用代码（如pandas的`to_latex()`或`to_markdown()`方法）来自动生成报告表格，以避免手动排版带来的繁琐工作。

## 1.1.3 End-to-End Benchmarking

### 内容概况
该部分旨在指导学生实现一个基础的性能评估脚本，用于测量模型前向传播和反向传播的时间与内存消耗。文档特别强调了测量GPU代码性能时的关键注意事项，并提供了具体的实现指导。

### 要点总结

1.  **核心任务**：
    *   实现一个简单的性能评估脚本，用于对模型进行端到端的基准测试，主要测量**前向传播和反向传播的速度（时间）和内存占用**。

2.  **脚本设计建议**：
    *   **通过命令行参数控制变量**：由于需要测试模型的多种变体（如改变精度、交换层等），建议通过命令行参数来实现这些变化，以便于后续快速、批量运行。
    *   **进行超参数扫描**：强烈建议使用工具（如`sbatch`或`submitit`在Slurm集群上）对基准测试的超参数（如模型规模、上下文长度等）进行扫描，以快速迭代。

3.  **关键技术与陷阱（最重要的要点）**：
    *   **CUDA调用是异步的**：在GPU上，当调用一个操作（如`torch.matmul`）时，函数会立即返回，而实际的计算则在GPU上异步进行。这意味着，如果简单地用计时函数包裹这个调用，测量的只是CPU发出指令的时间，而不是GPU实际完成计算的时间。
    *   **解决方案**：在测量GPU代码运行时间之前和之后，调用 **`torch.cuda.synchronize()`** 函数。这个函数会强制CPU等待GPU上所有未完成的内核执行完毕，从而确保计时器测量的是GPU内核的实际运行时间，从而获得准确的性能数据。

4.  **测试数据**：
    *   为了只测量速度和内存，可以使用**随机生成的权重和输入数据**，无需使用真实的训练数据集。

总而言之，这部分内容的核心是建立一个**准确、自动化**的基准测试流程，并着重指出了实现准确GPU计时所必须的一个关键技术细节。


### Problem (benchmarking_script): 4 points

- (a) Write a script to perform basic end-to-end benchmarking of the forward and backward passes in
your model. Specifically, your script should support the following:
    - Given hyperparameters (e.g., number of layers), initialize a model.
    - Generate a random batch of data.
    - Run w warm-up steps (before you start measuring time), then time the execution of n steps
(either only forward, or both forward and backward passes, depending on an argument). For
timing, you can use the Python timeit module (e.g., either using the timeit function, or
using timeit.default_timer(), which gives you the system’s highest resolution clock, thus
a better default for benchmarking than time.time()).
    - Call torch.cuda.synchronize() after each step.
- Deliverable: A script that will initialize a basics Transformer model with the given hyperpa-
rameters, create a random batch of data, and time forward and backward passes.

- (b) Time the forward and backward passes for the model sizes described in §1.1.2. Use 5 warmup
steps and compute the average and standard deviation of timings over 10 measurement steps.
How long does a forward pass take? How about a backward pass? Do you see high variability
across measurements, or is the standard deviation small?
- Deliverable: A 1-2 sentence response with your timings.

| size   |   batch_size |   seq_len |   d_model |   d_ff |   num_layers |   num_heads |   forward_mean |   forward_var |   backward_mean |   backward_var |   optimizer_mean |   optimizer_var |
|:-------|-------------:|----------:|----------:|-------:|-------------:|------------:|---------------:|--------------:|----------------:|---------------:|-----------------:|----------------:|
| small  |            4 |       256 |       768 |   3072 |           12 |          12 |        21.3158 |     0.0155995 |         30.5026 |     1.10226    |          12.9702 |     0.228053    |
| medium |            4 |       256 |      1024 |   4096 |           24 |          16 |        42.5995 |     0.0689277 |         63.5247 |     0.00878402 |          29.4225 |     0.000941833 |
| large  |            4 |       256 |      1280 |   5120 |           36 |          20 |        71.7948 |     0.0123283 |        145.156  |     0.0736406  |          88.6753 |     0.0005357   |


- (c) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without
the warm-up steps. How does this affect your results? Why do you think this happens? Also try
to run the script with 1 or 2 warm-up steps. Why might the result still be different?
- Deliverable: A 2-3 sentence response.

| size   |   batch_size |   seq_len |   d_model |   d_ff |   num_layers |   num_heads |   forward_mean |   forward_var |   backward_mean |   backward_var |   optimizer_mean |   optimizer_var |
|:-------|-------------:|----------:|----------:|-------:|-------------:|------------:|---------------:|--------------:|----------------:|---------------:|-----------------:|----------------:|
| small  |            4 |       256 |       768 |   3072 |           12 |          12 |        50.7101 |     5595.76   |          49.376 |     3404.91    |          15.3628 |        43.5968  |
| medium |            4 |       256 |      1024 |   4096 |           24 |          16 |        48.268  |       85.0164 |          63.036 |        1.13987 |          28.7369 |         3.28187 |
| large  |            4 |       256 |      1280 |   5120 |           36 |          20 |        80.7635 |      348.808  |         143.522 |       10.3281  |         105.472  |      2546.25    |

| size   |   batch_size |   seq_len |   d_model |   d_ff |   num_layers |   num_heads |   forward_mean |   forward_var |   backward_mean |   backward_var |   optimizer_mean |   optimizer_var |
|:-------|-------------:|----------:|----------:|-------:|-------------:|------------:|---------------:|--------------:|----------------:|---------------:|-----------------:|----------------:|
| small  |            4 |       256 |       768 |   3072 |           12 |          12 |        48.3992 |      5656.98  |         53.1437 |   2793.1       |          12.605  |        0.427401 |
| medium |            4 |       256 |      1024 |   4096 |           24 |          16 |        48.3252 |       105.031 |         63.3671 |      0.0420911 |          28.7181 |        3.55408  |
| large  |            4 |       256 |      1280 |   5120 |           36 |          20 |        80.539  |       327.473 |        143.58   |      9.97313   |         105.22   |     2462.54     |

不进行预热步骤时，首次测量时间显著延长，且测量变异性增大。这是因为GPU需要时间初始化内核、建立执行图并预热缓存。即使使用1-2次预热，结果仍可能不稳定，原因在于某些GPU优化（如内核融合、内存分配）需要多次迭代才能达到稳定状态。充分的预热确保了所有运行时优化完全生效，从而获得一致的性能基准。

## 1.1.4 Nsight Systems Profiler

### 内容概况

该部分详细介绍了如何使用NVIDIA的Nsight Systems性能剖析工具（`nsys`）来深入分析Transformer模型在GPU上的详细性能表现，超越简单的端到端计时，从而定位具体的优化机会。图片内容涵盖了从基本使用命令到高级代码注解（NVTX）等一系列技术。

---

### 要点总结

1.  **剖析的目的：从宏观到微观**
    *   **核心问题**：端到端基准测试只能得到总时间，无法知道时间具体消耗在哪些组件（如前向传播中的自注意力层、线性层等）。
    *   **解决方案**：使用性能剖析器（Profiler）来获取函数级别的详细执行统计数据（如调用次数、平均耗时、累计时间）。

2.  **工具选择：Nsight Systems (`nsys`)**
    *   **为何选择`nsys`**：标准的Python剖析器（如`cProfile`）无法剖析**异步执行**的CUDA内核。`nsys`是NVIDIA官方提供的GPU专用剖析工具。
    *   **基本用法**：在运行命令前加上 `nsys profile -o <输出文件>`。
        *   示例：`uv run nsys profile -o result python benchmark.py`

3.  **核心技术与高级用法：使用NVTX进行代码注解**
    *   **目的**：在剖析结果中自定义标记范围，使时间线更加清晰可读，能直接对应到代码的特定部分（如“自注意力计算”）。
    *   **操作方法**：
        *   **导入库**：`import torch.cuda.nvtx as nvtx`
        *   **注解函数**：使用 `@nvtx.range("描述性名称")` 装饰器来标记一个函数。
        *   **替换实现**：在基准测试脚本中，用注解后的函数替换原来的实现（例如：`cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention`）。
    *   **应用场景**：
        *   忽略预热步骤（通过过滤NVTX行）。
        *   分离前向传播和反向传播。
        *   进一步隔离模型内部组件（如自注意力层中的不同计算阶段）。

4.  **其他实用选项**
    *   **查看结果**：生成的 `.nsys-rep` 文件可在本地使用 **NVIDIA Nsight Systems桌面应用程序**打开并进行可视化分析。
    *   **获取Python调用栈**：使用 `--python-backtrace=cuda` 选项可以关联CUDA API调用和Python代码（可能增加开销）。
    *   **自动注解PyTorch**：使用 `--pytorch` 命令行选项，可以自动为PyTorch的C++ API调用添加NVTX范围，简化基础剖析。

总而言之，这部分内容的核心是引导学生使用专业的GPU性能剖析工具`nsys`，并通过**NVTX代码注解**这一关键技术，将宏观的性能指标精确地映射到微观的代码模块上，从而为后续的针对性优化（如实现更快的自定义内核）提供清晰的数据支持和方向指引。


### Problem (nsys_profile): 5 points

Profile your forward pass, backward pass, and optimizer step using nsys with each of the model
sizes described in Table 1 and context lengths of 128, 256, 512 and 1024 (you may run out of memory
with some of these context lengths for the larger models, in which case just note it in your report).

**24GB RTX 4090**

| size   |   batch_size |   seq_len |   d_model |   d_ff |   num_layers |   num_heads |   forward_mean |   forward_var |   backward_mean |   backward_var |   optimizer_mean |   optimizer_var |
|:-------|-------------:|----------:|----------:|-------:|-------------:|------------:|---------------:|--------------:|----------------:|---------------:|-----------------:|----------------:|
| small  |            4 |       128 |       768 |   3072 |           12 |          12 |        22.2989 |     0.490288  |         32.0786 |     1.8454     |          13.0647 |      0.223345   |
| small  |            4 |       256 |       768 |   3072 |           12 |          12 |        22.1783 |     0.298246  |         32.828  |    13.302      |          12.889  |      0.0117203  |
| small  |            4 |       512 |       768 |   3072 |           12 |          12 |        25.9893 |     0.0213326 |         56.2679 |     0.0154325  |          12.8765 |      0.0209329  |
| small  |            4 |      1024 |       768 |   3072 |           12 |          12 |        78.1306 |     0.0113086 |        163.597  |     0.00313449 |          13.1214 |      0.0179988  |
| medium |            4 |       128 |      1024 |   4096 |           24 |          16 |        44.822  |    20.0731    |         60.8045 |     0.555313   |          30.0259 |      0.846964   |
| medium |            4 |       256 |      1024 |   4096 |           24 |          16 |        44.0857 |     0.767401  |         65.6859 |     6.41022    |          29.7807 |      0.00983881 |
| medium |            4 |       512 |      1024 |   4096 |           24 |          16 |        78.8182 |     0.0129264 |        162.59   |     0.0483767  |          29.6864 |      0.00549085 |
| medium |            4 |      1024 |      1024 |   4096 |           24 |          16 |       nan      |   nan         |        nan      |   nan          |         nan      |    nan          |
| large  |            4 |       128 |      1280 |   5120 |           36 |          20 |        64.7108 |     0.99676   |         93.3342 |     8.27698    |          88.7608 |      0.00564377 |
| large  |            4 |       256 |      1280 |   5120 |           36 |          20 |        72.427  |     0.0991168 |        146.857  |     2.46411    |          88.9511 |      0.00853019 |
| large  |            4 |       512 |      1280 |   5120 |           36 |          20 |       nan      |   nan         |        nan      |   nan          |         nan      |    nan          |
| large  |            4 |      1024 |      1280 |   5120 |           36 |          20 |       nan      |   nan         |        nan      |   nan          |         nan      |    nan          |
| xl     |            4 |       128 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |   nan          |         nan      |    nan          |
| xl     |            4 |       256 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |   nan          |         nan      |    nan          |
| xl     |            4 |       512 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |   nan          |         nan      |    nan          |
| xl     |            4 |      1024 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |   nan          |         nan      |    nan          |
| 2.7B   |            4 |       128 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |   nan          |         nan      |    nan          |
| 2.7B   |            4 |       256 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |   nan          |         nan      |    nan          |
| 2.7B   |            4 |       512 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |   nan          |         nan      |    nan          |
| 2.7B   |            4 |      1024 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |   nan          |         nan      |    nan          |

- (a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

```python
# large seq_len = 256
{'forward': [88.8317299541086, 88.45556201413274, 88.48302403930575, 87.82706502825022, 88.01810094155371, 88.1556150270626, 88.03761599119753, 95.84963601082563, 88.90744601376355, 89.61411099880934], 'forward_mean': np.float64(89.21799060190096), 'forward_var': np.float64(5.139907585944255)}

Name	Start	Duration	TID	Category
step_7	6.6514s	100.412 ms	11157	
forward	6.65153s	94.545 ms	11157	
step_8	6.75183s	93.458 ms	11157	
forward	6.75196s	87.616 ms	11157	
step_9	6.84529s	94.793 ms	11157	
forward	6.84543s	88.328 ms	11157	

nsys统计结果略快
```

- (b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many
times is this kernel invoked during a single forward pass of your model? Is it the same kernel
that takes the most runtime when you do both forward and backward passes? (Hint: look at the
“CUDA GPU Kernel Summary” under “Stats Systems View”, and filter using NVTX ranges to
identify which parts of the model are responsible for which kernels.)

```python
# forward pass
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
60.9%	629.426 ms	2175	289.391 μs	363.905 μs	95.457 μs	745.186 μs	116.811 μs	ampere_sgemm_128x64_tn
14.8%	152.942 ms	540	283.225 μs	282.657 μs	276.769 μs	296.033 μs	3.384 μs	void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(T1::Params)
3.1%	31.571 ms	1080	29.232 μs	29.488 μs	14.528 μs	46.529 μs	13.449 μs	void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3)

# both forward + backward passes
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
27.1%	852.073 ms	2715	313.839 μs	379.136 μs	94.144 μs	760.671 μs	114.211 μs	ampere_sgemm_128x64_tn
19.2%	602.370 ms	2175	276.951 μs	276.383 μs	261.632 μs	582.911 μs	25.507 μs	void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_nt_align1>(T1::Params)
14.5%	456.236 ms	1620	281.627 μs	281.279 μs	272.671 μs	294.079 μs	3.450 μs	void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(T1::Params)
5.9%	185.166 ms	5415	34.195 μs	35.552 μs	4.160 μs	70.271 μs	20.659 μs	void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3)

# both forward + backward passes + optimizer
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
19.1%	836.557 ms	2715	308.124 μs	371.900 μs	94.047 μs	746.808 μs	111.915 μs	ampere_sgemm_128x64_tn
13.6%	596.970 ms	2175	274.468 μs	273.598 μs	258.269 μs	574.106 μs	25.085 μs	void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_nt_align1>(T1::Params)
13.3%	583.574 ms	24848	23.485 μs	17.920 μs	927 ns	114.751 μs	22.151 μs	void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)
11.4%	501.491 ms	25440	19.712 μs	8.736 μs	959 ns	165.310 μs	22.661 μs	void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)
10.4%	454.604 ms	1620	280.619 μs	280.029 μs	270.814 μs	294.045 μs	3.432 μs	void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(T1::Params)
```

**(1) `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_nt_align1>(T1::Params)`**
- **前向传播中未出现**，全流程中占比 **13.6%**（596.970 ms）。
- **特性**：这是 CUTLASS 库的矩阵乘法内核（`nt` 表示非转置×转置矩阵乘法），常用于反向传播中的梯度计算（如权重梯度 `dW` 的计算）。

**(2) 逐元素操作内核**
以下内核在前向传播中占比很低，但在全流程中耗时显著增加，属于反向传播的典型操作：
- **`void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, ...>`**  
  - 全流程占比 **13.3%**（583.574 ms），调用次数 **24,848 次**（远超前向传播的 1,080 次）。  
  - **作用**：可能是梯度逐元素乘法（如激活函数的梯度计算）。  
- **`void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, ...>`**  
  - 全流程占比 **11.4%**（501.491 ms），调用次数 **25,440 次**。  
  - **作用**：梯度累加（如多个分支梯度的求和）。

**(3) 其他可能的反向传播内核**
- **`void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(T1::Params)`**  
  - 前向传播中占比 **14.8%**，全流程中占比 **10.4%**。  
  - **可能性**：可能同时用于前向和反向传播（如梯度传播中的 `dX` 计算）。

- (c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice that
several other kernels still take a non-trivial amount of the overall runtime. What other kernels
besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward
pass?

```python
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
60.9%	629.426 ms	2175	289.391 μs	363.905 μs	95.457 μs	745.186 μs	116.811 μs	ampere_sgemm_128x64_tn
14.8%	152.942 ms	540	283.225 μs	282.657 μs	276.769 μs	296.033 μs	3.384 μs	void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(T1::Params)
3.1%	31.571 ms	1080	29.232 μs	29.488 μs	14.528 μs	46.529 μs	13.449 μs	void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3)
2.8%	28.928 ms	5415	5.342 μs	5.056 μs	4.224 μs	8.032 μs	774 ns	void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
2.7%	27.676 ms	1620	17.083 μs	19.360 μs	11.552 μs	21.088 μs	3.644 μs	void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
2.4%	24.456 ms	540	45.288 μs	44.832 μs	43.136 μs	51.200 μs	1.700 μs	ampere_sgemm_128x128_nn
```

- (d) Profile running one complete training step with your implementation of AdamW (i.e., the forward
pass, computing the loss and running a backward pass, and finally an optimizer step, as you’d do
during training). How does the fraction of time spent on matrix multiplication change, compared
to doing inference (forward pass only)? How about other kernels?
- ampere_sgemm_128x64_tn 下降 elementwise_kernel 上升

- (e) Compare the runtime of the softmax operation versus the matrix multiplication operations within
the self-attention layer of your model during a forward pass. How does the difference in runtimes
compare to the difference in FLOPs?

```python
# forward pass
Name	Start	Duration	TID	Category
forward	6.35478s	88.399 ms	1138	
computing attention scores	6.35609s	145.057 μs	1138	
computing softmax	6.35634s	137.019 μs	1138	
final matmul	6.35648s	146.757 μs	1138	
```
- softmax 的“运行时/FLOPs”比值通常更高，表明其作为一种内存带宽受限操作，计算效率低于计算受限的矩阵乘法。


## 1.1.5 Mixed Precision

### 内容概况

该部分详细阐述了为何以及如何在深度学习模型训练中使用混合精度技术，旨在利用现代GPU的专用硬件（张量核心）来大幅提升训练速度，同时通过特定的技术手段（如`torch.autocast`）来克服低精度数据类型带来的数值稳定性挑战。

---

### 要点总结

1.  **动机：利用硬件优势提升速度**
    *   **硬件基础**：现代NVIDIA GPU（如A100）配备了**张量核心**，这些专用核心在低精度（如FP16/BF16）下的计算吞吐量远高于标准FP32精度（例如：312 TFLOPS vs 19.5 TFLOPS）。
    *   **核心目标**：通过将模型参数和计算转换为低精度数据类型，可以**显著加速训练和推理过程**。

2.  **挑战：直接使用低精度的风险**
    *   **梯度下溢**：FP16的数值范围有限，许多实际训练中的梯度值过小，在FP16中会变为零，导致训练失败。
    *   **数值溢出**：FP16的动态范围较小，容易在计算中产生溢出，表现为损失值变成NaN。
    *   **性能损失**：虽然BF16具有与FP32相同的动态范围，数值上更稳定，但直接使用仍可能影响模型的最终性能。

3.  **解决方案：混合精度训练**
    *   **核心思想**：扬长避短。让计算密集的操作（如矩阵乘法）在**低精度**下运行以利用速度优势，而让对数值精度敏感的操作（如累加、归约）保持在**高精度**下以确保稳定性。
    *   **PyTorch实现**：使用 **`torch.autocast`** 上下文管理器。在该上下文中，PyTorch会自动为适用的操作选择指定的低精度数据类型。
        *   示例：`with torch.autocast(device="cuda", dtype=torch.float16): y = model(x)`
    *   **最佳实践**：即使参与计算的张量被转换为低精度，也建议将**累加操作保留在FP32精度**下进行。

总而言之，这部分内容清晰地指出了从FP32转向混合精度的**必要性（速度）**、**挑战（稳定性）** 和**标准实践方法（PyTorch的`autocast`）**，为学生实际实现混合精度训练提供了理论指导和代码示例。

### Problem (mixed_precision_accumulation): 1 point

Run the following code and commment on the (accuracy of the) results

```python
def mixed_precision_accumulation():
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)
    
    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)
    
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)
    
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)

# Result
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)
```

即使累加器使用FP32，若中间值以FP16存储，误差仍会累积。这凸显了在混合精度训练中，使用FP32作为累加器对保持数值稳定性的必要性。

### Problem (benchmarking_mixed_precision): 2 points

- (a) Consider the following model:

```python
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None=None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False, device=device)
        self.ln = nn.LayerNorm(10, device=device)
        self.fc2 = nn.Linear(10, out_features, bias=False, device=device)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.ln(x2)
        x4 = self.fc2(x3)
        return (x1, x2, x3, x4)
```

- Suppose we are training the model on a GPU and that the model parameters are originally in
FP32. We’d like to use autocasting mixed precision with FP16. What are the data types of:
    - the model parameters within the autocast context,
    - the output of the first feed-forward layer (ToyModel.fc1),
    - the output of layer norm (ToyModel.ln),
    - the model’s predicted logits,
    - the loss,
    - and the model’s gradients?

```python
fc1.weight: torch.float32
fc1.weight.grad: torch.float32
fc1_output: torch.float16
ln.weight: torch.float32
ln.weight.grad: torch.float32
ln_output: torch.float32
fc2.weight: torch.float32
fc2.weight.grad: torch.float32
fc2_output: torch.float16
loss: torch.float16
```

​​计算路径（如前向传播中的线性层）使用 FP16 加速，而对数值敏感的操作（如归一化、损失计算）和模型参数本身则保留在 FP32 上。​​


- (b) You should have seen that FP16 mixed precision autocasting treats the layer normalization layer
differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed
precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently?
Why or why not?

```python
fc1.weight: torch.float32
fc1.weight.grad: torch.float32
fc1_output: torch.bfloat16
ln.weight: torch.float32
ln.weight.grad: torch.float32
ln_output: torch.float32
fc2.weight: torch.float32
fc2.weight.grad: torch.float32
fc2_output: torch.bfloat16
loss: torch.bfloat16
```

- (c) Modify your benchmarking script to optionally run the model using mixed precision with BF16.
Time the forward and backward passes with and without mixed-precision for each language model
size described in §1.1.2. Compare the results of using full vs. mixed precision, and comment on
any trends as model size changes. You may find the nullcontext no-op context manager to be
useful

**24GB RTX 4090 F32**

| size   |   batch_size |   seq_len |   d_model |   d_ff |   num_layers |   num_heads |   forward_mean |   forward_var |   backward_mean |   backward_var |   optimizer_mean |   optimizer_var |
|:-------|-------------:|----------:|----------:|-------:|-------------:|------------:|---------------:|--------------:|----------------:|---------------:|-----------------:|----------------:|
| small  |            4 |       128 |       768 |   3072 |           12 |          12 |        23.4019 |     0.715428  |         33.2361 |      4.90607   |          14.6058 |       1.55479   |
| small  |            4 |       256 |       768 |   3072 |           12 |          12 |        24.1165 |     3.51745   |         32.908  |      2.87235   |          14.8995 |       2.19896   |
| small  |            4 |       512 |       768 |   3072 |           12 |          12 |        31.4861 |    33.1499    |         57.2082 |      0.18824   |          16.857  |      13.0927    |
| small  |            4 |      1024 |       768 |   3072 |           12 |          12 |        79.0697 |     0.119345  |        164.192  |      0.0447476 |          15.9431 |       4.34925   |
| medium |            4 |       128 |      1024 |   4096 |           24 |          16 |        45.5267 |     1.07442   |         69.7873 |      9.92401   |          29.9034 |       0.1966    |
| medium |            4 |       256 |      1024 |   4096 |           24 |          16 |        45.698  |     2.72527   |         76.1795 |    158.866     |          29.8369 |       0.0233066 |
| medium |            4 |       512 |      1024 |   4096 |           24 |          16 |        79.8016 |     0.0900261 |        164.043  |      0.150251  |          30.0881 |       0.0532259 |
| medium |            4 |      1024 |      1024 |   4096 |           24 |          16 |       nan      |   nan         |        nan      |    nan         |         nan      |     nan         |
| large  |            4 |       128 |      1280 |   5120 |           36 |          20 |        73.7877 |    11.9195    |        110.211  |     58.5238    |          89.2372 |       0.0160839 |
| large  |            4 |       256 |      1280 |   5120 |           36 |          20 |        76.228  |     8.40086   |        148.668  |      6.76378   |          89.4336 |       0.0210026 |
| large  |            4 |       512 |      1280 |   5120 |           36 |          20 |       nan      |   nan         |        nan      |    nan         |         nan      |     nan         |
| large  |            4 |      1024 |      1280 |   5120 |           36 |          20 |       nan      |   nan         |        nan      |    nan         |         nan      |     nan         |
| xl     |            4 |       128 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |    nan         |         nan      |     nan         |
| xl     |            4 |       256 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |    nan         |         nan      |     nan         |
| xl     |            4 |       512 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |    nan         |         nan      |     nan         |
| xl     |            4 |      1024 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |    nan         |         nan      |     nan         |
| 2.7B   |            4 |       128 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |    nan         |         nan      |     nan         |
| 2.7B   |            4 |       256 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |    nan         |         nan      |     nan         |
| 2.7B   |            4 |       512 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |    nan         |         nan      |     nan         |
| 2.7B   |            4 |      1024 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |    nan         |         nan      |     nan         |

**24GB RTX 4090 BF16**

| size   |   batch_size |   seq_len |   d_model |   d_ff |   num_layers |   num_heads |   forward_mean |   forward_var |   backward_mean |   backward_var |   optimizer_mean |   optimizer_var |
|:-------|-------------:|----------:|----------:|-------:|-------------:|------------:|---------------:|--------------:|----------------:|---------------:|-----------------:|----------------:|
| small  |            4 |       128 |       768 |   3072 |           12 |          12 |        25.8433 |     3.23157   |         34.8656 |      4.10286   |          13.0918 |      1.42804    |
| small  |            4 |       256 |       768 |   3072 |           12 |          12 |        25.6884 |     5.38257   |         34.7039 |      4.20348   |          12.6564 |      0.357178   |
| small  |            4 |       512 |       768 |   3072 |           12 |          12 |        25.866  |     2.25236   |         34.9353 |      0.257959  |          12.6695 |      0.135379   |
| small  |            4 |      1024 |       768 |   3072 |           12 |          12 |        50.1621 |     0.0115309 |        110.477  |      0.0069567 |          13.3875 |      3.75151    |
| medium |            4 |       128 |      1024 |   4096 |           24 |          16 |        50.2021 |     2.82307   |         66.6406 |      9.51961   |          29.7321 |      0.178686   |
| medium |            4 |       256 |      1024 |   4096 |           24 |          16 |        54.0458 |    95.8586    |         71.9505 |     12.5784    |          31.2455 |     21.6072     |
| medium |            4 |       512 |      1024 |   4096 |           24 |          16 |        50.2522 |     0.555699  |         88.4807 |      0.0222089 |          29.6949 |      0.00412734 |
| medium |            4 |      1024 |      1024 |   4096 |           24 |          16 |       nan      |   nan         |        nan      |    nan         |         nan      |    nan          |
| large  |            4 |       128 |      1280 |   5120 |           36 |          20 |        75.5397 |     0.741425  |         99.9751 |      1.18677   |          88.8969 |      0.00274986 |
| large  |            4 |       256 |      1280 |   5120 |           36 |          20 |        74.9646 |     1.79119   |        101.323  |      0.315989  |          88.8139 |      0.00218763 |
| large  |            4 |       512 |      1280 |   5120 |           36 |          20 |       nan      |   nan         |        nan      |    nan         |         nan      |    nan          |
| large  |            4 |      1024 |      1280 |   5120 |           36 |          20 |       nan      |   nan         |        nan      |    nan         |         nan      |    nan          |
| xl     |            4 |       128 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |    nan         |         nan      |    nan          |
| xl     |            4 |       256 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |    nan         |         nan      |    nan          |
| xl     |            4 |       512 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |    nan         |         nan      |    nan          |
| xl     |            4 |      1024 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |    nan         |         nan      |    nan          |
| 2.7B   |            4 |       128 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |    nan         |         nan      |    nan          |
| 2.7B   |            4 |       256 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |    nan         |         nan      |    nan          |
| 2.7B   |            4 |       512 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |    nan         |         nan      |    nan          |
| 2.7B   |            4 |      1024 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |    nan         |         nan      |    nan          |

随着模型规模增大（从 "small" 到 "2.7B"），​​BF16 带来的加速效果越明显​​。这是因为更大的模型包含更多的计算密集型操作（如矩阵乘法），这些操作能充分利用 BF16 在 Tensor Cores 上的高吞吐量。

## 1.1.6 Profiling Memory

### 内容概况
该部分将关注点从计算性能转向另一个关键资源——内存。它简要介绍了如何使用 PyTorch 内置的强大内存分析器来追踪语言模型训练和推理过程中的内存分配情况，并提供了一个具体的代码修改示例。

---

### 要点总结

1.  **剖析目标的转变：从计算到内存**
    *   明确指出性能分析不仅限于计算速度（Compute Performance），**内存（Memory）** 是大型语言模型训练和推理中的另一个主要资源瓶颈和优化方向。

2.  **工具介绍：PyTorch 内存分析器**
    *   PyTorch 自带了一个功能强大的内存分析器，能够**随时间推移跟踪内存的分配情况**。

3.  **核心操作流程（三步骤）**
    代码示例清晰地展示了在基准测试脚本中集成内存剖析的三个关键步骤：
    1.  **开始记录**：在需要剖析的代码段**之前**，调用 `torch.cuda.memory._record_memory_history(max_entries=1000000)` 开始记录内存分配历史。`max_entries` 参数设置了记录的最大事件数，以防止内存过度消耗。
    2.  **保存快照**：在目标代码段执行**之后**，调用 `torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")` 将内存分配历史保存到一个快照文件（如图中的 `memory_snapshot.pickle`）。
    3.  **停止记录**：调用 `torch.cuda.memory._record_memory_history(enabled=None)` 来停止记录。

4.  **结果分析方式**
    *   生成的快照文件（`.pickle` 文件）需要被加载到 [PyTorch 的在线可视化工具](https://docs.pytorch.org/memory_viz) 中进行查看和分析，而非直接阅读。


### Problem (memory_profiling): 4 points

Profile your forward pass, backward pass, and optimizer step of the 2.7B model from Table 1 with
context lengths of 128, 256 and 512.
- (a) Add an option to your profiling script to run your model through the memory profiler. It may
be helpful to reuse some of your previous infrastructure (e.g., to activate mixed-precision, load
specific model sizes, etc). Then, run your script to get a memory profile of the 2.7B model when
either doing inference only (just forward pass) or a full training step. How do your memory
timelines look like? Can you tell which stage is running based on the peaks you see?
- Deliverable: Two images of the “Active memory timeline” of a 2.7B model, from the memory_viz
tool: one for the forward pass, and one for running a full training step (forward and backward
passes, then optimizer step), and a 2-3 sentence response.

```python
# large seq_len = 256
# forward passes 
/public/cs336/assignment1-basics/cs336_basics/module.py:359:forward
forward阶段最后y = self.lm_head(y)

# both forward + backward passes
# Loss计算
/public/cs336/assignment1-basics/cs336_basics/optim.py:22:cross_entropy_func
# 启动反向传播梯度计算
??:0:torch::autograd::python::PythonEngine::thread_init(...)
??:0:torch::autograd::Engine::thread_main(...)
??:0:torch::autograd::Engine::evaluate_function(...)
```

- (b) What is the peak memory usage of each context length when doing a forward pass? What about
when doing a full training step? 

```python
# large seq_len = 256
# step1
Total memory used after allocation: 21.2GiB (22771642880 bytes)
# step10
Total memory used after allocation: 21.2GiB (22771642880 bytes)
```

- (c) Find the peak memory usage of the 2.7B model when using mixed-precision, for both a forward
pass and a full optimizer step. Does mixed-precision significantly affect memory usage?

```python
# large seq_len = 256
# 采用混合精度，不会显著降低内存申请
Total memory used after allocation: 20.4GiB (21911654912 bytes)
```

- (d) Consider the 2.7B model. At our reference hyperparameters, what is the size of a tensor of
activations in the Transformer residual stream, in single-precision? Give this size in MB (i.e.,
divide the number of bytes by $1024^2$ ).

```python
# large seq_len = 256
batchsize * seq_len * d_model = 4 * 256 * 1280 * 4 = 5MB
```


- (e) Now look closely at the “Active Memory Timeline” from pytorch.org/memory_viz of a memory
snapshot of the 2.7B model doing a forward pass. When you reduce the “Detail” level, the tool
hides the smallest allocations to the corresponding level (e.g., putting “Detail” at 10% only shows 
the 10% largest allocations). What is the size of the largest allocations shown? Looking through
the stack trace, can you tell where those allocations come from?
- scaled_dot_product_attention_func
```python
# forward passes
# 模型创建内存申请
/public/cs336/assignment2-systems/cs336_systems/benchemark.py:106:build_model
```