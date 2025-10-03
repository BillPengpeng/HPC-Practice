本文主要整理CS336 Kernels, Triton章节的主要内容。

## 1 - ​Hardware and execution interact

![Wave](https://developer-blogs.nvidia.com/wp-content/uploads/2019/06/pasted-image-0.png)

### 内容概括

这段文字描述了CUDA编程中一个称为“**Wave Quantization**”（波量化）的性能优化问题及其解决方案。核心问题是当线程块（Thread Block）总数不能整除流多处理器（SM）数量时，最后一波（Wave）调度会因线程块不足而导致部分SM空闲，从而降低计算 occupancy（占用率）。为此，提出了一个经验法则：创建足够多的线程块（至少为SM数量的4倍）来“淹没”硬件，以隐藏此调度细节并最大化硬件利用率。

---

### 要点总结

1.  **核心问题：低占用率 (Low Occupancy)**
    *   **现象**：GPU将线程块分批（Wave）调度到各个SM上执行。如果总的线程块数量不是SM数量的整数倍，最后一批调度的线程块数量会少于SM数，导致一部分SM没有任务可执行而空闲。
    *   **结果**：硬件计算资源未被充分利用，整体性能下降。

2.  **解决方案：波量化 (Wave Quantization)**
    *   **思路**：通过调整线程格（Grid）的规模，使其线程块总数是SM数量的整数倍，从而确保每一波调度都能让所有SM满载。

3.  **重要经验法则 (Rule of Thumb)**
    *   **具体方法**：建议创建的**线程块总数应至少是GPU SM总数的4倍或更多**。
    *   **目的**：
        *   确保有足够多的线程块来填满所有SM的多波调度，避免最后一波资源闲置的问题。
        *   提供充足的并行任务来隐藏内存访问延迟等其他延迟，进一步提升性能。

4.  **根本挑战 (Underlying Challenge)**
    *   硬件的一些关键细节（如**SM的具体数量**和**线程块的确切调度策略**）对CUDA执行模型是**隐藏的**（或说是抽象的）。
    *   这意味着程序员无法在代码中直接针对特定数量的SM进行精确的线程块分配，因此需要一个普适性的、保守的策略（即上述经验法则）来保证程序在不同型号的GPU上都能获得良好性能。

**总结而言**：为了规避GPU硬件调度细节带来的性能损失，最佳实践是启动远超SM数量的线程块（例如 ≥ 4x #SMs），以确保高占用率和稳定的高性能。

## 2 - 最基本benchmark

```python
def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: list[float] = [] # @inspect times, @inspect description
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()

        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

        end_time = time.time()
        times.append((end_time - start_time) * 1000) # @inspect times

    mean_time = mean(times) # @inspect mean_time
    return mean_time
```

这是一个用于性能基准测试的通用函数，特别针对 **PyTorch CUDA 操作**进行了优化，能够准确测量 GPU 和 CPU 代码的执行时间。

### 参数说明

```python
def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
```

- **`description`**: 测试描述，用于标识不同的测试场景
- **`run`**: 要测试的可调用函数/方法
- **`num_warmups`**: 预热次数，默认为1次
- **`num_trials`**: 正式测试次数，默认为3次

### 核心执行流程

#### 1. 预热阶段 (Warmup)
```python
for _ in range(num_warmups):
    run()
if torch.cuda.is_available():
    torch.cuda.synchronize()
```
**目的**: 消除首次执行的额外开销：
- JIT 编译时间（如 PyTorch 的图编译）
- CUDA 内核编译和加载
- 缓存预热
- 内存分配初始化

#### 2. 正式测试阶段
```python
times: list[float] = []
for trial in range(num_trials):
    start_time = time.time()
    
    run()  # 执行被测代码
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 关键：等待GPU完成
    
    end_time = time.time()
    times.append((end_time - start_time) * 1000)  # 转换为毫秒
```
**关键特性**:
- 多次测试获取稳定结果
- 使用 `time.time()` 获取墙钟时间
- **`torch.cuda.synchronize()` 确保 GPU 操作完成**

#### 3. 结果计算
```python
mean_time = mean(times)
return mean_time
```
返回多次测试的平均时间（毫秒）

### 关键技术点

#### 🔥 **CUDA 同步的重要性**
```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
```
- **原因**: CUDA 操作是**异步的** - CPU 发出指令后立即返回，不等待 GPU 完成
- **解决方案**: `synchronize()` 阻塞直到所有 GPU 操作完成，确保时间测量的准确性

#### 📊 **多次测试的意义**
- 捕获性能方差（JIT、缓存效应、系统负载）
- 提供更稳定的平均性能数据
- 默认3次在精度和效率间取得平衡

#### ⚡ **预热机制的价值**
- 消除一次性开销（编译、初始化）
- 反映"稳态"性能而非"冷启动"性能

### 使用示例

```python
# 测试矩阵乘法性能
def test_matmul():
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    return a @ b

time_ms = benchmark("1000x1000 Matrix Multiplication", test_matmul)
print(f"平均执行时间: {time_ms:.2f} ms")
```

## 3 - torch.profiler.profile

```python
def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Run the code with the profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # Output stack trace for visualization
            with_stack=with_stack,
            # Needed to export stack trace for visualization
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Print out table
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
    #text(f"## {description}")
    #text(table, verbatim=True)

    # Write stack trace visualization
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")

    return table
```

我来为您详细解析这段 `profile` 函数的源码，这是一个专业的 PyTorch 性能分析工具。

### 函数概述

这是一个用于 **深度性能分析** 的函数，使用 PyTorch 的官方性能分析器来获取 CPU 和 GPU 的详细执行信息，特别适合优化 CUDA 代码。

```python
def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
```

- **`description`**: 测试描述，用于标识和保存分析结果
- **`run`**: 要分析的可调用函数/方法
- **`num_warmups`**: 预热次数，默认为1次
- **`with_stack`**: 是否收集调用栈信息，用于生成可视化图表

### 核心执行流程

#### 1. 预热阶段 (Warmup)
```python
for _ in range(num_warmups):
    run()
if torch.cuda.is_available():
    torch.cuda.synchronize()
```
**目的**: 与 benchmark 函数相同，消除首次执行的编译和初始化开销。

#### 2. 性能分析阶段
```python
with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=with_stack,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
    run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
```
**关键配置**:
- **`activities`**: 同时监控 CPU 和 CUDA 活动
- **`with_stack`**: 启用调用栈跟踪（增加开销但提供更多信息）
- **`experimental_config`**: 启用详细配置以支持栈导出

#### 3. 结果处理与输出
```python
table = prof.key_averages().table(sort_by="cuda_time_total",
                                  max_name_column_width=80,
                                  row_limit=10)
```
生成排序的性能数据表格，按 CUDA 总时间排序，显示前10个最耗时的操作。

#### 4. 高级功能：调用栈可视化
```python
if with_stack:
    text_path = f"var/stacks_{description}.txt"
    svg_path = f"var/stacks_{description}.svg"
    prof.export_stacks(text_path, "self_cuda_time_total")
```
**生成两种格式的输出**：
- 文本文件：原始栈信息
- SVG 文件：可视化调用图（需要额外工具转换）

## 4 - NVTX示例

```python
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        # Mark the entire forward pass
        for i, layer in enumerate(self.layers):
            # Mark each layer's computation separately
            with nvtx.range(f"layer_{i}"):
                x = layer(x)
                x = torch.nn.functional.gelu(x)
        
        return x

def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int, use_optimizer: bool = False):
    """Run forward and backward passes through an MLP.
    
    Args:
        dim: Dimension of each layer
        num_layers: Number of linear+GeLU layers
        batch_size: Number of samples to process at once
        num_steps: Number of forward/backward iterations
        use_optimizer: Whether to use Adam optimizer for weight updates
    """
    # Define a model (with random weights)
    with nvtx.range("define_model"):
        model = MLP(dim, num_layers).to(get_device())
    
    # Initialize optimizer if requested
    optimizer = torch.optim.Adam(model.parameters()) if use_optimizer else None

    # Define an input (random)
    with nvtx.range("define_input"):
        x = torch.randn(batch_size, dim, device=get_device())

    # Run the model `num_steps` times
    for step in range(num_steps):
        if step > 10:
            # start profiling after 10 warmup iterations
            torch.cuda.cudart().cudaProfilerStart()

        nvtx.range_push(f"step_{step}")
        
        # Zero gradients
        if use_optimizer:
            optimizer.zero_grad()
        else:
            model.zero_grad(set_to_none=True)

        # Forward
        with nvtx.range("forward"):
            y = model(x).mean()

        # Backward
        with nvtx.range("backward"):
            y.backward()

        # Optimizer step if enabled
        if use_optimizer:
            with nvtx.range("optimizer_step"):
                #print(f"Step {step}, loss: {y.item():.6f}")
                optimizer.step()
        
        nvtx.range_pop()
```

这是一个使用 **NVTX (NVIDIA Tools Extension)** 进行深度性能分析的 MLP 训练脚本，专门为 GPU 性能优化设计。

### 核心组件分析

#### 1. 设备管理函数
```python
def get_device(index: int = 0) -> torch.device:
    """智能选择设备：优先GPU，后备CPU"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")
```
**特点**：支持多GPU，提供优雅降级到CPU。

#### 2. MLP 模型定义
```python
class MLP(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            with nvtx.range(f"layer_{i}"):  # 每层都有NVTX标记
                x = layer(x)
                x = torch.nn.functional.gelu(x)
        return x
```
**关键特性**：
- 每层线性变换 + GeLU 激活
- **每层都有独立的 NVTX 范围标记**，便于精细分析

## 核心训练循环分析

#### 3. 训练函数 `run_mlp`
```python
def run_mlp(dim: int, num_layers: int, batch_size: int, 
           num_steps: int, use_optimizer: bool = False):
```

##### 初始化阶段
```python
# 模型初始化（带NVTX标记）
with nvtx.range("define_model"):
    model = MLP(dim, num_layers).to(get_device())

# 优化器条件初始化
optimizer = torch.optim.Adam(model.parameters()) if use_optimizer else None

# 输入数据准备（带NVTX标记）
with nvtx.range("define_input"):
    x = torch.randn(batch_size, dim, device=get_device())
```

##### 训练循环（核心）
```python
for step in range(num_steps):
    if step > 10:
        # 10次预热后开始正式性能分析
        torch.cuda.cudart().cudaProfilerStart()
    
    nvtx.range_push(f"step_{step}")  # 标记整个训练步骤
```

##### 梯度管理
```python
# 两种梯度清零方式
if use_optimizer:
    optimizer.zero_grad()  # 优化器内置清零
else:
    model.zero_grad(set_to_none=True)  # 手动清零，更高效
```

##### 前向传播
```python
with nvtx.range("forward"):
    y = model(x).mean()  # 前向计算并取均值作为loss
```

##### 反向传播
```python
with nvtx.range("backward"):
    y.backward()  # 自动求导
```

##### 参数更新
```python
if use_optimizer:
    with nvtx.range("optimizer_step"):
        optimizer.step()  # Adam优化器更新权重
```

##### 循环结束
```python
nvtx.range_pop()  # 结束当前step的范围
```

### NVTX 标记策略分析

#### 分层标记体系
1. **最外层**: `step_{i}` - 整个训练迭代
2. **中间层**: `forward`, `backward`, `optimizer_step` - 主要阶段
3. **最内层**: `layer_{i}` - 每个神经网络层

#### 标记类型
```python
# 方式1：上下文管理器（推荐）
with nvtx.range("name"):
    # 代码块

# 方式2：手动push/pop
nvtx.range_push("name")
# 代码
nvtx.range_pop()
```

### 性能分析特性

#### 预热机制
```python
if step > 10:
    torch.cuda.cudart().cudaProfilerStart()
```
**目的**：跳过前10次迭代，避免编译、初始化等一次性开销影响性能分析。

#### 两种训练模式
- **`use_optimizer=True`**: 完整训练（前向+反向+优化）
- **`use_optimizer=False`**: 仅前向和反向传播，用于分析计算性能

### 使用示例

```python
# 完整训练分析
run_mlp(dim=1024, num_layers=8, batch_size=128, 
        num_steps=100, use_optimizer=True)

# 仅计算性能分析（无优化器）
run_mlp(dim=1024, num_layers=8, batch_size=128,
        num_steps=100, use_optimizer=False)
```

### 与 Nsight Systems 配合使用

#### 采集数据
```bash
# 使用Nsight Systems运行此脚本
nsys profile -o mlp_profile \
  --trace=cuda,nvtx \
  python this_script.py
```

#### 在Nsight中看到的层次结构
```
├── step_11
│   ├── forward
│   │   ├── layer_0
│   │   ├── layer_1
│   │   └── ...
│   ├── backward
│   └── optimizer_step
├── step_12
└── ...
```

### 性能优化价值

1. **精确计时**: 识别每层、每个操作的时间消耗
2. **瓶颈定位**: 找到最耗时的层或操作
3. **对比分析**: 比较不同配置的性能差异
4. **负载均衡**: 分析多GPU情况下的计算分布

### 改进建议

1. **添加内存统计**:
```python
torch.cuda.reset_peak_memory_stats()
# ...运行代码...
memory_used = torch.cuda.max_memory_allocated()
```

2. **添加学习率调度器支持**
3. **支持真实数据集加载**
4. **添加多GPU分布式训练支持**

这个代码是 **GPU性能分析的完美范例**，通过细致的 NVTX 标记，可以在 Nsight Systems 中获得极其详细的性能视图。