本文主要整理CS336 Lecture 8 Parallelism 章节的主要内容。

## 1 - collective_operations_main

```python
def setup(rank: int, world_size: int):
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def collective_operations_main(rank: int, world_size: int):
    """This function is running asynchronously for each process (rank = 0, ..., world_size - 1)."""
    setup(rank, world_size)

    # All-reduce
    dist.barrier()  # Waits for all processes to get to this point (in this case, for print statements)

    tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank  # Both input and output

    print(f"Rank {rank} [before all-reduce]: {tensor}", flush=True)
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
    print(f"Rank {rank} [after all-reduce]: {tensor}", flush=True)

    # Reduce-scatter
    dist.barrier()

    input = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank  # Input
    output = torch.empty(1, device=get_device(rank))  # Allocate output

    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)

    # All-gather
    dist.barrier()

    input = output  # Input is the output of reduce-scatter
    output = torch.empty(world_size, device=get_device(rank))  # Allocate output

    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)

    text("Indeed, all-reduce = reduce-scatter + all-gather!")

    cleanup()
```

### 1. 初始化和设置

```python
def collective_operations_main(rank: int, world_size: int):
    """This function is running asynchronously for each process (rank = 0, ..., world_size - 1)."""
    setup(rank, world_size)
```
*   `rank`: 当前进程的标识符（从 `0` 到 `world_size - 1`）。
*   `world_size`: 参与通信的进程总数。
*   `setup(rank, world_size)`: 这是一个假设的初始化函数，它通常会包含 `dist.init_process_group(...)`，用于建立进程组并让所有进程知道彼此的存在。这是所有分布式操作的前提。

---

### 2. All-Reduce 操作

```python
    # All-reduce
    dist.barrier()  # Waits for all processes to get to this point (in this case, for print statements)
```
*   `dist.barrier()`: 设置一个屏障。**所有进程**必须都执行到这一行代码时，才能继续向下执行。这确保了我们的打印输出不会交错混乱，便于观察结果。

```python
    tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank  # Both input and output
```
*   在每个 rank 上创建一个张量。假设 `world_size=2`：
    *   **Rank 0**: `tensor([0., 1., 2., 3.]) + 0 = [0., 1., 2., 3.]`
    *   **Rank 1**: `tensor([0., 1., 2., 3.]) + 1 = [1., 2., 3., 4.]`
*   `get_device(rank)` 是一个假设的函数，例如在 CUDA 环境中，它可能返回 `f'cuda:{rank}'`，确保每个进程的张量位于其对应的 GPU 上。

```python
    print(f"Rank {rank} [before all-reduce]: {tensor}", flush=True)
```
*   打印执行 all-reduce 之前的张量。`flush=True` 确保输出立即显示。

```python
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
```
*   **`dist.all_reduce`**: 这是核心的集体通信操作。
    *   **功能**: 对所有进程中的 `tensor` 进行某种操作（这里是 `SUM`），然后**将结果写回每个进程的 `tensor`**。
    *   **操作 (op)**: `dist.ReduceOp.SUM` 表示求和。也可以是 `PRODUCT`, `MIN`, `MAX`, `AVG` 等。
    *   **原地修改 (in-place)**: `tensor=` 参数既是输入也是输出，操作完成后，原始 `tensor` 会被覆盖。
    *   **异步 (async_op)**: `async_op=False` 表示这是一个阻塞操作。函数会一直等待，直到所有进程都完成这个 all-reduce 操作后才返回。如果设为 `True`，函数会立即返回一个 `Work` 对象，你需要手动调用 `wait()` 来等待操作完成。
*   **执行结果**:
    *   **Rank 0 的新 tensor**: `[0., 1., 2., 3.]` + `[1., 2., 3., 4.]` = `[1., 3., 5., 7.]`
    *   **Rank 1 的新 tensor**: 同样也是 `[1., 3., 5., 7.]`
*   All-Reduce 的通信模式如下图所示：
    

```python
    print(f"Rank {rank} [after all-reduce]: {tensor}", flush=True)
```
*   打印 all-reduce 之后的结果。所有 rank 的输出将会是完全一样的 `[1., 3., 5., 7.]`。

---

### 3. Reduce-Scatter 操作

```python
    # Reduce-scatter
    dist.barrier()
```
*   再次设置屏障，确保 all-reduce 阶段的所有打印完成后再开始 reduce-scatter。

```python
    input = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank  # Input
```
*   创建 reduce-scatter 的输入张量。假设 `world_size=2`：
    *   **Rank 0**: `[0., 1.] + 0 = [0., 1.]`
    *   **Rank 1**: `[0., 1.] + 1 = [1., 2.]`

```python
    output = torch.empty(1, device=get_device(rank))  # Allocate output
```
*   为 reduce-scatter 的输出分配空间。每个 rank 只接收结果的一部分，所以 `output` 的大小是 `1`。

```python
    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
```
*   打印操作前的输入和（未初始化的）输出。

```python
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
```
*   **`dist.reduce_scatter_tensor`**: 这是另一个核心操作。
    *   **功能**: 分为两步：
        1.  **Reduce**: 对所有进程的 `input` 张量进行按元素操作（如 `SUM`）。假设 `world_size=2`，则虚拟的全局 reduce 结果为 `[0.+1., 1.+2.] = [1., 3.]`。
        2.  **Scatter**: 将这个全局结果**散射**到各个进程。默认情况下，Rank i 会接收到结果张量的第 i 个块（chunk）。因为我们的 `output` 大小是 1，所以每个 rank 接收 1 个元素。
            *   **Rank 0** 接收第 0 个元素：`[1.]`
            *   **Rank 1** 接收第 1 个元素：`[3.]`
*   Reduce-Scatter 的通信模式如下图所示：
    

```python
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)
```
*   打印操作后的结果。注意 `input` 保持不变，`output` 被写入。
    *   **Rank 0**: `input = [0., 1.], output = [1.]`
    *   **Rank 1**: `input = [1., 2.], output = [3.]`

---

### 4. All-Gather 操作

```python
    # All-gather
    dist.barrier()
```
*   进入 all-gather 阶段的屏障。

```python
    input = output  # Input is the output of reduce-scatter
```
*   将 reduce-scatter 的输出作为 all-gather 的输入。
    *   **Rank 0**: `input = [1.]`
    *   **Rank 1**: `input = [3.]`

```python
    output = torch.empty(world_size, device=get_device(rank))  # Allocate output
```
*   为 all-gather 的输出分配空间。每个 rank 都会得到完整的全局结果，所以 `output` 的大小是 `world_size`（2）。

```python
    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
```
*   打印操作前的输入和输出。

```python
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
```
*   **`dist.all_gather_into_tensor`**: 这是第三个核心操作。
    *   **功能**: **收集**所有进程的 `input` 张量，并将其拼接起来，然后**将完整的拼接结果发送给所有进程**。
    *   每个 rank 提供自己的 `input`。
    *   操作完成后，所有 rank 的 `output` 都是 `[input_of_rank0, input_of_rank1, ..., input_of_rankN]`。
    *   本例中，所有 rank 的 `output` 都将变为 `[1., 3.]`。
*   All-Gather 的通信模式如下图所示：
    

```python
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)
```
*   打印操作后的结果。所有 rank 的输出将会是完全一样的 `[1., 3.]`。

---

### 5. 关键结论和清理

```python
    text("Indeed, all-reduce = reduce-scatter + all-gather!")
```
*   这行代码揭示了一个非常重要的概念。
    *   **All-Reduce** 的效果可以分解为两步：
        1.  **Reduce-Scatter**: 将全局求和的结果分散到各个进程中。
        2.  **All-Gather**: 让所有进程都获得分散后的完整结果。
    *   在很多高性能实现中，All-Reduce 算法确实被优化为等价于一次 Reduce-Scatter 加上一次 All-Gather，这通常比朴素的实现更高效。

```python
    cleanup()
```
*   这是一个假设的清理函数，通常会包含 `dist.destroy_process_group()`，用于优雅地关闭分布式环境。

### 总结

这段代码是一个非常好的教学示例，它清晰地展示了：

1.  **All-Reduce**: 全局聚合，结果全知。
2.  **Reduce-Scatter**: 全局聚合，结果分散。
3.  **All-Gather**: 分散收集，结果全知。

并且通过组合后两者（Reduce-Scatter + All-Gather）实现了与 All-Reduce 完全相同的最终效果，深刻地揭示了这些集体通信操作之间的内在联系。

## 2 - all_reduce & reduce_scatter

```python
def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # Create tensor
    tensor = torch.randn(num_elements, device=get_device(rank))

    # Warmup
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kernels to finish
        dist.barrier()            # Wait for all the processes to get here

    # Perform all-reduce
    start_time = time.time()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kernels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # Measure the effective bandwidth
    dist.barrier()
    size_bytes = tensor.element_size() * tensor.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send input and receive output
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    cleanup()


def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # Create input and outputs
    input = torch.randn(world_size, num_elements, device=get_device(rank))  # Each rank has a matrix
    output = torch.empty(num_elements, device=get_device(rank))

    # Warmup
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here

    # Perform reduce-scatter
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # Measure the effective bandwidth
    dist.barrier()
    data_bytes = input.element_size() * input.numel()  # How much data in the input
    sent_bytes = data_bytes * (world_size - 1)  # How much needs to be sent (no 2x here)
    total_duration = world_size * duration  # Total time for transmission
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    cleanup()
```

## 3 - data_parallelism_main

```python
def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = int_divide(batch_size, world_size)  # @inspect local_batch_size
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index:end_index].to(get_device(rank))

    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state

    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass
        loss.backward()

        # Sync gradients across workers (only difference between standard training and DDP)
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Update parameters
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)

    cleanup()
```

### 与 PyTorch DDP 的对比
*   **相同点**: 核心逻辑（数据分发、模型复制、梯度All-Reduce、参数更新）与官方的 `DistributedDataParallel` (DDP) 模块完全一致。
*   **不同点**: 这是一个**简化版**实现。真实的 DDP 做了大量优化：
    *   **桶化 (Bucketing)**: 将多个小参数的梯度打包成一个“桶”再进行 All-Reduce，以减少通信次数，提高效率。
    *   **重叠通信与计算**: 在反向传播期间，当一个层的梯度计算完成后，立即开始该层梯度的通信（All-Reduce），同时计算下一层的梯度，从而将通信时间隐藏起来。
    *   **更健壮的异常处理**。

### 总结

这段代码完美展示了数据并行训练的核心原则：

1.  **分割数据**，在多个设备/进程上复制模型。
2.  独立进行**前向和反向传播**。
3.  使用 **All-Reduce（平均）** 来同步所有设备上的梯度。
4.  独立但**同步地更新参数**，确保所有模型副本保持一致。

这种模式是现代大规模深度学习训练的基石，允许我们使用多个GPU来加速训练并处理更大的批次大小。手动实现它有助于深刻理解 `torch.nn.parallel.DistributedDataParallel` 的工作原理。

## 4 - tensor_parallelism_main

```python
def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_num_dim = int_divide(num_dim, world_size)  # Shard `num_dim`  @inspect local_num_dim

    # Create model (each rank gets 1/world_size of the parameters)
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]

    # Forward pass
    x = data
    for i in range(num_layers):
        # Compute activations (batch_size x local_num_dim)
        x = x @ params[i]  # Note: this is only on a slice of the parameters
        x = F.gelu(x)

        # Allocate memory for activations (world_size x batch_size x local_num_dim)
        activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]

        # Send activations via all gather
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # Concatenate them to get batch_size x num_dim
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)

    # Backward pass: homework exercise

    cleanup()
```

### 与数据并行 (Data Parallelism) 的对比
| 特性 | 数据并行 (前一个示例) | 张量并行 (本示例) |
| :--- | :--- | :--- |
| **数据** | 分割 | **复制** |
| **模型** | **复制** | 分割 |
| **通信时机** | 反向传播后 | **前向传播中** |
| **通信操作** | `all_reduce` (梯度) | `all_gather` (激活值) |
| **适用场景** | 模型能放入单个设备，需扩大批次 | **模型太大，无法放入单个设备** |

### 代码的局限性与扩展
*   **仅实现了前向传播**: 代码注释 `# Backward pass: homework exercise` 明确指出反向传播需要自己实现。这通常涉及：
    1.  在反向传播时，需要对上游传来的梯度进行 **`all-reduce`** 操作。
    2.  或者更复杂的通信模式（如 `reduce-scatter`）。
*   **简化实现**: 这是一个教学示例。真实的张量并行库（如 Megatron-LM）会进行大量优化，例如将通信（`all-gather`）和计算（GEMM）重叠起来以隐藏通信开销。

### 总结

这段代码清晰地展示了张量并行的基本原理：

1.  **复制数据**到所有设备。
2.  **分割模型参数**到各个设备。
3.  在前向传播中，每个设备进行**局部计算**。
4.  使用 **All-Gather** 通信操作**同步和拼接**局部结果，得到完整的输出。
5.  将完整输出传递给下一层（同样被分割的层）。

张量并行是训练**超大模型**的关键技术，它解决了当模型参数太多，无法放入单个设备内存时的问题。通常它会与数据并行结合使用，形成复杂的混合并行策略，用于训练像GPT、LLaMA这样的千亿级参数模型。

## 5 - pipeline_parallelism_main

```python
def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)

    # Use all the data
    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim

    # Split up layers
    local_num_layers = int_divide(num_layers, world_size)  # @inspect local_num_layers

    # Each rank gets a subset of layers
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]

    # Forward pass

    # Break up into micro batches to minimize the bubble
    micro_batch_size = int_divide(batch_size, num_micro_batches)  # @inspect micro_batch_size
    if rank == 0:
        # The data
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # Allocate memory for activations
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]

    for x in micro_batches:
        # Get activations from previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # Compute layers assigned to this rank
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # Send to the next rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)

    text("Not handled: overlapping communication/computation to eliminate pipeline bubbles")

    # Backward pass: homework exercise

    cleanup()
```

### 1. 模型分割 (Model Partitioning)
*   **核心思想**: “流水线并行”意味着**模型被纵向分割**。
*   **实现**: 
    ```python
    local_num_layers = num_layers // world_size
    local_params = [get_init_params(...) for i in range(local_num_layers)]
    ```
*   **要点**: 每个 Rank 只持有**模型的一部分层**（`local_num_layers`）。例如，一个12层的模型在4个设备上，每个设备负责3层。这是一种**按层划分**的策略。

### 2. 微批次 (Micro-batching) - 关键优化
*   **核心思想**: 将一个大批次拆分成多个小批次，以**减少"流水线气泡"（Bubble）**，提高设备利用率。
*   **实现**: 
    ```python
    micro_batch_size = batch_size // num_micro_batches
    micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    ```
*   **要点**:
    *   **流水线气泡**: 在朴素的实现中，整个批次在前一个设备计算时，后续设备处于空闲状态，造成计算资源浪费，形成“气泡”。
    *   **微批次**：通过将大批次切分成小块，可以让流水线不同阶段同时处理不同的微批次，从而填充气泡，提高硬件利用率。
    *   **通信模式如下图所示（展示了微批次如何填充气泡）**：
        

### 3. 点对点通信 (Point-to-Point Communication)
*   **核心思想**: 使用**发送（send）**和**接收（recv）** 这种点对点的通信原语，在相邻的流水线阶段之间传递数据。
*   **实现**: 
    ```python
    # 接收来自上一个阶段的数据
    if rank - 1 >= 0:
        dist.recv(tensor=x, src=rank - 1)

    # ...本地计算...

    # 发送数据到下一个阶段
    if rank + 1 < world_size:
        dist.send(tensor=x, dst=rank + 1)
    ```
*   **要点**:
    *   **`dist.recv`**: 阻塞式地从指定的源Rank（`src`）接收数据。
    *   **`dist.send`**: 阻塞式地向指定的目标Rank（`dst`）发送数据。
    *   这与数据并行（All-Reduce）和张量并行（All-Gather）使用的**集体通信**完全不同。流水线并行依赖于**相邻Rank间的点对点通信**。

### 4. 前向传播流程
1.  **Rank 0**:
    *   拥有原始数据，将其切分为微批次。
    *   对每个微批次，计算其负责的层，然后发送给Rank 1。
2.  **中间 Rank (1, 2, ...)**:
    *   为每个微批次分配内存（`torch.empty`）。
    *   从上一个Rank接收激活值。
    *   计算其负责的层。
    *   将结果发送给下一个Rank。
3.  **最后 Rank**:
    *   接收激活值，计算其负责的层，得到最终输出。

### 5. 代码的局限性与挑战
*   **仅实现了前向传播**: 注释 `# Backward pass: homework exercise` 表明反向传播未实现。流水线的反向传播更为复杂，需要沿相反方向传递梯度。
*   **存在流水线气泡**: 注释 `"Not handled: overlapping communication/computation..."` 指出了最大问题。这是一个**朴素实现**，通信和计算是串行的：
    *   `dist.recv` -> 计算 -> `dist.send`
    *   在通信时，计算设备在空闲等待，效率低下。
*   **优化方向**: 先进的流水线并行方案（如GPipe）会使用：
    *   **计算-通信重叠**: 使用异步通信（`async_op=True`）并在等待通信时进行计算。
    *   **梯度累积**: 在反向传播时正确处理来自多个微批次的梯度。
    *   **1F1B调度策略**: 更复杂的调度算法来进一步减少气泡。