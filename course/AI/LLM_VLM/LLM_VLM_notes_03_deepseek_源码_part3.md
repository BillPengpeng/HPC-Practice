本文主要整理Deepseek开源代码的主要内容。

## 7.0 - sample

```python
def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    # temperature > 1：放大低概率token的权重，增加随机性
    # temperature < 1：强化高概率token的优势，减少随机性
    # temperature = 1：保持原始分布不变
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    # 一个服从标准指数分布（由 .exponential_(1)生成）的随机变量 E，其负对数 -log(E)恰好服从标准Gumbel分布
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
```

### Gumbel-Max采样过程
1. 生成指数噪声：`noise = [0.3, 1.2, 0.8]`
2. 变换计算：`log(probs) - log(noise) = [log(0.7)-log(0.3), log(0.2)-log(1.2), log(0.1)-log(0.8)]`
3. 结果可能变为：`[0.85, -1.79, -2.08]` → `argmax`返回0
   或者：`[0.25, -0.45, 0.15]` → `argmax`返回0
   或者：`[-0.5, 0.8, -0.2]` → `argmax`返回1（低概率token被选中）

### 与标准采样方法的对比

| 采样方法 | 实现方式 | 随机性控制 |
|---------|---------|-----------|
| **Greedy** | `argmax(probs)` | 无随机性，总是选择最高概率 |
| **Multinomial** | `torch.multinomial(probs, 1)` | 直接按概率分布随机抽样 |
| **Gumbel-Max** | 本函数实现 | 通过噪声注入的argmax操作 |

## 7.1 - generate

```python
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    # 批量处理：支持同时生成多个序列，提高硬件利用率
    # 内存预分配：一次性分配最大长度的张量，避免动态扩容开销
    # 填充标记：用-1初始化，后续通过掩码区分真实token和填充位置
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    # 自回归生成循环
    # KV缓存优化：通过prev_pos参数实现高效的键值缓存，每次只计算新token的注意力，将复杂度从O(n²)降低到O(n)
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        # 掩码处理与提前终止
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        # 后处理逻辑
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens
```

### 与标准实现的对比优势

| 特性 | 标准序列生成 | 此优化实现 |
|------|-------------|------------|
| 内存使用 | 随序列长度平方增长（每一步都重新计算所有历史token的Key和Value） | 线性增长，支持更长序列（KV Cache缓存并复用历史token的Key和Value） |
| 推理速度 | 每次全序列计算 | 增量计算，利用缓存 |
| 批量处理 | 序列长度需对齐 | 支持不同长度序列 |
| 终止处理 | 统一终止 | 按序列独立终止 |

## 8.0 - act_quant_kernel

```python
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, scale_fmt: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    # 找到块内绝对值的最大值
    amax = tl.max(tl.abs(x)) # reduction
    # 钳位最小值，避免除零和溢出
    amax = tl.maximum(amax, 1e-4) # clamp to 1e-4
    s = amax / 448.
    if scale_fmt == "ue8m0":
        exp = tl.math.ceil(tl.math.log2(s))
        s = tl.math.exp2(exp)
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)

def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    # 例如，输入形状为 (A, B)，block_size=128，则 s的形状为 (A, B//128)。
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, scale_fmt=scale_fmt)
    return y, s
```

### 核心优势总结
- 精细的动态范围：按块量化相比全局量化，能更好地适应张量内部不同区域的数据分布变化，尤其在激活值分布不均匀时，精度保留得更好。
- 硬件友好：生成的FP8数据可以直接喂给支持FP8计算指令的硬件（如Tensor Core），实现加速。
- 与深度学习流程无缝集成：封装良好，在PyTorch生态中可以像普通算子一样调用 。

## 8.1 - weight_dequant_kernel

```python
@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    # 每个数据块共享一个缩放因子，大幅减少存储开销
    # 缩放因子索引计算：pid_m * n + pid_n实现二维到一维的映射
    # 逐元素乘法实现反量化，数学形式为：dequantized_value = quantized_value × scale
    # scale_out_features = (out_features + block_size - 1) // block_size
    # scale_in_features = (in_features + block_size - 1) // block_size
    # self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y
```

## 8.2 - fp8_gemm

```python
@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        # 数学等价于：(A_quant × scale_A) × (B_quant × scale_B) = A × B
        # FP32累加器：虽然输入是FP8，但使用FP32进行中间结果累加，避免精度损失
        # 数值稳定性：FP32的更大动态范围防止累加过程中的下溢和溢出
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c
```

### 核心参数与功能
| 组件 | 类型/形状 | 功能说明 |
|------|-----------|----------|
| 输入矩阵A | `[M, K]`, FP8 | 激活值矩阵，已量化 |
| 缩放因子a_s | `[M, K/BLOCK_K]`, FP32 | A矩阵的块级缩放因子 |
| 输入矩阵B | `[K, N]`, FP8 | 权重矩阵，已量化 |
| 缩放因子b_s | `[K/BLOCK_K, N/BLOCK_N]`, FP32 | B矩阵的块级缩放因子 |
| 输出矩阵C | `[M, N]`, 高精度 | 计算结果，保持高精度 |

### 自动调优机制
```python
@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
```
这是性能优化的关键：
- **自动配置选择**：根据问题规模`N`和`K`自动选择最优的块大小参数
- **硬件适配**：针对不同GPU架构生成最佳内核配置
- **性能优化**：避免手动调参，实现跨平台的性能可移植性
