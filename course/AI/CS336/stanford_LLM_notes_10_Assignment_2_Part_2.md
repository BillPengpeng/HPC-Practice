本文主要整理Assignment 2 (systems): Systems and Parallelism的主要内容。

## 1.2 Optimizing Attention with FlashAttention-2

## 1.2.1 Benchmarking PyTorch Attention

### **内容概况**

通过性能分析指出，标准的注意力层在**内存使用和计算效率**方面存在优化空间。接着，它给出了注意力机制的标准数学公式，并指出了传统“朴素实现”在处理长序列时会遇到的内存瓶颈问题。最后，它提出将依据FlashAttention-2论文的思想，实现一种新的、更高效的内核来解决这些问题。

---

### **要点总结**

1.  **核心问题**：
    *   传统的“朴素”注意力实现需要保存一个大小为 `seq_len × seq_len`（序列长度乘以序列长度）的注意力分数矩阵。
    *   当处理长序列时，这个矩阵会变得非常巨大，极易导致**内存不足（OOM）** 的错误，限制了模型处理长文本等任务的能力。

2.  **解决方案**：
    *   采用 **FlashAttention-2** 论文中提出的方法。
    *   核心思想是**分块计算**：将大型矩阵运算分解成更小的**tiles**在GPU高速缓存中进行处理。
    *   关键优势是**避免显式生成**那个巨大的 `seq_len × seq_len` 中间矩阵。

3.  **最终效益**：
    *   这种优化能显著**降低内存占用**，使得模型能够**处理更长的输入/输出序列**，从而扩展了模型的应用范围。

简单来说，这部分内容阐述了**为什么**标准注意力机制在长序列上效率低下，以及FlashAttention-2通过**分块计算、避免保存大矩阵**的方法是如何解决这个问题的。

### Problem (pytorch_attention): 2 points

(a) Benchmark your attention implementation at different scales. Write a script that will:

    - (a) Fix the batch size to 8 and don’t use multihead attention (i.e. remove the head dimension).
    - (b) Iterate through the cartesian product of [16, 32, 64, 128] for the head embedding dimension dmodel , and [256, 1024, 4096, 8192, 16384] for the sequence length.
    - (c) Create random inputs Q, K, V for the appropriate size.
    - (d) Time 100 forward passes through attention using the inputs.
    - (e) Measure how much memory is in use before the backward pass starts, and time 100 backward passes.
    - (f) Make sure to warm up, and to call torch.cuda.synchronize() after each forward/backward pass.

Report the timings (or out-of-memory errors) you get for these configurations. At what size do you get out-of-memory errors? Do the accounting for the memory usage of attention in one of the smallest configurations you find that runs out of memory (you can use the equations for memory usage of Transformers from Assignment 1). How does the memory saved for backward change with the sequence length? What would you do to eliminate this memory cost?

**24GB RTX 4090 F32 forward passes + backward passes + requires_grad=True**

**close memory_profiling**
|   batch_size |   seq_len |   d_model |   forward_mean |   forward_var |   backward_mean |   backward_var |
|-------------:|----------:|----------:|---------------:|--------------:|----------------:|---------------:|
|            8 |       256 |        16 |       0.453677 |   0.00401697  |        0.799328 |     0.0235938  |
|            8 |      1024 |        16 |       0.928505 |   0.0149303   |        1.8054   |     0.0412306  |
|            8 |      4096 |        16 |       7.22446  |   0.0416724   |       18.1238   |     0.0114806  |
|            8 |      8192 |        16 |      28.6113   |   0.0797124   |       71.215    |     0.010446   |
|            8 |     16384 |        16 |     nan        | nan           |      nan        |   nan          |
|            8 |       256 |        32 |       0.927647 |   0.00365832  |        1.79747  |     0.0023778  |
|            8 |      1024 |        32 |       0.935939 |   0.00958346  |        1.87103  |     0.00852865 |
|            8 |      4096 |        32 |       7.25614  |   0.000413922 |       18.252    |     0.00931891 |
|            8 |      8192 |        32 |      28.6892   |   0.00262787  |       71.6261   |     0.0713379  |
|            8 |     16384 |        32 |     nan        | nan           |      nan        |   nan          |
|            8 |       256 |        64 |       0.677461 |   0.0484902   |        1.78847  |     0.0312141  |
|            8 |      1024 |        64 |       0.886136 |   0.00751167  |        1.46219  |     0.116512   |
|            8 |      4096 |        64 |       7.37675  |   0.00038191  |       18.4439   |     0.00529295 |
|            8 |      8192 |        64 |      28.8853   |   0.0019475   |       72.0195   |     0.0190723  |
|            8 |     16384 |        64 |     nan        | nan           |      nan        |   nan          |
|            8 |       256 |       128 |       0.667343 |   0.0483345   |        1.76945  |     0.0134983  |
|            8 |      1024 |       128 |       0.945105 |   0.00360519  |        1.68116  |     0.082265   |
|            8 |      4096 |       128 |       7.65212  |   0.000391173 |       18.6211   |     0.00811945 |
|            8 |      8192 |       128 |      30.199    |   0.00133687  |       73.4746   |     0.0101508  |
|            8 |     16384 |       128 |     nan        | nan           |      nan        |   nan          |

**open memory_profiling**

|   batch_size |   seq_len |   d_model |   forward_mean |   forward_var |   backward_mean |   backward_var | memory_viz |
|-------------:|----------:|----------:|---------------:|--------------:|----------------:|---------------:| ----------:|
|            8 |       256 |        16 |       0.626304 |   0.0125875   |        1.12082  |     0.0780371  |        7.9 / 4.0 |
|            8 |      1024 |        16 |       0.992918 |   0.119447    |        2.27165  |     0.717805   |      189.8 / 96.7 |
|            8 |      4096 |        16 |       7.21603  |   0.00106616  |       18.2129   |     0.0173816  |     3077.6 / 1540.1 |
|            8 |      8192 |        16 |      28.6216   |   0.0756354   |       71.3872   |     0.0520851  |    12304.0 / 6154.6 |
|            8 |     16384 |        16 |     nan        | nan           |      nan        |   nan          |        nan |
|            8 |       256 |        32 |       1.05024  |   0.00622766  |        1.84885  |     0.152644   |        8.5 / 4.4 |
|            8 |      1024 |        32 |       0.908019 |   0.013483    |        1.27466  |     0.0352241  |      192.3 / 96.8 |
|            8 |      4096 |        32 |       7.26021  |   0.00075895  |       18.3003   |     0.0144825  |     3087.6 / 1546.1 |
|            8 |      8192 |        32 |      28.7046   |   0.000717308 |       71.7559   |     0.0646042  |    12324.0 / 6166.4 |
|            8 |     16384 |        32 |     nan        | nan           |      nan        |   nan          |        nan |
|            8 |       256 |        64 |       0.794733 |   0.0620842   |        2.09174  |     0.00766228 |        9.8 / 5.2 |
|            8 |      1024 |        64 |       1.07637  |   0.00920887  |        2.2737   |     0.00744811 |      197.4 / 101.7 |
|            8 |      4096 |        64 |       7.29983  |   0.00028886  |       18.4334   |     0.00540532 |     3107.6 / 1558.1 |
|            8 |      8192 |        64 |      28.9161   |   0.00323605  |       72.2276   |     0.0792161  |    12364.0 / 6190.6 |
|            8 |     16384 |        64 |     nan        | nan           |      nan        |   nan          |        nan |
|            8 |       256 |       128 |       0.375683 |   0.00149735  |        0.929131 |     1.14391    |       12.3 / 6.7 |
|            8 |      1024 |       128 |       0.446268 |   0.00105946  |        1.21873  |     1.09184    |      207.4 / 109.8 |
|            8 |      4096 |       128 |       7.66655  |   0.00111304  |       18.828    |     0.0175677  |     3147.7 / 1582.2 |
|            8 |      8192 |       128 |      30.2216   |   0.00207575  |       73.6611   |     0.0160511  |    12444.1 / 6238.4 |
|            8 |     16384 |       128 |     nan        | nan           |      nan        |   nan          |        nan |

```python
batch_size * seq_len * d_model * 3 * 2 + batch_size * seq_len * seq_len * 2 + batch_size * seq_len * d_model = batch_size * (7 * seq_len * d_model + 2 * seq_len * seq_len) => 8 * (7 * 1024 * 16 + 2 * 1024 * 1024) * 4 = 67.5
```

## 1.3 Benchmarking JIT-Compiled Attention


### **内容概况**

该章节主要介绍了自**PyTorch 2.0**版本起引入的即时编译器（JIT Compiler），并说明了如何利用它来优化模型（特别是注意力机制）的性能。

文档首先阐述了该编译器的强大功能——能够自动对PyTorch函数应用多种优化。核心亮点在于，它能通过动态分析计算图，**自动生成融合的Triton内核**。随后，文档通过简洁的代码示例（`torch.compile(layer)`）展示了其极其简单的使用接口，并说明它可以应用于单个层、整个模型甚至任意包含PyTorch操作的Python函数。

---

### **要点总结**

1.  **核心特性**：
    *   PyTorch 2.0 引入了一个**即时编译器（JIT Compiler）**。
    *   其主要优势是能**自动进行性能优化**，无需手动重写代码。
    *   一个关键的优化手段是**自动生成融合的Triton内核**，这能显著提升计算效率。

2.  **工作原理**：
    *   编译器会**动态分析用户的计算图**，识别出可以优化的模式。
    *   通过内核融合等技术，将多个操作合并为一个更高效的内核，减少内存访问开销。

3.  **使用方法**：
    *   接口**非常简单**，只需使用 `torch.compile()` 函数包装目标对象即可。
    *   其应用范围灵活，可以编译：
        *   单个层：`compiled_layer = torch.compile(layer)`
        *   整个模型：`compiled_model = torch.compile(model)`
        *   任意PyTorch函数。

4.  **行为一致性**：
    *   编译后的对象（如 `compiled_layer`）在功能上与原始对象（如 `layer`）**完全一致**，保留其前向传播和反向传播行为，用户无需改变其他代码。

5.  **学习资源**：
    *   文档提供了一个官方教程链接（`https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html`），供读者深入了解。

**总结来说**，该部分内容旨在说明，PyTorch 2.0 的 `torch.compile` 是一个强大且易用的工具，能够通过JIT编译技术（特别是自动内核融合）来提升模型的运行性能，尤其适用于计算密集的注意力机制。

### Problem (torch_compile): 2 points

- (a) Extend your attention benchmarking script to include a compiled version of your PyTorch imple-
mentation of attention, and compare its performance to the uncompiled version with the same
configuration as the pytorch_attention problem above.

**without complile**
|   batch_size |   seq_len |   d_model |   forward_mean |   forward_var |   backward_mean |   backward_var | memory_viz |
|-------------:|----------:|----------:|---------------:|--------------:|----------------:|---------------:| ----------:|
|            8 |       256 |        16 |       0.739294 |    0.0146192  |         1.18358 |      0.044634  |     4   |
|            8 |      1024 |        16 |       1.12543  |    0.0136842  |         1.70358 |      0.165873  |    95.7 | 
|            8 |      4096 |        16 |       7.38914  |    0.0290962  |        18.4043  |      0.0256124 |  1542.0 |
|            8 |      8192 |        16 |      28.6945   |    0.0752015  |        71.5764  |      0.0463954 |  6158.4 |
|            8 |     16384 |        16 |     nan        |  nan          |       nan       |    nan         |  nan    |
|            8 |       256 |        32 |       0.86928  |    0.111451   |         1.60194 |      0.451898  |     4.4 |
|            8 |      1024 |        32 |       0.473555 |    0.00493164 |         1.29327 |      3.09352   |    97.7 |
|            8 |      4096 |        32 |       7.35303  |    0.00126528 |        18.454   |      0.0126596 |  1550.0 |
|            8 |      8192 |        32 |      28.7734   |    0.0013337  |        72.0742  |      0.25045   |  6174.4 |
|            8 |     16384 |        32 |     nan        |  nan          |       nan       |    nan         |  nan    |
|            8 |       256 |        64 |       0.925159 |    0.10084    |         2.46431 |      0.26635   |     5.7 |
|            8 |      1024 |        64 |       1.22609  |    0.0477934  |         2.52549 |      0.179628  |   101.7 |
|            8 |      4096 |        64 |       7.42667  |    0.00790012 |        18.6134  |      0.011595  |  1566.0 |
|            8 |      8192 |        64 |      28.9823   |    0.00287432 |        72.2633  |      0.0250939 |  6206.4 |
|            8 |     16384 |        64 |     nan        |  nan          |       nan       |    nan         |  nan    |
|            8 |       256 |       128 |       0.446513 |    0.00513715 |         1.09011 |      1.52746   |     7.7 |
|            8 |      1024 |       128 |       0.538641 |    0.00479128 |         1.19652 |      0.0117769 |   109.8 | 
|            8 |      4096 |       128 |       7.82607  |    0.00811924 |        19.063   |      0.0369747 |  1598.1 |
|            8 |      8192 |       128 |      30.2725   |    0.00287844 |        73.7038  |      0.0225252 |  6270.4 |
|            8 |     16384 |       128 |     nan        |  nan          |       nan       |    nan         |  nan    |

**complile**
|   batch_size |   seq_len |   d_model |   forward_mean |   forward_var |   backward_mean |   backward_var | memory_viz |
|-------------:|----------:|----------:|---------------:|--------------:|----------------:|---------------:| ----------:|
|            8 |       256 |        16 |       0.744747 |   0.150715    |         1.2064  |     0.413881   |      2   |
|            8 |      1024 |        16 |       1.03543  |   0.087868    |         1.62442 |     0.0133532  |     63.8 |
|            8 |      4096 |        16 |       3.92644  |   0.00495015  |         8.46306 |     0.183564   |   1030.1 |
|            8 |      8192 |        16 |      15.1763   |   0.0350033   |        33.1019  |     0.0802343  |   4110.6 |
|            8 |     16384 |        16 |     nan        | nan           |       nan       |   nan          |   nan    |
|            8 |       256 |        32 |       1.32577  |   0.101575    |         1.29827 |     0.191482   |      2.4 |
|            8 |      1024 |        32 |       1.65335  |   0.0327419   |         1.62903 |     0.00837039 |     65.8 |
|            8 |      4096 |        32 |       3.93225  |   0.00367883  |         8.43288 |     0.0214639  |   1038.1 |
|            8 |      8192 |        32 |      15.1506   |   0.00671768  |        32.3933  |     0.0513368  |   4126.6 |
|            8 |     16384 |        32 |     nan        | nan           |       nan       |   nan          |   nan    |
|            8 |       256 |        64 |       1.13243  |   0.088422    |         1.6405  |     0.0191037  |      3.2 |
|            8 |      1024 |        64 |       1.67424  |   0.0295369   |         1.62411 |     0.101147   |     69.8 |
|            8 |      4096 |        64 |       3.97042  |   0.00324354  |         8.6155  |     0.00655299 |   1054.1 |
|            8 |      8192 |        64 |      15.3759   |   0.00636962  |        32.8668  |     0.0240108  |   4158.6 |
|            8 |     16384 |        64 |     nan        | nan           |       nan       |   nan          |   nan    |
|            8 |       256 |       128 |       1.07595  |   0.0691013   |         1.45349 |     0.0788327  |      4.7 |
|            8 |      1024 |       128 |       1.55032  |   0.0816265   |         1.59898 |     0.0785471  |     77.8 |
|            8 |      4096 |       128 |       4.32488  |   0.000800798 |         8.98573 |     0.0112802  |   1086.2 |
|            8 |      8192 |       128 |      16.62     |   0.00514793  |        34.3178  |     0.0302086  |   4222.7 |
|            8 |     16384 |       128 |     nan        | nan           |       nan       |   nan          |   nan    |


- (b) Now, compile your entire Transformer model in your end-to-end benchmarking script. How does
the performance of the forward pass change? What about the combined forward and backward
passes and optimizer steps?

**without complile**
| size   |   batch_size |   seq_len |   d_model |   d_ff |   num_layers |   num_heads |   forward_mean |   forward_var |   backward_mean |   backward_var |   optimizer_mean |   optimizer_var |
|:-------|-------------:|----------:|----------:|-------:|-------------:|------------:|---------------:|--------------:|----------------:|---------------:|-----------------:|----------------:|
| small  |            4 |       128 |       768 |   3072 |           12 |          12 |        22.4806 |     0.421173  |         30.7346 |    0.623644    |          12.9901 |      0.0357111  |
| small  |            4 |       256 |       768 |   3072 |           12 |          12 |        22.7798 |     0.484944  |         31.5711 |    1.09688     |          13.1528 |      0.13464    |
| small  |            4 |       512 |       768 |   3072 |           12 |          12 |        26.3742 |     0.0191905 |         56.7498 |    0.000874768 |          13.2298 |      0.250839   |
| small  |            4 |      1024 |       768 |   3072 |           12 |          12 |        78.8594 |     0.0606328 |        164.186  |    0.0189636   |          14.0294 |      0.490712   |
| medium |            4 |       128 |      1024 |   4096 |           24 |          16 |        46.8877 |     2.02771   |         62.8884 |    2.58636     |          29.8213 |      0.0796637  |
| medium |            4 |       256 |      1024 |   4096 |           24 |          16 |        47.9138 |     4.30487   |         65.435  |    1.34409     |          29.937  |      0.0136989  |
| medium |            4 |       512 |      1024 |   4096 |           24 |          16 |        79.9813 |     0.0392303 |        163.975  |    0.194378    |          29.9358 |      0.00872574 |
| medium |            4 |      1024 |      1024 |   4096 |           24 |          16 |       nan      |   nan         |        nan      |  nan           |         nan      |    nan          |
| large  |            4 |       128 |      1280 |   5120 |           36 |          20 |        68.8331 |     5.90341   |         96.4653 |   38.3772      |          89.1011 |      0.0125677  |
| large  |            4 |       256 |      1280 |   5120 |           36 |          20 |        73.9268 |     0.0212008 |        148.01   |    0.17282     |          89.3505 |      0.0114652  |
| large  |            4 |       512 |      1280 |   5120 |           36 |          20 |       nan      |   nan         |        nan      |  nan           |         nan      |    nan          |
| large  |            4 |      1024 |      1280 |   5120 |           36 |          20 |       nan      |   nan         |        nan      |  nan           |         nan      |    nan          |
| xl     |            4 |       128 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |  nan           |         nan      |    nan          |
| xl     |            4 |       256 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |  nan           |         nan      |    nan          |
| xl     |            4 |       512 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |  nan           |         nan      |    nan          |
| xl     |            4 |      1024 |      1600 |   6400 |           48 |          25 |       nan      |   nan         |        nan      |  nan           |         nan      |    nan          |
| 2.7B   |            4 |       128 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |  nan           |         nan      |    nan          |
| 2.7B   |            4 |       256 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |  nan           |         nan      |    nan          |
| 2.7B   |            4 |       512 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |  nan           |         nan      |    nan          |
| 2.7B   |            4 |      1024 |      2560 |  10240 |           32 |          32 |       nan      |   nan         |        nan      |  nan           |         nan      |    nan          |

**complile**

| size   |   batch_size |   seq_len |   d_model |   d_ff |   num_layers |   num_heads |   forward_mean |   forward_var |   backward_mean |   backward_var |   optimizer_mean |   optimizer_var |
|:-------|-------------:|----------:|----------:|-------:|-------------:|------------:|---------------:|--------------:|----------------:|---------------:|-----------------:|----------------:|
| small  |            4 |       128 |       768 |   3072 |           12 |          12 |        19.9172 |     6.07612   |         24.8689 |    11.6962     |          15.0129 |      2.09184    |
| small  |            4 |       256 |       768 |   3072 |           12 |          12 |        24.8855 |     8.29157   |         25.2631 |     4.99185    |          17.081  |      5.53381    |
| small  |            4 |       512 |       768 |   3072 |           12 |          12 |        27.7822 |     3.42429   |         41.828  |     0.0282265  |          16.2369 |      5.67695    |
| small  |            4 |      1024 |       768 |   3072 |           12 |          12 |        60.156  |     0.0421287 |        106.854  |     0.00534626 |          16.5264 |      5.23001    |
| medium |            4 |       128 |      1024 |   4096 |           24 |          16 |        43.7525 |     0.869523  |         39.9373 |     3.38526    |          29.8084 |      0.0663038  |
| medium |            4 |       256 |      1024 |   4096 |           24 |          16 |        46.856  |     2.15062   |         53.5242 |     0.572897   |          31.0393 |      8.97711    |
| medium |            4 |       512 |      1024 |   4096 |           24 |          16 |        69.2738 |     0.0894491 |        115.199  |     0.0113046  |          31.963  |      9.87271    |
| medium |            4 |      1024 |      1024 |   4096 |           24 |          16 |       nan      |   nan         |        nan      |   nan          |         nan      |    nan          |
| large  |            4 |       128 |      1280 |   5120 |           36 |          20 |        79.9695 |   165.98      |         81.4436 |    12.5027     |          89.1177 |      0.0435684  |
| large  |            4 |       256 |      1280 |   5120 |           36 |          20 |        78.0431 |     5.21439   |        134.221  |     0.16893    |          89.247  |      0.00987979 |
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

## 1.3.1 Example - Weighted Sum

### Forward pass Baseline

```python
def weighted_sum(x, weight):
    # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
    return (weight * x).sum(axis=-1)
```

### Forward pass triton

```python
import triton
import triton.language as tl

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,  # Input pointers
    output_ptr,  # Output pointer
    x_row_stride, x_stride_dim,  # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim,  # Likely 1
    output_stride_row,  # Likely 1
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,  # Tile shapes must be known at compile time
):
    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)

    # Block pointers give us a way to select from an ND region of memory
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to use load/store at a time
    # - The order of the dimensions in memory from major to minor
    #   axes (= np.argsort(strides)) for optimizations, especially useful on H100

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,), 
        strides=(x_row_stride, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), 
        strides=(weight_stride_dim,), 
        offsets=(0,), 
        block_shape=(D_TILE_SIZE,), 
        order=(0,), 
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,), 
        strides=(output_stride_row,), 
        offsets=(row_tile_idx * ROWS_TILE_SIZE,), 
        block_shape=(ROWS_TILE_SIZE,), 
        order=(0,), 
    )

    # Initialize a write buffer for output
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    
    # Loop over D dimension in tiles
    for d_offset in range(0, D, D_TILE_SIZE):
        # Load current tile of x
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # Load current tile of weights
        weight_tile = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        
        # Compute weighted sum for this tile
        weighted = x_tile * weight_tile
        output += tl.sum(weighted, axis=1)
        
        # Move pointers to next tile
        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))
    
    # Write output
    tl.store(output_block_ptr, output, boundary_check=(0,))
```

### call Forward pass triton


```python
import torch
import triton
import triton.language as tl
from einops import rearrange

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        D, output_dims = x.shape[-1], x.shape[:-1]
        
        # Reshape input tensor to 2D
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        
        # 上下文保存​​ 保存前向传播的输入
        ctx.save_for_backward(x, weight)
        
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"
        
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # Roughly 16 loops through the embedding dimension
        ctx.ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time
        ctx.input_shape = input_shape
        
        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        y = torch.empty(output_dims, device=x.device)
        
        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight, y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        
        return y.view(input_shape[:-1])
```

### Backward pass triton

```python
import triton
import triton.language as tl

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,  # Input
    grad_output_ptr,  # Grad input
    grad_x_ptr, partial_grad_weight_ptr,  # Grad outputs
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_rows, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    # Inputs
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_rows,), strides=(stride_gr,), 
        offsets=(row_tile_idx * ROWS_TILE_SIZE,), 
        block_shape=(ROWS_TILE_SIZE,), 
        order=(0,), 
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_rows, D,), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), 
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE), 
        order=(1, 0), 
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,), 
        offsets=(0,), block_shape=(D_TILE_SIZE,), 
        order=(0,), 
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_rows, D,), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D,), strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")  # (ROWS_TILE_SIZE,)
        
        # Outer product for grad_x
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))
        
        # Reduce as many rows as possible for the grad_weight result
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))  # Never out of bounds for dim 0
        
        # Move the pointers to the next tile along D
        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))
        partial_grad_weight_block_ptr = tl.advance(partial_grad_weight_block_ptr, (0, D_TILE_SIZE))
        grad_x_block_ptr = tl.advance(grad_x_block_ptr, (0, D_TILE_SIZE))
```

#### **梯度计算1：对x的梯度**
```python
grad_x_row = grad_output[:, None] * weight[None, :]
```
- **数学公式**: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \otimes weight$
- **实现**: 使用广播机制进行外积计算

#### **梯度计算2：对weight的梯度**
```python
grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
```
- **数学公式**: $\frac{\partial L}{\partial weight} = \frac{\partial L}{\partial y}^T \cdot x$
- **实现**: 沿批次维度进行归约求和

### call Backward pass triton

```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # ... (defined earlier)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE  # These don't have to be the same
        n_rows, D = x.shape

        # Our strategy is for each thread block to first write to a partial buffer,
        # then we reduce over this buffer to get the final gradient.
        partial_grad_weight = torch.empty((cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_rows=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight
```