本文主要整理PMPP Chapter 4 Exercise。

## Q1

1. Consider the following CUDA kernel and the corresponding host function that calls it:

```c
__global void void foo_kernel(int* a, int* b) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;
    }
    if(i%2 == 0) {
        a[i] = b[i]*2;
    }
    for(unsigned int j = 0; j < 5 - (i%3); ++j) {
        b[i] += j;
    }
}

void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
}
```
 - a. What is the number of warps per block?  => 128 / 32 = 4
 - b. What is the number of warps in the grid? => 8 * 128 / 32 = 32
 - c. For the statement on line 04:
    - i. How many warps in the grid are active?  => 8 * 3 = 24 warps
    - ii. How many warps in the grid are divergent? => 8 * 2 = 16 warps
    - iii. What is the SIMD efficiency (in %) of warp 0 of block 0? => 100%
    - iv. What is the SIMD efficiency (in %) of warp 1 of block 0? => 8 / 32 = 25%
    - v. What is the SIMD efficiency (in %) of warp 3 of block 0? => 24 / 32 = 75%
      - Warp 0（线程 0~31）：全部满足条件（均在 0~39 范围内）→ 活跃。
      - Warp 1（线程 32~63）：部分满足（32~39 满足，40~63 不满足）→ 活跃（至少有 1 个线程执行）。
      - Warp 2（线程 64~95）：全部不满足 → 不活跃？（注意：这里之前分析有误！Warp 2 的线程 64~95 均不满足条件，因此没有线程执行第 4 行语句，所以该 warp 不活跃。）
      - Warp 3（线程 96~127）：部分满足（104~127 满足，96~103 不满足）→ 活跃。

 - d. For the statement on line 07:
    - i. How many warps in the grid are active? => all 32 warps
    - ii. How many warps in the grid are divergent? => all 32 warps  
    - iii. What is the SIMD efficiency (in %) of warp 0 of block 0? => 50%
 - e. For the loop on line 09:
    - i. How many iterations have no divergence? => 0
    - ii. How many iterations have divergence? => 5
      - 一个迭代 “无分歧” 的前提是：warp 内所有线程在该次迭代中执行相同的操作（即所有线程要么都进入该迭代，要么都不进入）。

## Q2

2. For a vector addition, assume that the vector length is 2000, each thread
   calculates one output element, and the thread block size is 512 threads. How
   many threads will be in the grid?
- Answer: (2000 + 512 - 1) / 512 * 512 = 2048

## Q3

3. For the previous question, how many warps do you expect to have divergence
   due to the boundary check on vector length?
- Answer: (2000 + 512 - 1) / 512 = 4

## Q4

4. Consider a hypothetical block with 8 threads executing a section of code
 before reaching a barrier. The threads require the following amount of time
 (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and
 2.9; they spend the rest of their time waiting for the barrier. What percentage
 of the threads’ total execution time is spent waiting for the barrier?
- Answer: 1.0 + 0.7 + 0 + 0.2 + 0.6 + 1.1 + 0.4 + 0.1 = 4.1, 
Percentage = (Total waiting time / Total execution time) × 100 = (4.1 / 24.0) × 100 ≈ 17.08%

## Q5

5. A CUDA programmer says that if they launch a kernel with only 32 threads
 in each block, they can leave out the __syncthreads() instruction wherever
 barrier synchronization is needed. Do you think this is a good idea? Explain.

这种说法**不是一个好主意**，甚至可能导致程序运行结果错误或不可预期。要理解这一点，需要结合CUDA中“线程束（warp）执行模型”和“__syncthreads()的真正作用”来分析：

### 1. 先明确核心概念
在CUDA中，线程的执行以**线程束（warp）** 为基本单位：1个线程束固定包含32个线程（无论GPU架构如何），同一线程束内的线程会以“锁步（lock-step）”方式执行相同的指令（即SIMD执行模型）。  

而`__syncthreads()`是CUDA提供的**块级屏障同步指令**，作用是强制**同一block内的所有线程**都执行到该指令后，才允许任何线程继续执行后续代码——它确保的是“block内所有线程的同步”，而非“线程束内的同步”。


### 2. 为什么“32线程/block”不能替代`__syncthreads()`？
当block内线程数=32时，整个block恰好对应1个线程束（32线程=1 warp），此时线程束内的线程确实会“锁步执行”，不存在“线程束内线程进度不一致”的问题。但这并不意味着`__syncthreads()`可以省略，核心原因是：  
**`__syncthreads()`的核心作用是“保障共享内存（shared memory）的访问一致性”，而非仅解决“线程进度差”**。  

具体来说，当程序需要“线程间通过共享内存交换数据”时（这是CUDA中最常见的同步场景），即使是同一线程束内的线程，也需要`__syncthreads()`来避免“数据竞争”，因为：  
- 线程对共享内存的写操作（store）并非“立即生效”——GPU为了提升效率，会对内存访问做缓存和重排序优化；  
- 若线程A先写共享内存，线程B后读该内存，但没有`__syncthreads()`，线程B可能会读到“线程A写之前的旧数据”（因为A的写操作还没实际落盘到共享内存，或B的读操作被缓存优化提前了）。  

举个具体例子：  
假设block内32个线程要计算“共享内存中数组的累加和”，逻辑是“每个线程先写自己的计算结果到共享内存，再读取其他线程的结果汇总”。若省略`__syncthreads()`：  
- 线程1可能还没把自己的结果写入共享内存，线程2就已经开始读取线程1的位置——此时线程2读到的是垃圾值，最终累加结果必然错误。  

而`__syncthreads()`的作用，本质是强制所有线程的“内存操作（读/写）”在同步点之前完成，确保后续线程访问共享内存时，能拿到正确的“已同步数据”。


### 3. 额外风险：代码扩展性和可读性问题
即使某段代码在“32线程/block”下侥幸运行正确（比如没有共享内存访问，仅依赖线程束锁步），省略`__syncthreads()`也会导致：  
- **扩展性差**：若后续需要调整block大小（比如改为64线程/block，即2个warp），原代码会立即出现同步错误（不同warp的线程进度不一致，且共享内存访问混乱），需要重新逐个补加`__syncthreads()`，增加维护成本；  
- **可读性差**：其他开发者阅读代码时，无法通过`__syncthreads()`明确“此处需要同步”的逻辑意图，可能误改代码导致bug。


### 结论
无论block内线程数是否为32，只要程序需要“block级屏障同步”（尤其是涉及共享内存访问、线程间数据依赖时），就**必须显式调用`__syncthreads()`**。依赖“32线程=1 warp”的特性省略该指令，本质是依赖GPU底层执行细节而非规范的编程模型，会导致代码脆弱、易出错，绝对不是一个好主意。

## Q6

6. If a CUDA device’s SM can take up to 1536 threads and up to 4 thread
 blocks, which of the following block configurations would result in the most
 number of threads in the SM?
 - a. 128 threads per block
 - b. 256 threads per block
 - c. 512 threads per block
 - d. 1024 threads per block
 - Answer: c
    - a. 128 线程 / 块，最多可容纳 4 个块（受限于 4 个块的上限），总线程数 = 4 × 128 = 512
    - b. 256 线程 / 块，最多可容纳 4 个块，总线程数 = 4 × 256 = 1024
    - c. 512 线程 / 块，最多可容纳 3 个块（4 × 512 = 2048 > 1536，超过线程上限），总线程数 = 3 × 512 = 1536
    - d. 1024 线程 / 块，最多可容纳 1 个块（2 × 1024 = 2048 > 1536），总线程数 = 1 × 1024 = 1024

## Q7

7. Assume a device that allows up to 64 blocks per SM and 2048 threads per
 SM. Indicate which of the following assignments per SM are possible. In the
 cases in which it is possible, indicate the occupancy level.
 - a. 8 blocks with 128 threads each
 - b. 16 blocks with 64 threads each
 - c. 32 blocks with 32 threads each
 - d. 64 blocks with 32 threads each
 - e. 32 blocks with 64 threads each
 - Answer: All a-e

## Q8

8. Consider a GPU with the following hardware limits: 2048 threads per SM, 32
 blocks per SM, and 64K (65,536) registers per SM. For each of the following
 kernel characteristics, specify whether the kernel can achieve full occupancy.
 If not, specify the limiting factor.
 - a. The kernel uses 128 threads per block and 30 registers per thread.
 - b. The kernel uses 32 threads per block and 29 registers per thread.
 - c. The kernel uses 256 threads per block and 34 registers per thread.
 - Answer: a

    - a. 128 线程 / 块，30 寄存器 / 线程
      线程数限制：
      最多可容纳的块数 = min (2048÷128=16, 32) = 16 块
      总线程数 = 16×128 = 2048（达到线程上限）
      寄存器限制：
      总寄存器使用量 = 2048 线程 × 30 寄存器 / 线程 = 61440 ≤ 65536（满足）
      结论：所有限制均满足，且达到最大线程数 → 可实现完全占用率。
    - b. 32 线程 / 块，29 寄存器 / 线程
      线程数限制：
      最多可容纳的块数 = min (2048÷32=64, 32) = 32 块（受限于 32 块上限）
      总线程数 = 32×32 = 1024 < 2048（未达线程上限）
      寄存器限制：
      总寄存器使用量 = 1024 线程 × 29 寄存器 / 线程 = 29696 ≤ 65536（满足）
      结论：受限于块数上限（32 块），无法达到最大线程数 → 不可实现完全占用率，限制因素是块数。
    - c. 256 线程 / 块，34 寄存器 / 线程
      线程数限制：
      最多可容纳的块数 = min (2048÷256=8, 32) = 8 块
      总线程数 = 8×256 = 2048（达到线程上限）
      寄存器限制：
      总寄存器使用量 = 2048 线程 × 34 寄存器 / 线程 = 69632 > 65536（超出限制）
      实际最大线程数 = 65536 ÷ 34 ≈ 1927（向下取整），对应 7 块（7×256=1792 线程）
    - 结论：受限于寄存器数量，无法达到最大线程数 → 不可实现完全占用率，限制因素是寄存器。

## Q9

9. A student mentions that they were able to multiply two 1024*1024 matrices
 using a matrix multiplication kernel with 32*32 thread blocks. The student is
 using a CUDA device that allows up to 512 threads per block and up to 8 blocks
 per SM. The student further mentions that eachthreadinathreadblockcalculates
 one element of the result matrix. What would be your reaction and why?

- Answer:

我的反应会是**怀疑这个学生的描述存在矛盾或误解**，原因如下：

### 核心矛盾点分析
1. **线程块大小与线程总数的矛盾**  
   学生提到使用“32×32的线程块”（即每个块包含32×32=1024个线程），但同时说明其CUDA设备“每个块最多允许512个线程”。  
   - 32×32=1024线程/块，这显然超过了设备512线程/块的硬件限制，**这样的内核配置根本无法启动**（会触发运行时错误）。


2. **矩阵规模与线程映射的合理性问题**  
   学生称“每个线程计算结果矩阵的一个元素”，对于1024×1024的矩阵，需要1024×1024=1,048,576个线程。  
   - 若按合理的块大小（不超过512线程/块），至少需要1,048,576 ÷ 512 = 2048个块。  
   - 但学生使用的设备“每个SM最多允许8个块”，这意味着需要足够多的SM才能容纳2048个块（例如2048 ÷ 8 = 256个SM）。虽然理论上存在这样的设备（如多SM的GPU），但**最初的线程块大小超限问题是根本性的，会直接导致程序失败**。

### 结论
学生的描述存在明显的硬件限制冲突：32×32的线程块（1024线程）超过了设备允许的512线程/块上限，这样的内核配置无法在其设备上运行。这说明学生可能混淆了线程块的维度定义（例如误将“32×32的线程块”理解为32线程），或对设备的硬件限制缺乏清晰认识。实际运行时，CUDA会拒绝启动这种超出线程块大小限制的内核，并返回错误代码。