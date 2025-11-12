# HPC-Practice

## 1 AI 

- Related course
  - [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/)
  - [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)
  - [Stanford CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/index.html)
  - [ZJU LLM: Foundations-of-LLMs](https://github.com/ZJU-LLMs/Foundations-of-LLMs)
  - [CMU 10-414/714: Deep Learning Systems](https://dlsyscourse.org/)
- Related link
  - [mmpretrain](https://github.com/open-mmlab/mmpretrain/tree/main)
  - [transformers](https://github.com/huggingface/transformers)
  - [tiktokenizer](https://tiktokenizer.vercel.app)
  - [scaling-book](https://jax-ml.github.io/scaling-book/)
- Stanford LLM Notes
  - [stanford-LLM Notes: Overview / tokenization (Lecture 1)](./course/AI/CS336/stanford_LLM_notes_01_Overview_tokenization.md)
  - [stanford-LLM Notes: tensors_memory / tensor_operations / tensor_einops (Lecture 2)](./course/AI/CS336/stanford_LLM_notes_02_PyTorch_resource_accounting_Part_1.md)
  - [stanford-LLM Notes: tensor_operations_flops / gradients_flops / optimizer (Lecture 2)](./course/AI/CS336/stanford_LLM_notes_02_PyTorch_resource_accounting_Part_2.md)
  - [stanford-LLM Notes: 经典Transformer / Pre-Norm / RMSNorm / Drop Bias / GLU (Lecture 3)](./course/AI/CS336/stanford_LLM_notes_03_Architectures_hyperparameters_Part_1.md)
  - [stanford-LLM Notes: Sine embeddings / Absolute embeddings / Relative embeddings / RoPE embeddings (Lecture 3)](./course/AI/CS336/stanford_LLM_notes_03_Architectures_hyperparameters_Part_2.md)
  - [stanford-LLM Notes: Feedforward hyperparameters / Head dim / Aspect ratio / Regularization (Lecture 3)](./course/AI/CS336/stanford_LLM_notes_03_Architectures_hyperparameters_Part_3.md)
  - [stanford-LLM Notes: Z-Loss / QK Norm / Soft-Capping / GQA/MQA / SWA (Lecture 3)](./course/AI/CS336/stanford_LLM_notes_03_Architectures_hyperparameters_Part_4.md)
  - [stanford-LLM Notes: MoE/非MoE区别 / Routing / Shared experts / Train routing (Lecture 4)](./course/AI/CS336/stanford_LLM_notes_04_Mixture_of_expert_Part_1.md)
  - [stanford-LLM Notes: Issues with MoEs / upcycling / DeepSeek MoE（待完善） (Lecture 4)](./course/AI/CS336/stanford_LLM_notes_04_Mixture_of_expert_Part_2.md)
  - [stanford-LLM Notes: Unicode Encodings / UTF-8 / Subword Tokenization (Assignment 1)](./course/AI/CS336/stanford_LLM_notes_05_Assignment_1_Part_1.md)
  - [stanford-LLM Notes: BPE Tokenizer Training / BPE Tokenizer: Encoding and Decoding (Assignment 1)](./course/AI/CS336/stanford_LLM_notes_05_Assignment_1_Part_2.md)
  - [stanford-LLM Notes: Transformer Language Model Architecture / Einsum (Assignment 1)](./course/AI/CS336/stanford_LLM_notes_05_Assignment_1_Part_3.md)
  - [stanford-LLM Notes: Linear / Embedding / LayerNorm / SwiGLU / RoPE(*) (Assignment 1)](./course/AI/CS336/stanford_LLM_notes_05_Assignment_1_Part_4.md)
  - [stanford-LLM Notes: Softmax / Self-Attention / The Full Transformer LM(*) (Assignment 1)](./course/AI/CS336/stanford_LLM_notes_05_Assignment_1_Part_5.md)
  - [stanford-LLM Notes: Training a Transformer LM / Training loop / Generating text(*) (Assignment 1)](./course/AI/CS336/stanford_LLM_notes_05_Assignment_1_Part_6.md)
  - [stanford-LLM Notes: GPU & TPU / Low Precision Computation / Operator fusion / Recomputation (Lecture 5)](./course/AI/CS336/stanford_LLM_notes_06_GPUs_Part_1.md)
  - [stanford-LLM Notes: Memory coalescing / Tiling / A matrix mystery / Flash attention (Lecture 5)](./course/AI/CS336/stanford_LLM_notes_06_GPUs_Part_2.md)
  - [stanford-LLM Notes: benchmark & profile / NVTX示例 (Lecture 6)](./course/AI/CS336/stanford_LLM_notes_07_Kernels_Triton_Part_1.md)
  - [stanford-LLM Notes: cuda_gelu / triton_gelu / torch.compile (Lecture 6)](./course/AI/CS336/stanford_LLM_notes_07_Kernels_Triton_Part_2.md)
  - [stanford-LLM Notes: Collective communication / Naïve data parallelism / ZeRO (Lecture 7)](./course/AI/CS336/stanford_LLM_notes_08_Parallelism_Part_1.md)
  - [stanford-LLM Notes: Model parallelism / Pipeline parallel / Tensor parallel / Sequence parallel (Lecture 7)](./course/AI/CS336/stanford_LLM_notes_08_Parallelism_Part_2.md)
  - [stanford-LLM Notes: collective_operations / data_parallelism / tensor_parallelism / pipeline_parallelism (Lecture 8)](./course/AI/CS336/stanford_LLM_notes_09_Parallelism.md)
  - [stanford-LLM Notes: Nsight Systems Profiler / Mixed Precision (Assignment 2)](./course/AI/CS336/stanford_LLM_notes_10_Assignment_2_Part_1.md)
  - [stanford-LLM Notes: Benchmarking PyTorch Attention / Weighted Sum (Assignment 2)](./course/AI/CS336/stanford_LLM_notes_10_Assignment_2_Part_2.md)
  - [stanford-LLM Notes: FlashAttention-2 Forward Pass / Backward pass with recomputation (Assignment 2)](./course/AI/CS336/stanford_LLM_notes_10_Assignment_2_Part_3.md)
  - [stanford-LLM Notes: PyTorch collective_operations / Naïve DDP / Overlapping Computation (Assignment 2)](./course/AI/CS336/stanford_LLM_notes_10_Assignment_2_Part_4.md)
  - [stanford-LLM Notes: 4D Parallelism / Optimizer State Sharding (Assignment 2)](./course/AI/CS336/stanford_LLM_notes_10_Assignment_2_Part_5.md)
  - [stanford-LLM Notes: Data scaling / Neural (LLM) scaling (Lecture 9)](./course/AI/CS336/stanford_LLM_notes_11_Scaling_laws_basic_Part_1.md)
  - [stanford-LLM Notes: Joint data-model scaling laws (Lecture 9)](./course/AI/CS336/stanford_LLM_notes_11_Scaling_laws_basic_Part_2.md)
  - [stanford-LLM Notes: Cerebras-GPT、MiniCPM、DeepSeek有关scaling laws (Lecture 11)](./course/AI/CS336/stanford_LLM_notes_12_Scaling_laws_case_study.md)
- scaling-book Notes
  - [scaling-book Notes: data parallelism / FSDP / tensor parallelism / Combining FSDP and Tensor Parallelism (Part 5)](./course/AI/scaling-book/scaling_book_notes_01_training_Part_1.md)
  - [scaling-book Notes: Combining FSDP and Tensor Parallelism Example / Pipelining / Scaling Across Pods (Part 5)](./course/AI/scaling-book/scaling_book_notes_01_training_Part_2.md)
  - [scaling-book Notes: Background & goals / High-Level Outline (Part 0)](./course/AI/scaling-book/scaling_book_notes_02_intro.md)
- ZJU LLM Notes
  - [ZJU LLM Notes: LLM架构 / Prompt工程 / 参数高效微调 / 模型编辑 / 检索增强生成](./course/AI/ZJU_LLM/LLM_notes_01_LLM基础.md)
  - [Hugging_Face Bert有关Embeddings及Tokenizer: Bert / Albert / Roberta / Electra](./course/AI/ZJU_LLM/LLM_notes_02_Hugging_Face_Bert有关Embeddings及Tokenizer.md)
  - [Hugging_Face Encoder-only代表性方法: Bert / Bert下游任务 / Albert / Roberta / Electra](./course/AI/ZJU_LLM/LLM_notes_03_Hugging_Face_Bert有关模型架构.md)
  - [Hugging_Face Encoder-Decoder代表性方法：T5 / BART](./course/AI/ZJU_LLM/LLM_notes_04_Hugging_Face_T5_BART.md)
- MLSys Notes
  - [CMU-DLSys Notes: Key elements / Learning objects / Tips for AI tools (Lecture1)](./course/AI/CMU_10-414-714/CMU-DLSys_notes_01_Introduction_Logistics.md)
  - [CMU-DLSys Notes: Three ingredients of a ML algorithm / softmax regression example (Lecture2)](./course/AI/CMU_10-414-714/CMU-DLSys_notes_02_ML_Refresher_Softmax_Regression.md)
  - [CMU-DLSys Notes: Neural networks / Universal function approximation (Lecture3)](./course/AI/CMU_10-414-714/CMU-DLSys_notes_03_Manual_Neural_Networks.md)
  - [CMU-DLSys Notes: The gradient(s) of a two-layer network / Backpropagation (Lecture3)](./course/AI/CMU_10-414-714/CMU-DLSys_notes_03_Backprop.md)
  - [CMU-DLSys Notes: 微分计算 / Reverse AD algorithm (Lecture4)](./course/AI/CMU_10-414-714/CMU-DLSys_notes_04_Automatic_Differentiation.md)
  - [CMU-DLSys Notes: Gradient descent / Newton’s Method (Lecture6)](./course/AI/CMU_10-414-714/CMU-DLSys_notes_05_Optimization_Part1.md)
  - [CMU-DLSys Notes: Momentum / Adam / Stochastic variants (Lecture6)](./course/AI/CMU_10-414-714/CMU-DLSys_notes_05_Optimization_Part2.md)
  - [CMU-DLSys Notes: Initialization / Normalization / Regularization (Lecture8)](./course/AI/CMU_10-414-714/CMU-DLSys_notes_06_Normalization.md)
  - [CMU-DLSys Notes: Neural Network Library Abstractions (Lecture7)](./course/AI/CMU_10-414-714/CMU-DLSys_notes_07_Neural_Network_Library_Abstractions.md)
  - [CMU-DLSys Notes: Convolutional Networks / Convolutions as matrix multiplication (Lecture10)](./course/AI/CMU_10-414-714/CMU-DLSys_notes_08_Convolutional_Networks.md)
- CV Notes
  - [CNN Backbone Notes：VGG / ResNet / MobileNet系列 / Inception系列](./course/AI/CV/CV_notes_01_经典CNN_Backbone.md)
  - [CNN Backbone Notes：ShuffleNet系列 / RepVGG / ResNet改进系列 / HRNet](./course/AI/CV/CV_notes_02_经典CNN_Backbone.md)
  - [Transformer Backbone Notes：ViT / Swin Transformer V1 & V2 / DeiT / DeiT-3](./course/AI/CV/CV_notes_03_经典Transformer_Backbone.md)
  - [Self-supervised Notes：MoCo系列 / SimCLR / BYOL / SWAV / DenseCL / SimSiam](./course/AI/CV/CV_notes_04_经典对比学习自监督算法.md)
  - [Self-supervised Notes：MAE / BeiT V1 & V2 / SimMIM / MixMIM / MFF](./course/AI/CV/CV_notes_05_经典MIM自监督算法.md)
  - [Multimodal Notes：CLIP / Chinese CLIP / BLIP V1 & V2](./course/AI/CV/CV_notes_06_经典多模态自监督算法.md)

## 2 Parallel Programming

- Related course
  - [Getting_Started_with_Accelerated_Computing_in_CUDA_C_C++](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-AC-01+V1/)
  - [CUDA MODE](https://github.com/gpu-mode/lectures)
- Related link
  - [CUDA-Learn-Notes](https://github.com/DefTruth/cuda-learn-notes)
  - [leetgpu](https://www.leetgpu.com/)
  - [cuda-math-api](https://docs.nvidia.com/cuda/cuda-math-api/index.html)
  - [triton-doc](https://triton-lang.cn/main/index.html)
- PMPP Notes 
  - [PMPP Notes: Volta/Turing/Ampere简介 / CUDA9到CUDA11演进 (Perface)](./course/CUDA/PMPP/PMPP_notes_01_Perface.md)
  - [PMPP Notes: Heterogeneous parallel computing / Challenges (Chapter 1)](./course/CUDA/PMPP/PMPP_notes_02_Introduce.md)
  - [PMPP Notes: Data Parallelism / CUDA C program structure / CUDA C Compilation (Chapter 2)](./course/CUDA/PMPP/PMPP_notes_03_Heterogeneous_data_parallel_computing.md)
  - [PMPP Notes: Chapter 2 Exercises / cudaError_t / CUDA C practice (Chapter 2)](./course/CUDA/PMPP/PMPP_notes_03_chapter_2_exercise.md)
  - [PMPP Notes: Multidimensional grid organization / Matrix multiplication (Chapter 3)](./course/CUDA/PMPP/PMPP_notes_04_Multidimensional_grids_and_data.md)
  - [PMPP Notes: Chapter 3 Exercises / Multidimensional grid practice (Chapter 3)](./course/CUDA/PMPP/PMPP_notes_04_chapter_3_exercise.md)
  - [PMPP Notes: SM / Block scheduling / Synchronization (Chapter 4)](./course/CUDA/PMPP/PMPP_notes_05_Architecture_of_a_modern_GPU.md)
  - [PMPP Notes: Warps / Control divergence / Latency tolerance / Occupancy (Chapter 4)](./course/CUDA/PMPP/PMPP_notes_05_Warps_and_SIMD_hardware.md)
  - [PMPP Notes: Chapter 4 Exercises / Divergent warps exercises / Occupancy exercises (Chapter 4)](./course/CUDA/PMPP/PMPP_notes_05_chapter_4_exercise.md)
  - [PMPP Notes: Roofline Model / CUDA memory types / Tiling (Chapter 5)](./course/CUDA/PMPP/PMPP_notes_06_Memory_architecture.md)
  - [PMPP Notes: Tiled matrix multiplication / Boundary checks / Memory and occupancy (Chapter 5)](./course/CUDA/PMPP/PMPP_notes_06_Tiled_matrix_multiplication.md)
  - [PMPP Notes: Chapter 5 Exercises / Tiling practice / Memory and occupancy practice (Chapter 5)](./course/CUDA/PMPP/PMPP_notes_06_chapter_5_exercise.md)
  - [PMPP Notes: Memory coalescing / Hiding memory latency / Bank Conflict (Chapter 6)](./course/CUDA/PMPP/PMPP_notes_07_Memory_coalescing.md)
  - [PMPP Notes: Thread coarsening / Thread coarsening (Chapter 6)](./course/CUDA/PMPP/PMPP_notes_07_Thread_coarsening.md)
  - [PMPP Notes: Chapter 6 Exercises / Coalesced access practice (Chapter 6)](./course/CUDA/PMPP/PMPP_notes_07_chapter_6_exercise.md)
  - [PMPP Notes: Convolution (Chapter 7)](./course/CUDA/PMPP/PMPP_notes_08_Convolution.md)
  - [PMPP Notes: Reduce (Chapter 10)](./course/CUDA/PMPP/PMPP_notes_09_Reduce.md)
  - [PMPP Notes: Parallel histogram (Chapter 9)](./course/CUDA/PMPP/PMPP_notes_10_Parallel_histogram.md)
  - [PMPP Notes: Stencil (Chapter 8)](./course/CUDA/PMPP/PMPP_notes_11_Stencil.md)
  - [PMPP Notes: Prefix sum (Chapter 11)](./course/CUDA/PMPP/PMPP_notes_12_Prefix_sum.md)
  - [PMPP Notes: Merge (Chapter 12)](./course/CUDA/PMPP/PMPP_notes_13_Merge.md)
  - [PMPP Notes: Sorting (Chapter 13)](./course/CUDA/PMPP/PMPP_notes_14_Sorting.md)
- CUDA MODE Notes
  - [CUDA MODE Notes: PMPP Ch1 Introduction / PMPP Ch2 Heterogeneous data parallel computing (Lecture 2)](./course/CUDA/CUDA_MODE/CUDA_MODE_notes_01_PMPP_Ch1-2.md) 
  - [CUDA MODE Notes: PMPP Ch3 Multidimensional grids and data (Lecture 2)](./course/CUDA/CUDA_MODE/CUDA_MODE_notes_01_PMPP_Ch3.md) 
  - [CUDA MODE Notes: Streaming Multiprocessor (SM) / ​Threads、Warps和Blocks (Lecture 4)](./course/CUDA/CUDA_MODE/CUDA_MODE_notes_02_PMPP_Ch4_Part1.md) 
  - [CUDA MODE Notes: Warp shuffle / Warp divergence / Getting good occupancy (Lecture 4)](./course/CUDA/CUDA_MODE/CUDA_MODE_notes_02_PMPP_Ch4_Part2.md) 
  - [CUDA MODE Notes: Memory access / GPU HBM / GeLU fuse / FlashAttention (Lecture 4)](./course/CUDA/CUDA_MODE/CUDA_MODE_notes_02_PMPP_Ch5.md) 
  - [CUDA MODE Notes: DRAM/SRAM / Latency / **Throughput** [关注Memory Throughput] / Occupancy / Arithmetic Intensity (Lecture 8)](./course/CUDA/CUDA_MODE/CUDA_MODE_notes_03_Cuda_Performance_Checklist_Part1.md) 
  - [CUDA MODE Notes: **Minimize control divergence** [关注Branch] / **Thread Coarsening** [关注Memory Throughput] / Privatization (Lecture 8)](./course/CUDA/CUDA_MODE/CUDA_MODE_notes_03_Cuda_Performance_Checklist_Part2.md) 
  - [CUDA MODE Notes: **Reduction** [关注Branch & L1/TEX Cache] (Lecture 9)](./course/CUDA/CUDA_MODE/CUDA_MODE_notes_03_Cuda_Performance_Checklist_Part2.md) 
- LeetGPU Notes
  - LeetGPU Notes: Vector Addition [cuda](./course/CUDA/LeetGPU/LeetGPU_notes_01_Vector_Addition_cuda.md),[triton](./course/CUDA/LeetGPU/LeetGPU_notes_01_Vector_Addition_triton.md)
  - LeetGPU Notes: ReLU / Leaky ReLU / Color Inversion / Matrix Copy / Matrix Transpose / Count 2D Array Element / Reverse Array [cuda](./course/CUDA/LeetGPU/LeetGPU_notes_02_easy_cuda_Part_1.md),[triton](./course/CUDA/LeetGPU/LeetGPU_notes_02_easy_triton_Part_1.md)
  - LeetGPU Notes: Count Array Element / Sigmoid / Swish-Gated / 1D Convolution / Rainbow Table / Matrix Multiplication [cuda](./course/CUDA/LeetGPU/LeetGPU_notes_02_easy_cuda_Part_2.md),[triton](./course/CUDA/LeetGPU/LeetGPU_notes_02_easy_triton_Part_2.md)
  - LeetGPU Notes: Reduce / Softmax [cuda](./course/CUDA/LeetGPU/LeetGPU_notes_03_medium_cuda_Part_1.md),[triton](./course/CUDA/LeetGPU/LeetGPU_notes_03_medium_triton_Part_1.md)
  - LeetGPU Notes: 2D Convolution / 2D Max Pooling [cuda](./course/CUDA/LeetGPU/LeetGPU_notes_03_medium_cuda_Part_2.md),[triton](./course/CUDA/LeetGPU/LeetGPU_notes_03_medium_triton_Part_2.md)
  - LeetGPU Notes: Batch Normalization / RMS Normalization [cuda](./course/CUDA/LeetGPU/LeetGPU_notes_03_medium_cuda_Part_3.md),[triton](./course/CUDA/LeetGPU/LeetGPU_notes_03_medium_triton_Part_3.md)
  - LeetGPU Notes: Histogramming / Dot Product / Mean Squared Error [cuda](./course/CUDA/LeetGPU/LeetGPU_notes_03_medium_cuda_Part_4.md),[triton](./course/CUDA/LeetGPU/LeetGPU_notes_03_medium_triton_Part_4.md)
- ARM NEON Notes
  - [ARM NEON基础使用: 基础数据类型 / 基本函数 / 编译和优化](./course/ARM/notes/ARM_NEON_notes_01_NEON基本操作.md)

## 3 Computer System 

- Related course
  - [ICS-PA2024](http://www.why.ink:8080/ICS/2024/Main_Page)
  - [NEMU](https://ysyx.oscc.cc/docs/ics-pa/)
  - [一生一芯](https://ysyx.oscc.cc/)
  - [NJU-OS](https://jyywiki.cn/OS/2025/)
  - [MIT-OS](https://pdos.csail.mit.edu/6.828/2024/index.html)
- ICS-PA2024 & YSYX Notes
  - [NJU-PA Notes: NEMU简介 / Linux和C语言拾遗 (Lecture1-3)](./course/Computer_System/notes/PA_ICS2024_notes_01_NEMU简介_Linux和C语言拾遗.md)
  - [NJU-PA Notes: NEMU编译运行 / NEMU代码导读 (Lecture4-5)](./course/Computer_System/notes/PA_ICS2024_notes_02_NEMU编译运行_NEMU代码导读.md)
  - [NJU-PA Notes: 数据的机器级表述 / ABI与内联汇编 (Lecture6-7)](./course/Computer_System/notes/PA_ICS2024_notes_03_数据的机器级表述_ABI与内联汇编.md)
  - [NJU-PA Notes: I/O设备 / 链接与加载 (Lecture8-9)](./course/Computer_System/notes/PA_ICS2024_notes_04_IO设备_链接与加载.md)
  - [NJU-PA Notes: 基础设施 / 中断 / 虚拟内存 (Lecture10/12/14)](./course/Computer_System/notes/PA_ICS2024_notes_05_基础设施_中断_虚拟内存.md)
  - [NJU-PA Notes: PA1 & PA2要点（NEMU框架 / 计算机是个抽象层）](./course/Computer_System/notes/YSYX_notes_01_PA1&PA2要点.md)
  - [NJU-PA Notes: RISC-V指令集基础（RISC-V寄存器 / 指令格式）](./course/Computer_System/notes/YSYX_notes_02_RISCV指令集基础.md)
  - [NJU-PA Notes: PA3要点（系统编程 / 虚拟文件系统）](./course/Computer_System/notes/YSYX_notes_03_PA3要点.md)
  - [NJU-PA Notes: PA4要点（虚存管理 / 用户进程切换）](./course/Computer_System/notes/YSYX_notes_04_PA4要点.md)
- NJU_OS2025 Notes
  - [NJU-OS Notes: 应用视角 / 硬件视角 / 数学视角的操作系统](./course/OS/NJU_OS2025/NJU_OS_notes_01_绪论.md)
  - [NJU-OS Notes: 进程与程序 / fork / execve](./course/OS/NJU_OS2025/NJU_OS_notes_02_进程与程序.md)
  - [NJU-OS Notes: 进程的地址空间 / mmap](./course/OS/NJU_OS2025/NJU_OS_notes_03_进程的地址空间.md)
  - [NJU-OS Notes: 访问操作系统中的对象 / pipe](./course/OS/NJU_OS2025/NJU_OS_notes_04_访问操作系统中的对象.md)
  - [NJU-OS Notes: 终端、进程组 / Ctrl-C](./course/OS/NJU_OS2025/NJU_OS_notes_05_终端、进程组.md)
  - [NJU-OS Notes: UNIX Shell / Freestanding Shell](./course/OS/NJU_OS2025/NJU_OS_notes_06_UNIX_Shell.md)
  - [NJU-OS Notes: libc / musl-gcc](./course/OS/NJU_OS2025/NJU_OS_notes_07_libc原理与实现.md)
  - [NJU-OS Notes: 可执行文件 / Core Dump / fle示例源码阅读](./course/OS/NJU_OS2025/NJU_OS_notes_08_可执行文件、静态链接和加载.md)
  - [NJU-OS Notes: 重定位差异 / ld-linux.so / -fPIC / LD_PRELOAD](./course/OS/NJU_OS2025/NJU_OS_notes_09_动态链接和加载.md)
  - [NJU-OS Notes: initramfs概述 / 最小Linux initramfs解读](./course/OS/NJU_OS2025/NJU_OS_notes_10_操作系统上的应用生态世界.md)
  - [NJU-OS Notes: 多线程栈4kB内存 / T_sum编译器行为](./course/OS/NJU_OS2025/NJU_OS_notes_11_多处理器编程.md)
  - [NJU-OS Notes: lock/unlock / pthread_mutex_t / sum-spin / sum-atomic](./course/OS/NJU_OS2025/NJU_OS_notes_12_并发编程：互斥.md)
  - [NJU-OS Notes: 条件变量 / 生产者-消费者示例源码阅读 / 同步机制实现计算图](./course/OS/NJU_OS2025/NJU_OS_notes_13_并发编程：条件变量.md)
  - [NJU-OS Notes: 使用互斥锁实现计算图 / 使用信号量实现生产者-消费者问题](./course/OS/NJU_OS2025/NJU_OS_notes_14_并发编程：信号量.md)
  - [NJU-OS Notes: 死锁 / Lock Ordering / lockdep示例源码解读 / Atomicity/Order Violation](./course/OS/NJU_OS2025/NJU_OS_notes_15_并发bug及应对.md)
  - [NJU-OS Notes: OpenMP基本 / Perf基本 / SIMD指令集发展史](./course/OS/NJU_OS2025/NJU_OS_notes_16_真实世界的并发编程.md)
  - [NJU-OS Notes: file_operations / launcher示例源码分析 / ioctl基本 / kvm示例源码解析](./course/OS/NJU_OS2025/NJU_OS_notes_17_设备和驱动程序.md)
  - [NJU-OS Notes: FLASH、DDR对比 / SSD/优盘/TF卡内计算机系统 / 块设备读写与BIO](./course/OS/NJU_OS2025/NJU_OS_notes_18_存储设备原理.md)
  - [NJU-OS Notes: linux目录树关联API / VFS层与文件系统驱动层分工 / mount/loopback协同](./course/OS/NJU_OS2025/NJU_OS_notes_19_目录树管理API.md)
  - [NJU-OS Notes: FAT文件系统 / ex2文件系统 / 崩溃一致性](./course/OS/NJU_OS2025/NJU_OS_notes_20_文件系统实现.md)
- MIT_6.S081 Notes
  - [MIT-OS Notes: XV6 Shell实现源码解析](./course/OS/MIT_6.S081/MIT_XV6_notes_01_XV6_Shell.md)
  - [MIT-OS Notes: XV6 Operating system organization (Chapter 2)](./course/OS/MIT_6.S081/MIT_XV6_notes_02_XV6_Operating_system_organization.md)
  - [MIT-OS Notes: XV6 System calls (Chapter 4)](./course/OS/MIT_6.S081/MIT_XV6_notes_03_XV6_System_calls.md)
  - [MIT-OS Notes: XV6 Page tables (Chapter 3)](./course/OS/MIT_6.S081/MIT_XV6_notes_04_XV6_Page_tables.md)
  - [MIT-OS Notes: XV6 Traps (Chapter 4)](./course/OS/MIT_6.S081/MIT_XV6_notes_05_XV6_Traps.md)
  - [MIT-OS Notes: XV6 Interrupts and device drivers (Chapter 5)](./course/OS/MIT_6.S081/MIT_XV6_notes_06_XV6_Interrupts_and_device_drivers.md)
  - [MIT-OS Notes: XV6 Locking (Chapter 6)](./course/OS/MIT_6.S081/MIT_XV6_notes_07_XV6_Locking.md)
  - [MIT-OS Notes: XV6 Buffer cache (Chapter 8)](./course/OS/MIT_6.S081/MIT_XV6_notes_08_XV6_Buffer_cache.md)
  - [MIT-OS Notes: XV6 Logging layer (Chapter 8)](./course/OS/MIT_6.S081/MIT_XV6_notes_08_XV6_Logging_layer.md)
  - [MIT-OS Notes: XV6 Block allocator / Inode layer (Chapter 8)](./course/OS/MIT_6.S081/MIT_XV6_notes_08_XV6_Inode_layer.md)
  - [MIT-OS Notes: XV6 Directory layer (Chapter 8)](./course/OS/MIT_6.S081/MIT_XV6_notes_08_XV6_Directory_layer.md)
  - [MIT-OS Notes: XV6 File descriptor layer (Chapter 8)](./course/OS/MIT_6.S081/MIT_XV6_notes_08_XV6_File_descriptor_layer.md)
  - [MIT-OS Notes: XV6 Multiplexing (Chapter 7)](./course/OS/MIT_6.S081/MIT_XV6_notes_09_XV6_Multiplexing.md)
  - [MIT-OS Notes: XV6 Sleep and wakeup (Chapter 7)](./course/OS/MIT_6.S081/MIT_XV6_notes_09_XV6_Sleep_wakeup.md)
  - [MIT-OS Notes: XV6 Pipes / Wait, exit, and kill / Process Locking (Chapter 7)](./course/OS/MIT_6.S081/MIT_XV6_notes_09_XV6_Pipes.md)
  - [MIT-OS Notes: XV6 Concurrency revisited (Chapter 9)](./course/OS/MIT_6.S081/MMIT_XV6_notes_10_XV6_Concurrency_revisited.md)

## 4 Compiler
- Related course
  - [USTC: Principles and Techniques of Compiler](https://ustc-compiler-principles.github.io/2023/)
  - [NJU Compilers](http://docs.compilers.cpl.icu/#/)
  - [LLVM IR Animation](https://blog.piovezan.ca/compilers/llvm_ir_animation/llvm_ir.html)
  - [LLVM Tutorial](https://llvm.org/docs/tutorial/)
- Related link
  - [LLVM](https://llvm.org/)
  - [MLIR](https://mlir.llvm.org/getting_started/)
  - [ONNX-MLIR](https://github.com/onnx/onnx-mlir)
- Compiler Principles Notes
  - [ANTLR 4基本操作](./course/Compiler/notes/Compiler_notes_01_ANTLR_4基本操作.md)
- LLVM / Clang Notes
  - [LLVM基本语法](./course/Compiler/notes/LLVM_notes_01_LLVM基本语法.md)

## 5 Deployment

- Related link 
  - [NCNN](https://github.com/Tencent/ncnn)
- NCNN notes
  - 基础使用
  [模型加载](./deployment/ncnn/notes/NCNN源码分析01-ncnn模型加载.md), 
  [内存管理](./deployment/ncnn/notes/NCNN源码分析02-CPU内存管理.md), 
  [类Mat](./deployment/ncnn/notes/NCNN源码分析03-类Mat.md), 
  [from_pixels](./deployment/ncnn/notes/NCNN源码分析04-图像处理函数之from_pixels.md), 
  [yuv420sp2rgb](./deployment/ncnn/notes/NCNN源码分析04-图像处理函数之yuv420sp2rgb.md), 
  [resize](./deployment/ncnn/notes/NCNN源码分析04-图像处理函数之resize.md), 
  [kanna_rotate](./deployment/ncnn/notes/NCNN源码分析04-图像处理函数之kanna_rotate.md)
  - operator
  [AbsVal](./deployment/ncnn/notes/NCNN源码分析05-激活函数之absval算子.md), 
  [Batchnorm](./deployment/ncnn/notes/NCNN源码分析05-激活函数之bn算子.md), 
  [Bias](./deployment/ncnn/notes/NCNN源码分析05-激活函数之bias算子.md), 
  [Binaryop](./deployment/ncnn/notes/NCNN源码分析05-激活函数之binaryop算子.md), 
  [Bnll](./deployment/ncnn/notes/NCNN源码分析05-激活函数之bnll算子.md), 
  [Clip](./deployment/ncnn/notes/NCNN源码分析05-激活函数之clip算子.md), 
  [ReLU](./deployment/ncnn/notes/NCNN源码分析05-激活函数之relu算子.md), 
  [convolutione](./deployment/ncnn/notes/NCNN源码分析06-convolution与convolutiondepthwise基础实现.md), [deconvolution](./deployment/ncnn/notes/NCNN源码分析06-deconvolution与deconvolutiondepthwise基础实现.md)

## 6 其他
- Related link
  - [CS自学指南](https://csdiy.wiki/)





