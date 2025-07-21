# HPC-Practice

## 1 AI 

- Related course
  - [10-414-714_Deep_Learning_Systems（2024）](https://dlsyscourse.org/)
  - [Foundations-of-LLMs](https://github.com/ZJU-LLMs/Foundations-of-LLMs)
- Related link
  - [mmpretrain](https://github.com/open-mmlab/mmpretrain/tree/main)
  - [transformers](https://github.com/huggingface/transformers)
- LLM Notes
  - [LLM基础: LLM架构 / Prompt工程 / 参数高效微调 / 模型编辑 / 检索增强生成](./course/AI/notes/LLM_notes_01_LLM基础.md)
  - [Hugging_Face Bert有关Embeddings及Tokenizer: Bert / Albert / Roberta / Electra](./course/AI/notes/LLM_notes_02_Hugging_Face_Bert有关Embeddings及Tokenizer.md)
  - [Hugging_Face Encoder-only代表性方法: Bert / Bert下游任务 / Albert / Roberta / Electra](./course/AI/notes/LLM_notes_03_Hugging_Face_Bert有关模型架构.md)
  - [Hugging_Face Encoder-Decoder代表性方法：T5 / BART](./course/AI/notes/LLM_notes_04_Hugging_Face_T5_BART.md)
- 10-414-714 Notes
  - [Lecture1-6: Softmax / Backprop / Automatic Differentiation / Optimization](./course/AI/notes/10-414-714_notes_01.md)
  - [Lecture7/9-11: NN Library Abstractions / Normalization and Regularization](./course/AI/notes/10-414-714_notes_02.md)
- CV Notes
  - [经典CNN Backbone之一：VGG / ResNet / MobileNet系列 / Inception系列](./course/AI/notes/CV_notes_01_经典CNN_Backbone.md)
  - [经典CNN Backbone之二：ShuffleNet系列 / RepVGG / ResNet改进系列 / HRNet](./course/AI/notes/CV_notes_02_经典CNN_Backbone.md)
  - [经典Transformer_Backbone：ViT / Swin Transformer V1 & V2 / DeiT / DeiT-3](./course/AI/notes/CV_notes_03_经典Transformer_Backbone.md)
  - [经典对比学习自监督算法：MoCo系列 / SimCLR / BYOL / SWAV / DenseCL / SimSiam](./course/AI/notes/CV_notes_04_经典对比学习自监督算法.md)
  - [经典MIM自监督算法：MAE / BeiT V1 & V2 / SimMIM / MixMIM / MFF](./course/AI/notes/CV_notes_05_经典MIM自监督算法.md)
  - [经典多模态自监督算法：CLIP / Chinese CLIP / BLIP V1 & V2](./course/AI/notes/CV_notes_06_经典多模态自监督算法.md)

## 2 Parallel Programming

- Related course
  - [Getting_Started_with_Accelerated_Computing_in_CUDA_C_C++](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-AC-01+V1/)
  - [CUDA MODE](https://github.com/gpu-mode/lectures)
- Related link
  - [CUDA-Learn-Notes](https://github.com/DefTruth/cuda-learn-notes)
- CUDA MODE Notes
  - [Lecture1/2/4: cuda基础](./course/CUDA/notes/cuda笔记01-cuda基础/01-cuda基础.md)
- CUDA Kernal Notes
  - [cuda-kernel-easy: 常见激活函数](./course/CUDA/notes/cuda源码分析01-cuda-kernel-easy/01-cuda-kernel-easy.md)
  - [cuda-kernel-medium: reduce及其应用 ](./course/CUDA/notes/cuda源码分析02-cuda-kernel-medium/02-cuda-kernel-medium.md) 
- ARM NEON Notes
  - [ARM NEON基础使用: 基础数据类型 / 基本函数 / 编译和优化](./course/ARM/notes/ARM_NEON_notes_01_NEON基本操作.md)

## 3 Computer System 
  
- Related course
  - [ICS-PA2024](http://www.why.ink:8080/ICS/2024/Main_Page)
  - [NEMU](https://ysyx.oscc.cc/docs/ics-pa/)
  - [一生一芯](https://ysyx.oscc.cc/)
  - [NJU-OS](https://jyywiki.cn/OS/2025/)
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
- OS Notes
  - [NJU-OS Notes: 应用视角 / 硬件视角 / 数学视角的操作系统](./course/OS/NJU_OS_notes_01_绪论.md)
  - [NJU-OS Notes: 进程与程序 / fork / execve](./course/OS/NJU_OS_notes_02_进程与程序.md)
  - [NJU-OS Notes: 进程的地址空间 / mmap](./course/OS/NJU_OS_notes_03_进程的地址空间.md)
  - [NJU-OS Notes: 访问操作系统中的对象 / pipe](./course/OS/NJU_OS_notes_04_访问操作系统中的对象.md)
  - [NJU-OS Notes: 终端、进程组 / Ctrl-C](./course/OS/NJU_OS_notes_05_终端、进程组.md)
  - [NJU-OS Notes: UNIX Shell / Freestanding Shell](./course/OS/NJU_OS_notes_06_UNIX_Shell.md)
  - [NJU-OS Notes: libc / musl-gcc](./course/OS/NJU_OS_notes_07_libc原理与实现.md)
  - [NJU-OS Notes: 可执行文件 / Core Dump / fle示例源码阅读](./course/OS/NJU_OS_notes_08_可执行文件、静态链接和加载.md)
  - [NJU-OS Notes: 重定位差异 / ld-linux.so / -fPIC / LD_PRELOAD](./course/OS/NJU_OS_notes_09_动态链接和加载.md)
  - [NJU-OS Notes: initramfs概述 / 最小Linux initramfs解读](./course/OS/NJU_OS_notes_10_操作系统上的应用生态世界.md)
  - [NJU-OS Notes: 多线程栈4kB内存 / T_sum编译器行为](./course/OS/NJU_OS_notes_11_多处理器编程.md)
  - [NJU-OS Notes: lock/unlock / pthread_mutex_t / sum-spin / sum-atomic](./course/OS/NJU_OS_notes_12_并发编程：互斥.md)
  - [NJU-OS Notes: 条件变量 / 生产者-消费者示例源码阅读 / 同步机制实现计算图](./course/OS/NJU_OS_notes_13_并发编程：条件变量.md)
  - [NJU-OS Notes: 使用互斥锁实现计算图 / 使用信号量实现生产者-消费者问题](./course/OS/NJU_OS_notes_14_并发编程：信号量.md)
  - [NJU-OS Notes: 死锁 / Lock Ordering / lockdep示例源码解读 / Atomicity/Order Violation](./course/OS/NJU_OS_notes_15_并发bug及应对.md)
  - [NJU-OS Notes: OpenMP基本 / Perf基本 / SIMD指令集发展史](./course/OS/NJU_OS_notes_16_真实世界的并发编程.md)
  - [NJU-OS Notes: file_operations / launcher示例源码分析 / ioctl基本 / kvm示例源码解析](./course/OS/NJU_OS_notes_17_设备和驱动程序.md)
  - [NJU-OS Notes: FLASH、DDR对比 / SSD/优盘/TF卡内计算机系统 / 块设备读写与BIO](./course/OS/NJU_OS_notes_18_存储设备原理.md)
  - [NJU-OS Notes: linux目录树关联API / VFS层与文件系统驱动层分工 / mount/loopback协同](./course/OS/NJU_OS_notes_19_目录树管理API.md)
  - [NJU-OS Notes: FAT文件系统 / ex2文件系统 / 崩溃一致性](./course/OS/NJU_OS_notes_20_文件系统实现.md)
  - [MIT-OS Notes: XV6 Shell实现源码解析](./course/OS/MIT_XV6_notes_01_XV6_Shell.md)

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





