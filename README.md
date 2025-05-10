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
  - [Lecture1-3: NEMU简介 / Linux和C语言拾遗](./course/Computer_System/notes/PA_ICS2024_notes_01_NEMU简介_Linux和C语言拾遗.md)
  - [Lecture4-5: NEMU编译运行 / NEMU代码导读](./course/Computer_System/notes/PA_ICS2024_notes_02_NEMU编译运行_NEMU代码导读.md)
  - [Lecture6-7: 数据的机器级表述 / ABI与内联汇编](./course/Computer_System/notes/PA_ICS2024_notes_03_数据的机器级表述_ABI与内联汇编.md)
  - [Lecture8-9: I/O设备 / 链接与加载](./course/Computer_System/notes/PA_ICS2024_notes_04_IO设备_链接与加载.md)
  - [Lecture10/12/14: 基础设施 / 中断 / 虚拟内存](./course/Computer_System/notes/PA_ICS2024_notes_05_基础设施_中断_虚拟内存.md)
  - [YSYX Notes: PA1 & PA2要点（NEMU框架 / 计算机是个抽象层）](./course/Computer_System/notes/YSYX_notes_01_PA1&PA2要点.md)
  - [YSYX Notes: RISC-V指令集基础（RISC-V寄存器 / 指令格式）](./course/Computer_System/notes/YSYX_notes_02_RISCV指令集基础.md)
  - [YSYX Notes: PA3要点（系统编程 / 虚拟文件系统）](./course/Computer_System/notes/YSYX_notes_03_PA3要点.md)
  - [YSYX Notes: PA4要点（虚存管理 / 用户进程切换）](./course/Computer_System/notes/YSYX_notes_04_PA4要点.md)
- OS Notes
  - [NJU-OS Notes: 应用视角 / 硬件视角 / 数学视角的操作系统](./course/OS/NJU_OS_notes_01_绪论要点.md)
  - [NJU-OS Notes: 进程与程序 / fork / execve](./course/OS/NJU_OS_notes_02_进程与程序要点.md)

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





