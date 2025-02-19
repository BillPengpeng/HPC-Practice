# HPC-Practice

## 1. 深度学习部署

### 1.1 NCNN [link](https://github.com/Tencent/ncnn)
- 基础使用，
[模型加载](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析01-ncnn模型加载.md)，
[内存管理](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析02-CPU内存管理.md)，
[类Mat](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析03-类Mat.md)
- 图像处理函数，
[from_pixels](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析04-图像处理函数之from_pixels.md)，
[yuv420sp2rgb](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析04-图像处理函数之yuv420sp2rgb.md)，
[resize](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析04-图像处理函数之resize.md)，
[kanna_rotate](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析04-图像处理函数之kanna_rotate.md)
- 基础简单算子，
[AbsVal](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析05-激活函数之absval算子.md)，
[Batchnorm](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析05-激活函数之bn算子.md)，
[Bias](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析05-激活函数之bias算子.md)，
[Binaryop](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析05-激活函数之binaryop算子.md)，
[Bnll](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析05-激活函数之bnll算子.md)，
[Clip](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析05-激活函数之clip算子.md)，
[ReLU](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析05-激活函数之relu算子.md)
- 密集计算算子，
[convolutione](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析06-convolution与convolutiondepthwise基础实现.md)，
[deconvolution](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ncnn/notes/NCNN源码分析06-deconvolution与deconvolutiondepthwise基础实现.md)

### 1.2 MLIR [link](https://mlir.llvm.org/getting_started/)

- 基本实践，
[Toy扩展](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ai-compiler/llvm-practice/toy/Ch6)，参考博文：[MLIR入门教程-Toy扩展实践-1-添加一个新Op](https://zhuanlan.zhihu.com/p/441237921)、[MLIR入门教程-Toy扩展实践-2-Interface使用](https://zhuanlan.zhihu.com/p/441471026?utm_id=0)、[MLIR-Toy-实践-4-转换到LLVM IR运行](https://zhuanlan.zhihu.com/p/447202920)、[关于MLIR的学习实践分析与思考](https://zhuanlan.zhihu.com/p/599281935)

### 1.3 ONNX-MLIR [link](https://github.com/onnx/onnx-mlir)

- 基本实践，
[mnist_example](https://github.com/BillPengpeng/HPC-Practice/tree/master/deployment/ai-compiler/onnx-mlir/mnist_example)

## 2. 国内外相关课程

### 2.1 AI 

- Related course，[10-414-714_Deep_Learning_Systems](https://dlsyscourse.org/)，[Foundations-of-LLMs](https://github.com/ZJU-LLMs/Foundations-of-LLMs)
- LLM Notes，[LLM基础](https://github.com/BillPengpeng/HPC-Practice/tree/master/course/AI/notes/LLM_notes_01_LLM基础.md)
- 10-414-714 Notes，[Lecture1-6](https://github.com/BillPengpeng/HPC-Practice/tree/master/course/AI/notes/10-414-714_notes_01.md),
                    [Lecture7/9-11](https://github.com/BillPengpeng/HPC-Practice/tree/master/course/AI/notes/10-414-714_notes_02.md)

### 2.2 CUDA

- Related course，[Getting_Started_with_Accelerated_Computing_in_CUDA_C_C++](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-AC-01+V1/)，[CUDA MODE](https://github.com/gpu-mode/lectures)
- Related github, [CUDA-Learn-Notes](https://github.com/DefTruth/cuda-learn-notes)
- CUDA MODE Notes, [Lecture1/2/4](https://github.com/BillPengpeng/HPC-Practice/tree/master/course/CUDA/notes/cuda笔记01-cuda基础/01-cuda基础.ipynb)


