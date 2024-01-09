# HPC-Practice

  **这里提供深度学习部署框架、工具链相关的汇总教程。**  

## 1. 推理框架

### 1.1 NCNN
- [NCNN源码分析01-ncnn模型加载](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析01-ncnn模型加载.md)
- [NCNN源码分析02-CPU内存管理](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析02-CPU内存管理.md)
- [NCNN源码分析03-类Mat](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析03-类Mat.md)
- NCNN源码分析04-图像处理函数，
[from_pixels](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析04-图像处理函数之from_pixels.md)，
[yuv420sp2rgb](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析04-图像处理函数之yuv420sp2rgb.md)，
[resize](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析04-图像处理函数之resize.md)，
[kanna_rotate](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析04-图像处理函数之kanna_rotate.md)
- NCNN源码分析05-基础简单算子，
[AbsVal](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析05-激活函数之absval算子.md)，
[Batchnorm](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析05-激活函数之bn算子.md)，
[Bias](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析05-激活函数之bias算子.md)，
[Binaryop](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析05-激活函数之binaryop算子.md)，
[Bnll](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析05-激活函数之bnll算子.md)，
[Clip](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析05-激活函数之clip算子.md)，
[ReLU](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析05-激活函数之relu算子.md)

| Name |  one_blob_only | support_inplace | support_packing | support_bf16_storage | int8 forward | 
| ---  |  --- | --- | --- | --- | --- |
| AbsVal       |     True      |      True       |      True     |      False      |      ?       |
| Batchnorm    |     True      |      True       |      True     |      True       |      ?       |
| Bias         |     True      |      True       |      False    |      False      |      ?       |
| Binaryop     |     False     |      False      |      True     |      True       |      ?       |
| Bnll         |     True      |      True       |      True     |      False      |      ?       |
| Clip         |     True      |      True       |      True     |      True       |      ?       |
| ReLU         |     True      |      True       |      True     |      True       |     True     |

- NCNN源码分析06-密集计算算子，
[convolution与convolutiondepthwise基础实现](https://github.com/BillPengpeng/HPC-Practice/tree/master/notes/ncnn/NCNN源码分析06-convolution与convolutiondepthwise基础实现.md)

## 2. 部署工具链

### 2.1 MLIR

[MLIR](https://mlir.llvm.org/getting_started/)是多级IR表示，目的是通过多级IR表示提高编译框架的可扩展性和可重用性。

- [Toy实践一](https://github.com/BillPengpeng/HPC-Practice/tree/master/llvm-practice/toy/Ch2)，参考[MLIR入门教程-Toy扩展实践-1-添加一个新Op](https://zhuanlan.zhihu.com/p/441237921)
- [Toy实践二](https://github.com/BillPengpeng/HPC-Practice/tree/master/llvm-practice/toy/Ch6)，参考[MLIR入门教程-Toy扩展实践-2-Interface使用](https://zhuanlan.zhihu.com/p/441471026?utm_id=0)
- [Toy实践三](https://github.com/BillPengpeng/HPC-Practice/tree/master/llvm-practice/toy/Ch6)，参考[MLIR-Toy-实践-4-转换到LLVM IR运行](https://zhuanlan.zhihu.com/p/447202920)、[关于MLIR的学习实践分析与思考](https://zhuanlan.zhihu.com/p/599281935)

### 2.2 ONNX-MLIR
ONNX-MLIR将ONNX模型接入到MLIR中，并且提供了编译工具以及Python/C Runtime。

- [ONNX-MLIR实践一](https://github.com/BillPengpeng/HPC-Practice/tree/master/onnx-mlir/mnist_example)，参考[onnx-mlir例子](https://github.com/onnx/onnx-mlir/blob/main/docs/mnist_example/README.md)

