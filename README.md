# HPC-Practice
HPC日常学习实践记录。

## LLVM部分
MLIR是多级IR表示，目的是通过多级IR表示提高编译框架的可扩展性和可重用性，详情参见[MLIR文档](https://mlir.llvm.org/getting_started/)。

Toy实践一：HPC-Practice/llvm-practice/toy/Ch2/，参考《MLIR入门教程-Toy扩展实践-1-添加一个新Op》<br>&emsp;&emsp;&emsp;&emsp;&emsp;(https://zhuanlan.zhihu.com/p/441237921)。

Toy实践二：HPC-Practice/llvm-practice/toy/Ch6/，参考《MLIR入门教程-Toy扩展实践-2-Interface使用》<br>&emsp;&emsp;&emsp;&emsp;&emsp;(https://zhuanlan.zhihu.com/p/441471026?utm_id=0)。

Toy实践三：HPC-Practice/llvm-practice/toy/Ch6/
          参考《MLIR-Toy-实践-3-Dialect转换》<br>&emsp;&emsp;&emsp;&emsp;&emsp;(https://zhuanlan.zhihu.com/p/444428735?utm_id=0<br>&emsp;&emsp;&emsp;&emsp;&emsp;《MLIR-Toy-实践-4-转换到LLVM IR运行》<br>&emsp;&emsp;&emsp;&emsp;&emsp;(https://zhuanlan.zhihu.com/p/447202920)<br>&emsp;&emsp;&emsp;&emsp;&emsp;《关于MLIR的学习实践分析与思考》<br>&emsp;&emsp;&emsp;&emsp;&emsp;(https://zhuanlan.zhihu.com/p/599281935)

## ONNX-MLIR部分
ONNX-MLIR将ONNX模型接入到MLIR中，并且提供了编译工具以及Python/C Runtime。

实践一：HPC-Practice/llvm-practice/onnx-mlir/mnist_example/，onnx-mlir例子<br>&emsp;&emsp;&emsp;（https://github.com/onnx/onnx-mlir/blob/main/docs/mnist_example/README.md）