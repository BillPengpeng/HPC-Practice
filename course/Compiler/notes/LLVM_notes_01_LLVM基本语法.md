
## 1. LLVM 工具链核心组件
| **工具**      | **用途**                                | **常用场景**                     |
|---------------|-----------------------------------------|----------------------------------|
| `clang`       | C/C++ 前端编译器，生成 LLVM IR          | 源码转 IR、预处理、语法分析       |
| `opt`         | IR 优化器，应用优化 Pass                | 代码优化、Pass 调试              |
| `llc`         | IR 到目标代码的后端编译器               | 生成汇编或目标文件               |
| `lli`         | 直接执行 LLVM IR（JIT 编译）            | 快速测试 IR 代码                 |
| `llvm-dis`    | 将二进制 IR（.bc）转换为文本 IR（.ll）  | 查看二进制 IR 内容               |
| `llvm-as`     | 将文本 IR（.ll）转换为二进制 IR（.bc）  | 压缩和快速加载 IR                |
| `llvm-link`   | 链接多个 IR 文件                        | 模块化编译                      |
| `llvm-objdump`| 反汇编目标文件                          | 查看生成的目标代码               |

## 2. LLVM基本操作流程**

### 2.1 词法分析（Clang）
Clang 的 `-Xclang -dump-tokens` 可直接输出词法分析结果：
```bash
clang -fsyntax-only -Xclang -dump-tokens input.c
```
-fsyntax-only：这个选项告诉 Clang 只进行语法分析，不进行代码生成、优化或链接。它主要用于检查源代码的语法正确性，而不会生成任何输出文件。  
-Xclang：这个选项允许你向 Clang 传递一个特定的、Clang 专有的命令行参数。在这个上下文中，它用于传递 -dump-tokens 参数，该参数不是标准的 Clang 选项，而是需要通过 -Xclang 传递给 Clang 的内部选项。  
-dump-tokens：这是一个 Clang 的内部选项，用于在语法分析过程中输出源代码的词法单元。每个词法单元都表示源代码中的一个基本语法元素，如关键字、标识符、运算符、分隔符或常数。

### 2.2 语法分析（Clang AST）
```bash
clang -fsyntax-only -Xclang -ast-dump  input.c
```
-ast-dump：这是 Clang 的一个内部选项，用于在语法分析过程中输出源代码的AST。AST是源代码的抽象表示，它捕捉了源代码的结构和语法元素之间的关系。

### 2.3 中间代码生成

| **操作**             | **命令**                                 | **输出文件**       |
|----------------------|-----------------------------------------|--------------------|
| 生成文本格式 IR       | `clang -S -emit-llvm input.c -o input.ll` | `.ll`（可读）      |
| 生成二进制格式 IR     | `clang -c -emit-llvm input.c -o input.bc` | `.bc`（二进制）    |
| 优化 IR              | `opt -S -O3 input.ll -o optimized.ll`   | 优化后的 `.ll`     |
| 生成可执行文件        | `clang input.ll -o output`              | 可执行文件         |

### 2.4 查看/转换 IR

example.ll示例
```
define i32 @sum(i32 %a, i32 %b) {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}

define i32 @main() {
entry:
  %x = call i32 @sum(i32 3, i32 5)
  ret i32 %x
}
```

```bash
# 查看文本 IR
cat output.ll

# 二进制 IR 转文本 IR
llvm-dis output.bc -o output.ll

# 文本 IR 转二进制 IR
llvm-as output.ll -o output.bc
```

### 2.5 优化 IR
```bash
# 应用默认优化 (-O3)
opt -S -O3 input.ll -o optimized.ll

# 指定优化 Pass（如内联、循环展开）
opt -S -passes='inline,loop-unroll' input.ll -o optimized.ll
```

### 2.6 生成目标代码
```bash
# 生成汇编代码
llc -O3 input.ll -o output.s

# 直接生成目标文件
llc -filetype=obj input.ll -o output.o
```

### 2.7 链接与执行
```bash
# 链接 IR 生成可执行文件
clang output.ll -o output

# 直接执行 IR（JIT 模式）
lli output.ll
```

## 3. 关键调试与分析操作
### 3.1. 打印优化过程
```bash
# 打印所有优化 Pass 后的 IR
opt -S -O3 -print-after-all input.ll -o /dev/null

# 打印特定 Pass 后的 IR（如内联）
opt -S -passes=inline -print-after=inline input.ll
```

### 3.2. 查看控制流图（CFG）
```bash
# 生成 DOT 格式的 CFG
opt -dot-cfg input.ll -disable-output

# 转换为 PNG（需安装 Graphviz）
dot -Tpng .main.dot -o cfg.png
```

### 3.3. 反汇编目标代码**
```bash
# 查看目标文件的汇编代码
llvm-objdump -d output.o
```

## 3. LLVM IR语法**

LLVM IR（Intermediate Representation）是LLVM编译器框架的中间表示形式，其语法设计强调可读性、强类型化和静态单赋值（SSA）形式。

### 3.1. 模块（Module）
LLVM IR的顶层结构是**模块**，对应一个完整的编译单元（如一个C/C++文件）。模块包含：
- **全局变量**
- **函数声明/定义**
- **元数据**（调试信息等）
- **目标平台信息**（可选）

示例：
```llvm
; 模块级全局变量
@global_var = global i32 42, align 4

; 函数声明（外部函数）
declare i32 @external_func(ptr)

; 函数定义
define i32 @add(i32 %a, i32 %b) {
  ; 函数体...
}
```

### 3.2. 函数（Function）
函数定义格式：
```llvm
define [返回类型] @函数名(参数列表) [属性] {
  ; 基本块（Basic Blocks）
}
```

- **参数列表**：每个参数需显式指定类型，例如 `i32 %a, ptr %b`
- **属性**：如 `nounwind`（不抛出异常）、`optnone`（禁用优化）

示例：
```llvm
define i32 @max(i32 %x, i32 %y) {
entry:
  %cmp = icmp sgt i32 %x, %y
  br i1 %cmp, label %ret_x, label %ret_y

ret_x:
  ret i32 %x

ret_y:
  ret i32 %y
}
```

### 3.3. 基本块（Basic Block）
- 每个基本块以**标签**（如 `label:`）开头，包含一系列指令。
- 最后一个指令必须是**终止指令**（如 `br`, `ret`, `switch`）。
- 符合SSA（静态单赋值）原则：每个变量只能赋值一次。

示例：
```llvm
loop:
  %i = phi i32 [0, %entry], [%i.next, %loop]  ; Phi节点合并不同路径的值
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, 10
  br i1 %cond, label %loop, label %exit
```

### 3.4. 指令格式
LLVM IR指令遵循以下模式：
```llvm
%结果变量 = 操作码 类型 操作数 [, 附加属性]
```
- **操作码**：如 `add`, `load`, `call` 等。
- **类型**：显式指定操作数/结果的类型（如 `i32`, `ptr`, `<4 x float>`）。
- **操作数**：变量、常量或立即数。

示例：
```llvm
%sum = add i32 %a, %b          ; 整数加法
%val = load i32, ptr %ptr      ; 从指针加载值
call void @print(i32 %sum)     ; 调用函数
```

### 3.5. 类型系统
LLVM IR是强类型的，常见类型包括：
| 类型              | 示例                 | 说明                     |
|-------------------|----------------------|--------------------------|
| **整数类型**      | `i32`, `i1`         | `iN` 表示N位整数         |
| **浮点类型**      | `float`, `double`    | 单精度/双精度浮点        |
| **指针类型**      | `ptr`, `i32*`        | 指向其他类型的指针       |
| **数组类型**      | `[4 x i32]`          | 固定长度数组             |
| **结构体类型**    | `{i32, float}`       | 类似C的结构体            |
| **向量类型**      | `<4 x float>`        | SIMD向量（如AVX指令集）  |

### 3.6. 常量
- **整型常量**：`i32 42`, `i8 -10`
- **浮点常量**：`double 3.1415`, `float 1.0`
- **全局变量地址**：`ptr @global_var`
- **字符串常量**：`@str = constant [6 x i8] c"hello\00"`

### 3.7. 控制流指令
- **分支**：`br i1 %cond, label %true_block, label %false_block`
- **返回**：`ret i32 %result` 或 `ret void`
- **Switch**：
  ```llvm
  switch i32 %val, label %default [
    i32 0, label %case0
    i32 1, label %case1
  ]
  ```

### 3.8. 内存操作
- **分配栈空间**：`%ptr = alloca i32, align 4`
- **加载/存储**：
  ```llvm
  %val = load i32, ptr %ptr     ; 加载
  store i32 42, ptr %ptr       ; 存储
  ```
- **地址计算**（`getelementptr`）：
  ```llvm
  ; 计算数组第3个元素的地址
  %elem_ptr = getelementptr inbounds [10 x i32], ptr %arr, i64 0, i64 2
  ```


### 3.9. 元数据与属性
- **元数据**：用于调试信息或优化提示：
  ```llvm
  call void @func(), !dbg !123  ; 附加调试元数据
  ```
- **函数属性**：如 `noinline`, `optnone`：
  ```llvm
  define void @foo() noinline { ... }
  ```

### 3.10. 完整示例
```llvm
; 模块级定义
@message = constant [12 x i8] c"Hello IR!\0A\00"

define i32 @main() {
entry:
  ; 调用C标准库的printf
  %str = getelementptr inbounds [12 x i8], ptr @message, i32 0, i32 0
  %ret = call i32 (ptr, ...) @printf(ptr %str)
  ret i32 0
}

; 声明外部函数
declare i32 @printf(ptr, ...)
```

## 4. LLVM IR常见指令**

### 4.1. 算术与逻辑运算
- **整数运算**  
  `add`, `sub`, `mul`, `udiv` (无符号除), `sdiv` (有符号除), `urem` (无符号取余), `srem` (有符号取余)
  ```llvm
  %result = add i32 %a, %b   ; 32位整数相加
  ```

- **浮点运算**  
  `fadd`, `fsub`, `fmul`, `fdiv`, `frem`
  ```llvm
  %result = fadd double %x, %y  ; 双精度浮点相加
  ```

- **位运算**  
  `shl` (左移), `lshr` (逻辑右移), `ashr` (算术右移), `and`, `or`, `xor`
  ```llvm
  %result = shl i8 %val, 3  ; 左移3位
  ```

### 4.2. 比较操作
- **整数比较** (`icmp`)  
  条件码如 `eq`, `ne`, `slt` (有符号小于), `ult` (无符号小于)
  ```llvm
  %cmp = icmp slt i32 %a, %b  ; 有符号比较 a < b
  ```

- **浮点比较** (`fcmp`)  
  条件码如 `oeq` (有序等于), `ult` (无序小于)
  ```llvm
  %cmp = fcmp oeq float %x, %y  ; 浮点数是否相等
  ```

### 4.3. 内存操作
- **分配栈空间** (`alloca`)  
  ```llvm
  %ptr = alloca i32, align 4  ; 分配4字节对齐的i32栈空间
  ```

- **加载/存储** (`load`, `store`)  
  ```llvm
  %val = load i32, ptr %ptr     ; 从指针加载值
  store i32 42, ptr %ptr        ; 存储42到指针
  ```

- **计算地址** (`getelementptr`, GEP)  
  用于结构体/数组元素地址计算：
  ```llvm
  ; 计算结构体%s的第2个字段地址（假设类型为{ i32, float }）
  %field_ptr = getelementptr inbounds {i32, float}, ptr %s, i32 0, i32 1
  ```

### 4.4. 控制流
- **无条件跳转** (`br`)  
  ```llvm
  br label %next_block  ; 跳转到标签%next_block
  ```

- **条件跳转** (`br i1`)  
  ```llvm
  br i1 %condition, label %true_block, label %false_block
  ```

- **函数返回** (`ret`)  
  ```llvm
  ret i32 0        ; 返回0
  ret void         ; 无返回值
  ```

- **Switch跳转** (`switch`)  
  ```llvm
  switch i32 %val, label %default [i32 0, label %case0
                                   i32 1, label %case1]
  ```

- **Phi节点** (`phi`)  
  用于合并不同基本块的值（SSA形式）：
  ```llvm
  %result = phi i32 [0, %entry], [%x, %loop]
  ```

### 4.5. 函数调用
- **普通调用** (`call`)  
  ```llvm
  %ret = call i32 @func(ptr %arg1, i64 %arg2)
  ```

- **尾调用优化** (`tail call`)  
  ```llvm
  tail call void @foo()  ; 提示编译器进行尾调用优化
  ```

### 4.6. 类型转换
- **整数扩展/截断**  
  `zext` (零扩展), `sext` (符号扩展), `trunc` (截断)
  ```llvm
  %ext = zext i8 %byte to i32  ; 8位零扩展到32位
  ```

- **浮点转换**  
  `fptrunc` (双精度转单精度), `fpext` (单精度转双精度)
  ```llvm
  %double = fpext float %x to double
  ```

- **指针与整数互转**  
  `ptrtoint`, `inttoptr`
  ```llvm
  %addr = ptrtoint ptr %ptr to i64
  ```

### 4.7. 向量化操作
- **插入/提取元素**  
  `insertelement`, `extractelement`
  ```llvm
  %vec = insertelement <4 x i32> %old, i32 42, i32 2  ; 在第2位插入42
  ```

- **向量混洗** (`shufflevector`)  
  ```llvm
  %newvec = shufflevector <4 x i32> %v1, <4 x i32> %v2, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ```

### 4.8. 异常处理
- **异常调用** (`invoke`)  
  结合 `landingpad` 块处理异常：
  ```llvm
  %result = invoke i32 @may_throw() to label %normal unwind label %exception
  ```

### 4.9. 其他操作
- **原子操作** (`atomicrmw`, `cmpxchg`)  
  用于多线程同步：
  ```llvm
  atomicrmw add ptr %ptr, i32 1 seq_cst  ; 原子加1
  ```

- **内联汇编** (`asm`)  
  ```llvm
  call void asm sideeffect "nop", ""()  ; 插入汇编指令
  ```

### **4.10 示例代码片段**
```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
```

