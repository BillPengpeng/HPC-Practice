本文主要整理NEMU PA1及PA2的要点。

## 1 - NEMU是什么?

NEMU模拟了一个硬件的世界, 你可以在这个硬件世界中执行程序. 换句话说, 你将要在PA中编写一个用来执行其它程序的程序! 为了更好地理解NEMU的功能, 下面将

在GNU/Linux中运行Hello World程序
在GNU/Linux中通过红白机模拟器玩超级玛丽
在GNU/Linux中通过NEMU运行Hello World程序
这三种情况进行比较.

```
                         +---------------------+  +---------------------+
                         |     Super Mario     |  |    "Hello World"    |
                         +---------------------+  +---------------------+
                         |    Simulated NES    |  |      Simulated      |
                         |       hardware      |  |       hardware      |
+---------------------+  +---------------------+  +---------------------+
|    "Hello World"    |  |     NES Emulator    |  |        NEMU         |
+---------------------+  +---------------------+  +---------------------+
|      GNU/Linux      |  |      GNU/Linux      |  |      GNU/Linux      |
+---------------------+  +---------------------+  +---------------------+
|    Real hardware    |  |    Real hardware    |  |    Real hardware    |
+---------------------+  +---------------------+  +---------------------+
          (a)                      (b)                     (c)
```

图中(a)展示了"在GNU/Linux中运行Hello World"的情况. GNU/Linux操作系统直接运行在真实的计算机硬件上, 对计算机底层硬件进行了抽象, 同时向上层的用户程序提供接口和服务. Hello World程序输出信息的时候, 需要用到操作系统提供的接口, 因此Hello World程序并不是直接运行在真实的计算机硬件上, 而是运行在操作系统(在这里是GNU/Linux)上.

图中(b)展示了"在GNU/Linux中通过红白机模拟器玩超级玛丽"的情况. 在GNU/Linux看来, 运行在其上的红白机模拟器NES Emulator和上面提到的Hello World程序一样, 都只不过是一个用户程序而已. 神奇的是, 红白机模拟器的功能是负责模拟出一套完整的红白机硬件, 让超级玛丽可以在其上运行. 事实上, 对于超级玛丽来说, 它并不能区分自己是运行在真实的红白机硬件之上, 还是运行在模拟出来的红白机硬件之上, 这正是"模拟"的障眼法.

图中(c)展示了"在GNU/Linux中通过NEMU执行Hello World"的情况. 在GNU/Linux看来, 运行在其上的NEMU和上面提到的Hello World程序一样, 都只不过是一个用户程序而已. 但NEMU的功能是负责模拟出一套计算机硬件, 让程序可以在其上运行. 事实上, 上图只是给出了对NEMU的一个基本理解, 更多细节会在后续PA中逐渐补充.

## 2. AM - 裸机(bare-metal)运行时环境

AM(Abstract machine)项目就是这样诞生的. 作为一个向程序提供运行时环境的库, AM根据程序的需求把库划分成以下模块：
```
AM = TRM + IOE + CTE + VME + MPE
```
TRM(Turing Machine) - 图灵机, 最简单的运行时环境, 为程序提供基本的计算能力。  
IOE(I/O Extension) - 输入输出扩展, 为程序提供输出输入的能力。  
CTE(Context Extension) - 上下文扩展, 为程序提供上下文管理的能力。  
VME(Virtual Memory Extension) - 虚存扩展, 为程序提供虚存管理的能力。  
MPE(Multi-Processor Extension) - 多处理器扩展, 为程序提供多处理器通信的能力 (MPE超出了ICS课程的范围, 在PA中不会涉及)。  
AM给我们展示了程序与计算机的关系: 利用计算机硬件的功能实现AM, 为程序的运行提供它们所需要的运行时环境. 感谢AM项目的诞生, 让NEMU和程序的界线更加泾渭分明, 同时使得PA的流程更加明确:  

```
(在NEMU中)实现硬件功能 -> (在AM中)提供运行时环境 -> (在APP层)运行程序
(在NEMU中)实现更强大的硬件功能 -> (在AM中)提供更丰富的运行时环境 -> (在APP层)运行更复杂的程序
```

这个流程其实与PA1中开天辟地的故事遥相呼应: 先驱希望创造一个计算机的世界, 并赋予它执行程序的使命. 亲自搭建NEMU(硬件)和AM(软件)之间的桥梁来支撑程序的运行, 是"理解程序如何在计算机上运行"这一终极目标的不二选择。

## 3. NEMU框架

### 3.1 代码初探

```
ics2024
├── abstract-machine   # 抽象计算机
├── am-kernels         # 基于抽象计算机开发的应用程序
├── fceux-am           # 红白机模拟器
├── init.sh            # 初始化脚本
├── Makefile           # 用于工程打包提交
├── nemu               # NEMU
└── README.md
```

### 3.2 nemu

```
nemu
├── configs                    # 预先提供的一些配置文件
├── include                    # 存放全局使用的头文件
│   ├── common.h               # 公用的头文件
│   ├── config                 # 配置系统生成的头文件, 用于维护配置选项更新的时间戳
│   ├── cpu
│   │   ├── cpu.h
│   │   ├── decode.h           # 译码相关
│   │   ├── difftest.h
│   │   └── ifetch.h           # 取指相关
│   ├── debug.h                # 一些方便调试用的宏
│   ├── device                 # 设备相关
│   ├── difftest-def.h
│   ├── generated
│   │   └── autoconf.h         # 配置系统生成的头文件, 用于根据配置信息定义相关的宏
│   ├── isa.h                  # ISA相关
│   ├── macro.h                # 一些方便的宏定义
│   ├── memory                 # 访问内存相关
│   └── utils.h
├── Kconfig                    # 配置信息管理的规则
├── Makefile                   # Makefile构建脚本
├── README.md
├── resource                   # 一些辅助资源
├── scripts                    # Makefile构建脚本
│   ├── build.mk
│   ├── config.mk
│   ├── git.mk                 # git版本控制相关
│   └── native.mk
├── src                        # 源文件
│   ├── cpu
│   │   └── cpu-exec.c         # 指令执行的主循环
│   ├── device                 # 设备相关
│   ├── engine
│   │   └── interpreter        # 解释器的实现
│   ├── filelist.mk
│   ├── isa                    # ISA相关的实现
│   │   ├── mips32
│   │   ├── riscv32
│   │   ├── riscv64
│   │   └── x86
│   ├── memory                 # 内存访问的实现
│   ├── monitor
│   │   ├── monitor.c
│   │   └── sdb                # 简易调试器
│   │       ├── expr.c         # 表达式求值的实现
│   │       ├── sdb.c          # 简易调试器的命令处理
│   │       └── watchpoint.c   # 监视点的实现
│   ├── nemu-main.c            # 你知道的...
│   └── utils                  # 一些公共的功能
│       ├── log.c              # 日志文件相关
│       ├── rand.c
│       ├── state.c
│       └── timer.c
└── tools                      # 一些工具
    ├── fixdep                 # 依赖修复, 配合配置系统进行使用
    ├── gen-expr
    ├── kconfig                # 配置系统
    ├── kvm-diff
    ├── qemu-diff
    └── spike-diff
```

为了支持不同的ISA, 框架代码把NEMU分成两部分: ISA无关的基本框架和ISA相关的具体实现. NEMU把ISA相关的代码专门放在nemu/src/isa/目录下, 并通过nemu/include/isa.h提供ISA相关API的声明。

### 3.3 abstract-machine

```
abstract-machine
├── am                                  # AM相关
│   ├── include
│   │   ├── amdev.h
│   │   ├── am.h
│   │   └── arch                        # 架构相关的头文件定义
│   ├── Makefile
│   └── src
│       ├── mips
│       │   ├── mips32.h
│       │   └── nemu                    # mips32-nemu相关的实现
│       ├── native
│       ├── platform
│       │   └── nemu                    # 以NEMU为平台的AM实现
│       │       ├── include
│       │       │   └── nemu.h
│       │       ├── ioe                 # IOE
│       │       │   ├── audio.c
│       │       │   ├── disk.c
│       │       │   ├── gpu.c
│       │       │   ├── input.c
│       │       │   ├── ioe.c
│       │       │   └── timer.c
│       │       ├── mpe.c               # MPE, 当前为空
│       │       └── trm.c               # TRM
│       ├── riscv
│       │   ├── nemu                    # riscv32(64)相关的实现
│       │   │   ├── cte.c               # CTE
│       │   │   ├── start.S             # 程序入口
│       │   │   ├── trap.S
│       │   │   └── vme.c               # VME
│       │   └── riscv.h
│       └── x86
│           ├── nemu                    # x86-nemu相关的实现
│           └── x86.h
├── klib                                # 常用函数库
├── Makefile                            # 公用的Makefile规则
└── scripts                             # 构建/运行二进制文件/镜像的Makefile
    ├── isa
    │   ├── mips32.mk
    │   ├── riscv32.mk
    │   ├── riscv64.mk
    │   └── x86.mk
    ├── linker.ld                       # 链接脚本
    ├── mips32-nemu.mk
    ├── native.mk
    ├── platform
    │   └── nemu.mk
    ├── riscv32-nemu.mk
    ├── riscv64-nemu.mk
    └── x86-nemu.mk
```

整个AM项目分为两大部分:
- abstract-machine/am/ - 不同架构的AM API实现, abstract-machine/am/include/am.h列出了AM中的所有API。
- abstract-machine/klib/ - 一些架构无关的库函数, 方便应用程序的开发。

阅读abstract-machine/am/src/platform/nemu/trm.c中的代码, 你会发现只需要实现很少的API就可以支撑起程序在TRM上运行了:

- Area heap结构用于指示堆区的起始和末尾
- void putch(char ch)用于输出一个字符
- void halt(int code)用于结束程序的运行
- void _trm_init()用于进行TRM相关的初始化工作

堆区是给程序自由使用的一段内存区间, 为程序提供动态分配内存的功能. TRM的API只提供堆区的起始和末尾, 而堆区的分配和管理需要程序自行维护. 当然, 程序也可以不使用堆区, 例如dummy. 把putch()作为TRM的API是一个很有趣的考虑, 我们在不久的将来再讨论它, 目前我们暂不打算运行需要调用putch()的程序.

最后来看看halt(). halt()里面调用了nemu_trap()宏 (在abstract-machine/am/src/platform/nemu/include/nemu.h中定义), 这个宏展开之后是一条内联汇编语句. 内联汇编语句允许我们在C代码中嵌入汇编语句, 以riscv32为例, 宏展开之后将会得到:
```
asm volatile("mv a0, %0; ebreak" : :"r"(code));
```
显然, 这个宏的定义是和ISA相关的, 如果你查看nemu/src/isa/$ISA/inst.c, 你会发现这条指令正是那条特殊的nemu_trap! nemu_trap()宏还会把一个标识结束的结束码移动到通用寄存器中, 这样, 这段汇编代码的功能就和nemu/src/isa/$ISA/inst.c中nemu_trap的行为对应起来了: 通用寄存器中的值将会作为参数传给set_nemu_state(), 将halt()中的结束码设置到NEMU的monitor中, monitor将会根据结束码来报告程序结束的原因. 

### 3.4 am-kernels

am-kernels子项目用于收录一些可以在AM上运行的测试集和简单程序。

```
am-kernels
├── benchmarks                  # 可用于衡量性能的基准测试程序
│   ├── coremark
│   ├── dhrystone
│   └── microbench
├── kernels                     # 可展示的应用程序
│   ├── hello
│   ├── litenes                 # 简单的NES模拟器
│   ├── nemu                    # NEMU
│   ├── slider                  # 简易图片浏览器
│   ├── thread-os               # 内核线程操作系统
│   └── typing-game             # 打字小游戏
└── tests                       # 一些具有针对性的测试集
    ├── am-tests                # 针对AM API实现的测试集
    └── cpu-tests               # 针对CPU指令实现的测试集
```

### 3.5 编译流程

在让NEMU运行客户程序之前, 我们需要将客户程序的代码编译成可执行文件. 需要说明的是, 我们不能使用gcc的默认选项直接编译, 因为默认选项会根据GNU/Linux的运行时环境将代码编译成运行在GNU/Linux下的可执行文件. 但此时的NEMU并不能为客户程序提供GNU/Linux的运行时环境, 在NEMU中无法正确运行上述可执行文件, 因此我们不能使用gcc的默认选项来编译用户程序。

解决这个问题的方法是交叉编译. 我们需要在GNU/Linux下根据AM的运行时环境编译出能够在$ISA-nemu这个新环境中运行的可执行文件. 为了不让链接器ld使用默认的方式链接, 我们还需要提供描述$ISA-nemu的运行时环境的链接脚本. AM的框架代码已经把相应的配置准备好了, 上述编译和链接选项主要位于abstract-machine/Makefile 以及abstract-machine/scripts/目录下的相关.mk文件中. 编译生成一个可以在NEMU的运行时环境上运行的程序的过程大致如下:

- gcc将$ISA-nemu的AM实现源文件编译成目标文件, 然后通过ar将这些目标文件作为一个库, 打包成一个归档文件abstract-machine/am/build/am-$ISA-nemu.a
- gcc把应用程序源文件(如am-kernels/tests/cpu-tests/tests/dummy.c)编译成目标文件
- 通过gcc和ar把程序依赖的运行库(如abstract-machine/klib/)也编译并打包成归档文件
- 根据Makefile文件abstract-machine/scripts/$ISA-nemu.mk中的指示, 让ld根据链接脚本abstract-machine/scripts/linker.ld, 将上述目标文件和归档文件链接成可执行文件

根据上述链接脚本的指示, 可执行程序重定位后的节从0x100000或0x80000000开始 (取决于_pmem_start和_entry_offset的值), 首先是.text节, 其中又以abstract-machine/am/src/$ISA/nemu/start.S中自定义的entry节开始, 然后接下来是其它目标文件的.text节. 这样, 可执行程序起始处总是放置start.S的代码, 而不是其它代码, 保证客户程序总能从start.S开始正确执行. 链接脚本也定义了其它节(包括.rodata, .data, .bss)的链接顺序, 还定义了一些关于位置信息的符号, 包括每个节的末尾, 栈顶位置, 堆区的起始和末尾.

我们对编译得到的可执行文件的行为进行简单的梳理:
- 第一条指令从abstract-machine/am/src/$ISA/nemu/start.S开始, 设置好栈顶之后就跳转到abstract-machine/am/src/platform/nemu/trm.c的_trm_init()函数处执行.

```
.section entry, "ax"
.globl _start
.type _start, @function

_start:
  mv s0, zero
  la sp, _stack_pointer
  jal _trm_init


. = _pmem_start + _entry_offset;
  .text : {
    *(entry)
    *(.text*)
  } : text
```

- 在_trm_init()中调用main()函数执行程序的主体功能, main()函数还带一个参数。

```
void _trm_init() {
  int ret = main(mainargs);
  halt(ret);
}
```

- 从main()函数返回后, 调用halt()结束运行.

```
# define nemu_trap(code) asm volatile("mv a0, %0; ebreak" : :"r"(code))

INSTPAT("0000000 00001 00000 000 00000 11100 11", ebreak , N, NEMUTRAP(s->pc, R(10))); // R(10) is $a0
```

有了TRM这个简单的运行时环境, 我们就可以很容易地在上面运行各种"简单"的程序了. 当然, 我们也可以运行"不简单"的程序: 我们可以实现任意复杂的算法, 甚至是各种理论上可计算的问题, 都可以在TRM上解决.

### 3.6 库函数

把运行时环境分成两部分: 一部分是架构相关的运行时环境, 也就是我们之前介绍的AM; 另一部分是架构无关的运行时环境, 类似memcpy()这种常用的函数应该归入这部分, abstract-machine/klib/用于收录这些架构无关的库函数. klib是kernel library的意思, 用于提供一些兼容libc的基础功能. 框架代码在abstract-machine/klib/src/string.c和abstract-machine/klib/src/stdio.c 中列出了将来可能会用到的库函数。

## 4. 重新认识计算机: 计算机是个抽象层

|TRM  |	计算  |	内存申请  |	结束运行  |	打印信息  |
| :--:    | :--:    | :--:       | :--:    | :--:        |
|运行环境 |	- | malloc()/free()	| -	| printf() |
|AM API	 |-	| heap | halt()	| putch() |
|ISA接口  |	指令 |  物理内存地址空间	| nemu_trap指令	| I/O方式 |
|硬件模块 |	处理器 |  物理内存	| Monitor | 串口 |
|电路实现 |	cpu_exec()	| pmem[] | nemu_state | serial_io_handler() |

- 计算. 这是程序最基本的需求, 以至于它甚至不属于运行时环境和AM的范畴. 所有计算相关的代码(顺序语句, 分支, 循环, 函数调用等), 都会被编译器编译成功能等价的指令序列, 最终在CPU上执行. 在NEMU中, 我们通过cpu_exec()函数来实现"CPU执行指令"的功能.
- 内存申请. 有的程序需要在运行时刻动态地申请内存来使用. 和libc类似, klib提供了malloc()和free()来实现内存的动态管理(你将来会实现它们), 它们又会使用TRM中提供的API heap来获得堆区的起始和末尾. 而heap的区间又是由ISA-平台这个二元组对应的物理内存地址空间来决定的. 这一地址空间对应着物理内存的大小, 在NEMU中, 它就是大数组pmem[]的大小.
- 结束运行. 一般程序都会有结束运行的时候, TRM提供了一个halt()的API来实现这一功能. 由于这个需求过于简单, 因此无需运行时环境提供更复杂的接口. halt()的具体实现和ISA有关, 我们使用了人为添加的nemu_trap指令来实现这一点. 执行nemu_trap指令会让NEMU从CPU执行指令的循环中跳出, 返回到Monitor中, 这是通过设置Monitor中的一个状态变量nemu_state来实现的.
- 打印信息. 输出是程序的另一个基本需求. 程序可以调用klib中的printf()来输出, 它会通过TRM的API putch()来输出字符. 不同的ISA-平台有不同的字符输出方式, 在$ISA-nemu中, putch()通过I/O相关的指令把字符写入到串口, 最终在NEMU中通过serial_io_handler()将字符打印到终端. 关于输入输出的更多细节会在PA2的最后部分进行介绍.