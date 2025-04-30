本文主要整理NEMU PA3的要点。

## 1 自陷

为了实现最简单的操作系统, 硬件还需要提供一种可以限制入口的执行流切换方式. 这种方式就是自陷指令, 程序执行自陷指令之后, 就会陷入到操作系统预先设置好的跳转目标. 这个跳转目标也称为异常入口地址。

**riscv32**
riscv32提供ecall指令作为自陷指令, 并提供一个mtvec寄存器来存放异常入口地址. 为了保存程序当前的状态, riscv32提供了一些特殊的系统寄存器, 叫控制状态寄存器(CSR寄存器). 在PA中, 我们只使用如下3个CSR寄存器:
- mepc寄存器 - 存放触发异常的PC
- mstatus寄存器 - 存放处理器的状态
- mcause寄存器 - 存放触发异常的原因
riscv32触发异常后硬件的响应过程如下:
- 将当前PC值保存到mepc寄存器
- 在mcause寄存器中设置异常号
- 从mtvec寄存器中取出异常入口地址
- 跳转到异常入口地址

## 2 将上下文管理抽象成CTE

**设置异常入口地址**

CTE定义了名为"事件"的如下数据结构(见abstract-machine/am/include/am.h):
```
typedef struct Event {
  enum { ... } event;
  uintptr_t cause, ref;
  const char *msg;
} Event;
```
其中event表示事件编号, cause和ref是一些描述事件的补充信息, msg是事件信息字符串, 我们在PA中只会用到event. 然后, 我们只要定义一些统一的事件编号(上述枚举常量), 让每个架构在实现各自的CTE API时, 都统一通过上述结构体来描述执行流切换的原因, 就可以实现切换原因的抽象了。

还有另外两个统一的API:
- bool cte_init(Context* (*handler)(Event ev, Context *ctx))用于进行CTE相关的初始化操作. 其中它还接受一个来自操作系统的事件处理回调函数的指针, 当发生事件时, CTE将会把事件和相关的上下文作为参数, 来调用这个回调函数, 交由操作系统进行后续处理.
- void yield()用于进行自陷操作, 会触发一个编号为EVENT_YIELD事件. 不同的ISA会使用不同的自陷指令来触发自陷操作, 具体实现请RTFSC.

**保存上下文**

成功跳转到异常入口地址之后, 我们就要在软件上开始真正的异常处理过程了. 但是, 进行异常处理的时候不可避免地需要用到通用寄存器, 然而看看现在的通用寄存器, 里面存放的都是执行流切换之前的内容. 这些内容也是上下文的一部分, 如果不保存就覆盖它们, 将来就无法恢复这一上下文了. 但通常硬件并不负责保存它们, 因此需要通过软件代码来保存它们的值. x86提供了pusha指令, 用于把通用寄存器的值压栈; 而mips32和riscv32则通过sw指令将各个通用寄存器依次压栈.

除了通用寄存器之外, 上下文还包括:
- 触发异常时的PC和处理器状态. 对于x86来说就是eflags, cs和eip, x86的异常响应机制已经将它们保存在堆栈上了; 对于mips32和riscv32来说, 就是epc/mepc和status/mstatus寄存器, 异常响应机制把它们保存在相应的系统寄存器中, 我们还需要将它们从系统寄存器中读出, 然后保存在堆栈上.
- 异常号. 对于x86, 异常号由软件保存; 而对于mips32和riscv32, 异常号已经由硬件保存在cause/mcause寄存器中, 我们还需要将其保存在堆栈上.
- 地址空间. 这是为PA4准备的, 在x86中对应的是CR3寄存器, 代码通过一条pushl $0指令在堆栈上占位, mips32和riscv32则是将地址空间信息与0号寄存器共用存储空间, 反正0号寄存器的值总是0, 也不需要保存和恢复. 不过目前我们暂时不使用地址空间信息, 你目前可以忽略它们的含义.

**事件分发**

__am_irq_handle()的代码会把执行流切换的原因打包成事件, 然后调用在cte_init()中注册的事件处理回调函数, 将事件交给yield test来处理. 在yield test中, 这一回调函数是am-kernels/tests/am-tests/src/tests/intr.c中的simple_trap()函数. simple_trap()函数会根据事件类型再次进行分发. 不过我们在这里会触发一个未处理的事件:
```
AM Panic: Unhandled event @ am-kernels/tests/am-tests/src/tests/intr.c:12
```
这是因为CTE的__am_irq_handle()函数并未正确识别出自陷事件. 根据yield()的定义, __am_irq_handle()函数需要将自陷事件打包成编号为EVENT_YIELD的事件.

**恢复上下文**
代码将会一路返回到trap.S的__am_asm_trap()中, 接下来的事情就是恢复程序的上下文. __am_asm_trap()将根据之前保存的上下文内容, 恢复程序的状态, 最后执行"异常返回指令"返回到程序触发异常之前的状态. 对于mips32的syscall和riscv32的ecall, 保存的是自陷指令的PC, 因此软件需要在适当的地方对保存的PC加上4, 使得将来返回到自陷指令的下一条指令.

## 3 Nanos-lite框架

通过nanos-lite/include/common.h中一些与实验进度相关的宏来控制Nanos-lite的功能.

```
nanos-lite
├── include
│   ├── common.h
│   ├── debug.h
│   ├── fs.h
│   ├── memory.h
│   └── proc.h
├── Makefile
├── README.md
├── resources
│   └── logo.txt    # Project-N logo文本
└── src
    ├── device.c    # 设备抽象
    ├── fs.c        # 文件系统
    ├── irq.c       # 中断异常处理
    ├── loader.c    # 加载器
    ├── main.c
    ├── mm.c        # 存储管理
    ├── proc.c      # 进程调度
    ├── ramdisk.c   # ramdisk驱动程序
    ├── resources.S # ramdisk内容和Project-N logo
    └── syscall.c   # 系统调用处理
```

## 4 加载第一个用户程序

在操作系统中, 加载用户程序是由loader(加载器)模块负责的. 我们知道程序中包括代码和数据, 它们都是存储在可执行文件中. 加载的过程就是把可执行文件中的代码和数据放置在正确的内存位置, 然后跳转到程序入口, 程序就开始执行了. 更具体的, 为了实现loader()函数, 我们需要解决以下问题:
- 可执行文件在哪里? <= Navy-apps, 专门用于编译出操作系统的用户程序.
- 代码和数据在可执行文件的哪个位置?
- 代码和数据有多少?
- "正确的内存位置"在哪里?

```
navy-apps
├── apps            # 用户程序
│   ├── am-kernels
│   ├── busybox
│   ├── fceux
│   ├── lua
│   ├── menu
│   ├── nplayer
│   ├── nslider
│   ├── nterm
│   ├── nwm
│   ├── onscripter
│   ├── oslab0
│   └── pal         # 仙剑奇侠传
├── fsimg           # 根文件系统
├── libs            # 运行库
│   ├── libc        # Newlib C库
│   ├── libam
│   ├── libbdf
│   ├── libbmp
│   ├── libfixedptc
│   ├── libminiSDL
│   ├── libndl
│   ├── libos       # 系统调用的用户层封装
│   ├── libSDL_image
│   ├── libSDL_mixer
│   ├── libSDL_ttf
│   └── libvorbis
├── Makefile
├── README.md
├── scripts
└── tests           # 一些测试
```

用户程序的入口位于navy-apps/libs/libos/src/crt0/start.S中的_start()函数, 这里的crt是C RunTime的缩写, 0的含义表示最开始. _start()函数会调用navy-apps/libs/libos/src/crt0/crt0.c中的call_main()函数, 然后调用用户程序的main()函数, 从main()函数返回后会调用exit()结束运行。

## 5 系统调用

作为资源管理者管理着系统中的所有资源, 操作系统需要为用户程序提供相应的服务, 这些服务需要以一种统一的接口来呈现, 用户程序也只能通过这一接口来请求服务，这一接口就是系统调用。

系统调用把整个运行时环境分成两部分, **一部分是操作系统内核区, 另一部分是用户区**。 那些会访问系统资源的功能会放到内核区中实现, 而用户区则保留一些无需使用系统资源的功能(比如strcpy()), 以及用于请求系统资源相关服务的系统调用接口。

在GNU/Linux中, 用户程序通过自陷指令来触发系统调用, Nanos-lite也沿用这个约定。 CTE中的yield()也是通过自陷指令来实现, 虽然它们触发了不同的事件, 但从上下文保存到事件分发, 它们的过程都是非常相似的. 既然我们通过自陷指令来触发系统调用, 那么对用户程序来说, 用来向操作系统描述需求的最方便手段就是**使用通用寄存器**了, 因为执行自陷指令之后, 执行流就会马上切换到事先设置好的入口, 通用寄存器也会作为上下文的一部分被保存起来. 系统调用处理函数只需要从上下文中获取必要的信息, 就能知道用户程序发出的服务请求是什么了。

Nanos-lite收到系统调用事件之后, 就会调出系统调用处理函数do_syscall()进行处理。do_syscall()首先通过宏GPR1从上下文c中获取用户进程之前设置好的系统调用参数, 通过第一个参数 - 系统调用号 - 进行分发。

添加一个系统调用比你想象中要简单, 所有信息都已经准备好了。 我们只需要在分发的过程中**添加相应的系统调用号**, 并编写相应的系统调用处理函数sys_xxx(), 然后调用它即可。回过头来看dummy程序, 它触发了一个SYS_yield系统调用。我们约定, 这个系统调用直接调用CTE的yield()即可, 然后返回0。

处理系统调用的最后一件事就是**设置系统调用的返回值**。对于不同的ISA, 系统调用的返回值存放在不同的寄存器中, 宏GPRx用于实现这一抽象, 所以我们通过GPRx来进行设置系统调用返回值即可。

经过CTE, 执行流会从do_syscall()一路返回到用户程序的_syscall_()函数中。代码最后会从相应寄存器中取出系统调用的返回值, 并返回给_syscall_()的调用者, 告知其系统调用执行的情况(如是否成功等)。

## 6 虚拟文件系统

为了实现**一切皆文件**的思想, 我们之前实现的文件操作就需要进行扩展了: 我们不仅需要对普通文件进行读写, 还需要支持各种"特殊文件"的操作。 至于扩展的方式, 你是再熟悉不过的了, 那就是抽象!

我们对之前实现的文件操作API的语义进行扩展, 让它们可以支持任意文件(包括"特殊文件")的操作:

```
int fs_open(const char *pathname, int flags, int mode);
size_t fs_read(int fd, void *buf, size_t len);
size_t fs_write(int fd, const void *buf, size_t len);
size_t fs_lseek(int fd, size_t offset, int whence);
int fs_close(int fd);
```

这组扩展语义之后的API有一个酷炫的名字, 叫**VFS(虚拟文件系统)**。 既然有虚拟文件系统, 那相应地也应该有"真实文件系统", 这里所谓的真实文件系统, 其实是指具体如何操作某一类文件。 比如在Nanos-lite上, 普通文件通过ramdisk的API进行操作; 在真实的操作系统上, 真实文件系统的种类更是数不胜数: 比如熟悉Windows的你应该知道管理普通文件的NTFS, 目前在GNU/Linux上比较流行的则是EXT4; 至于特殊文件的种类就更多了, 于是相应地有procfs, tmpfs, devfs, sysfs, initramfs... 这些不同的真实文件系统, 它们都分别实现了这些文件的具体操作方式。

所以, VFS其实是对不同种类的真实文件系统的抽象, 它用一组API来描述了这些真实文件系统的抽象行为, 屏蔽了真实文件系统之间的差异, 上层模块(比如系统调用处理函数)不必关心当前操作的文件具体是什么类型, 只要调用这一组API即可完成相应的文件操作。 有了VFS的概念, 要添加一个真实文件系统就非常容易了: 只要把真实文件系统的访问方式包装成VFS的API, 上层模块无需修改任何代码, 就能支持一个新的真实文件系统了。

在Nanos-lite中, 实现VFS的关键就是Finfo结构体中的两个读写函数指针:

```
typedef struct {
  char *name;         // 文件名
  size_t size;        // 文件大小
  size_t disk_offset;  // 文件在ramdisk中的偏移
  ReadFn read;        // 读函数指针
  WriteFn write;      // 写函数指针
} Finfo;
```

其中ReadFn和WriteFn分别是两种函数指针, 它们用于指向真正进行读写的函数, 并返回成功读写的字节数。 有了这两个函数指针, 我们只需要在文件记录表中对不同的文件设置不同的读写函数, 就可以通过f->read()和f->write()的方式来调用具体的读写函数了。