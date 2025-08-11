本文主要整理NJU-OS绪论章节的要点。

## 一、一个操作系统上的最小程序

以下是一个在 **Linux 系统** 上完全脱离标准库、直接通过系统调用实现的极简 C 程序（仅 **200 字节** 左右）：

```c
// minimal.c
// 编译命令：gcc -nostdlib -o minimal minimal.c

void _start() {
    // 系统调用号定义 (x86_64)
    const unsigned long sys_write = 1;
    const unsigned long sys_exit  = 60;

    // 字符串内容
    char msg[] = "Hello, Minimal World!\n";
    unsigned long len = sizeof(msg)-1; // 排除结尾的\0

    // 直接调用 write 系统调用 (输出到 stdout)
    __asm__ volatile (
        "syscall"
        : 
        : "a"(sys_write),   // rax = 1 (sys_write)
          "D"(1),           // rdi = 1 (文件描述符 stdout)
          "S"(msg),         // rsi = 字符串地址
          "d"(len)          // rdx = 字符串长度
    );

    // 调用 exit 系统调用
    __asm__ volatile (
        "syscall"
        : 
        : "a"(sys_exit),   // rax = 60 (sys_exit)
          "D"(0)           // rdi = 0 (退出状态码)
    );
}
```

---

### **关键解析**

#### 1. **程序结构**
- 入口函数 `_start`：替代标准库的 `main`，直接作为程序入口
- 无标准库依赖：完全通过 `syscall` 指令与操作系统交互

#### 2. **系统调用参数传递 (x86_64)**
| 寄存器 | 用途            | 示例值          |
|-------|----------------|----------------|
| `rax` | 系统调用号       | 1=write, 60=exit|
| `rdi` | 第一个参数       | 文件描述符=1    |
| `rsi` | 第二个参数       | 字符串地址      |
| `rdx` | 第三个参数       | 字符串长度      |

#### 3. **编译与运行**
```bash
# 编译 (禁止标准库链接)
gcc -nostdlib -o minimal minimal.c

# 查看文件大小
ls -lh minimal  # 约 16KB (含ELF头部信息)

# 运行
./minimal       # 输出 Hello, Minimal World!
```

#### 4. **进一步优化 (仅演示原理)**
若使用汇编直接编写并精简 ELF 头部，可压缩到 **200 字节** 以下：
```nasm
; tiny.asm
global _start
section .text
_start:
    mov rax, 1      ; sys_write
    mov rdi, 1      ; stdout
    lea rsi, [rel msg]
    mov rdx, len
    syscall

    mov rax, 60     ; sys_exit
    xor rdi, rdi
    syscall

msg: db "Hello, Tiny World!", 10
len: equ $ - msg
```
编译：
```bash
nasm -f elf64 tiny.asm && ld -s -static -o tiny tiny.o
ls -lh tiny  # 约 1KB
```

---

### **对比传统 Hello World**
```c
// hello.c (标准库依赖)
#include <stdio.h>
int main() {
    printf("Hello World!\n");
    return 0;
}
```
编译后约 **8KB**，但实际依赖 glibc 动态库。

---

### **技术总结**
| **特性**          | **极简程序**                | **传统程序**          |
|-------------------|---------------------------|---------------------|
| 文件大小           | ~1KB (汇编优化)            | ~8KB (动态链接)      |
| 依赖库            | 无                        | 依赖 glibc          |
| 启动速度          | 极快 (无动态链接)          | 稍慢                |
| 开发复杂度        | 高 (需直接处理系统调用)     | 低 (使用标准库)      |
| 可移植性          | 架构/系统相关              | 跨平台              |

实际开发中建议使用标准库，但理解这种底层实现有助于深入掌握 **操作系统交互原理**。


## 二、观察程序的运行

### **1. 观察程序运行的常用工具**
#### **1.1 动态追踪工具**
| 工具          | 功能描述                     | 示例命令                      |
|---------------|-----------------------------|-----------------------------|
| **strace**    | 跟踪系统调用和信号           | `strace -e trace=open,read,write ./program` |
| **ltrace**    | 跟踪动态库函数调用           | `ltrace ./program`          |
| **perf**      | 性能分析（含系统调用统计）   | `perf stat -e 'syscalls:*' ./program` |
| **gdb**       | 调试器（单步执行、断点）     | `gdb -q ./program` → `b main` → `r` |

#### **1.2 静态分析工具**
| 工具          | 功能描述                     |
|---------------|-----------------------------|
| **objdump**   | 反汇编查看程序逻辑           | `objdump -d ./program`      |
| **readelf**   | 查看 ELF 文件结构            | `readelf -a ./program`      |

### **2. 系统调用流程详解**
#### **2.1 系统调用触发机制**
以 **x86_64 Linux** 为例：
1. **用户态准备参数**  
   - 程序将系统调用号存入 `rax`，参数依次存入 `rdi`, `rsi`, `rdx`, `r10`, `r8`, `r9`。  
2. **陷入内核态**  
   - 执行 `syscall` 指令，触发 CPU 特权级切换（从 Ring 3 → Ring 0）。  
3. **内核处理**  
   - 根据 `rax` 中的系统调用号，跳转到 `sys_call_table[rax]` 对应的内核函数。  
4. **返回用户态**  
   - 内核将结果存入 `rax`，执行 `sysret` 指令返回用户态。  

#### **2.2 系统调用示例分析**
跟踪 `ls` 命令的系统调用：
```bash
strace -o ls.log ls
```
输出片段解析：
```text
execve("/usr/bin/ls", ["ls"], 0x7ffd2d9d8b20 /* 23 vars */) = 0
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
getdents64(3, /* 6 entries */, 32768) = 144
write(1, "file1.txt  file2.txt\n", 20) = 20
```
- **execve**: 加载可执行文件  
- **openat**: 打开当前目录  
- **getdents64**: 读取目录条目  
- **write**: 输出结果到终端  

---

### **3. 分步骤观察实践**
#### **步骤 1：使用 strace 跟踪系统调用**
```bash
# 跟踪所有系统调用（输出到终端）
strace ./your_program

# 跟踪特定系统调用（如文件操作）
strace -e trace=open,read,write,close ./your_program

# 统计系统调用耗时
strace -c ./your_program
```
输出示例：
```text
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 45.23    0.000123          12        10           openat
 32.11    0.000087           8        11           read
 12.89    0.000035           3        12           write
```

#### **步骤 2：使用 GDB 单步调试**
```bash
gdb -q ./your_program
(gdb) break main          # 在 main 函数设断点
(gdb) run                 # 启动程序
(gdb) stepi               # 单步执行汇编指令
(gdb) info registers      # 查看寄存器状态
(gdb) x/10i $pc           # 查看当前指令附近代码
```

#### **步骤 3：查看实时进程状态**
```bash
# 查看进程资源使用
top -p $(pgrep your_program)

# 监控进程系统调用（需 root）
perf trace -p $(pgrep your_program)
```

## 三、探索文件的内容

要探索 `a.out` 文件的内容和功能，可以按以下步骤进行：

---

### **1. 确认文件类型和基本信息**
#### **1.1 查看文件类型和架构**
```bash
file a.out
```
- **输出示例**：
  ```text
  a.out: ELF 64-bit LSB executable, x86-64, dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, for GNU/Linux 3.2.0, not stripped
  ```
  - **关键信息**：
    - **ELF 格式**：Linux 可执行文件。
    - **64-bit**：64 位架构。
    - **not stripped**：保留调试符号（可查看函数名）。

#### **1.2 检查依赖的动态库**
```bash
ldd a.out
```
- **输出示例**：
  ```text
    linux-vdso.so.1 (0x00007ffd55df0000)
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f8a3a200000)
    /lib64/ld-linux-x86-64.so.2 (0x00007f8a3a600000)
  ```
  - **用途**：确认依赖的共享库是否存在。

---

### **2. 静态分析：查看内部结构**
#### **2.1 查看符号表（函数/变量名）**
```bash
nm a.out
```
- **输出示例**：
  ```text
  0000000000401116 T main
  0000000000404010 D __data_start
  ```
  - **关键字段**：
    - **T**：代码段中的符号（如函数）。
    - **D**：已初始化的全局变量。

#### **2.2 提取可打印字符串**
```bash
strings a.out
```
- **输出示例**：
  ```text
  /lib64/ld-linux-x86-64.so.2
  Hello, World!
  ;*3$"
  ```
  - **用途**：发现硬编码的路径、密钥、调试信息等。

#### **2.3 查看 ELF 文件头信息**
```bash
readelf -h a.out
```
- **输出字段**：
  - **Entry point address**：程序入口地址（如 `0x401050`）。
  - **Section headers**：节头表位置。

---

### **3. 动态分析：运行时行为**
#### **3.1 跟踪系统调用**
```bash
strace ./a.out
```
- **输出示例**：
  ```text
  execve("./a.out", ["./a.out"], 0x7ffe7b1f8a20 /* 23 vars */) = 0
  write(1, "Hello, World!\n", 13)          = 13
  exit_group(0)                           = ?
  ```
  - **用途**：监控文件操作、网络连接等行为。

#### **3.2 调试程序逻辑**
```bash
gdb ./a.out
```
- **常用命令**：
  ```text
  (gdb) break main     # 在 main 函数设置断点
  (gdb) run            # 启动程序
  (gdb) disassemble    # 反汇编当前函数
  (gdb) info registers # 查看寄存器状态
  ```

---

### **4. 逆向工程：反汇编与反编译**
#### **4.1 反汇编代码段**
```bash
objdump -d a.out
```
- **输出示例**：
  ```asm
  0000000000401116 <main>:
    401116:       55                      push   %rbp
    401117:       48 89 e5                mov    %rsp,%rbp
    40111a:       48 83 ec 10             sub    $0x10,%rsp
    40111e:       c7 45 fc 00 00 00 00    movl   $0x0,-0x4(%rbp)
  ```
  - **用途**：分析汇编代码逻辑。

#### **4.2 使用反编译工具（如 Ghidra）**
- **步骤**：
  1. 安装 Ghidra。
  2. 导入 `a.out`，自动分析伪代码。
  3. 查看 `main` 函数的反编译结果：
     ```c
     int main() {
         printf("Hello, World!\n");
         return 0;
     }
     ```

---

### **5. 高级技巧**
#### **5.1 检查哈希值（确认文件唯一性）**
```bash
sha256sum a.out
```
- **输出示例**：
  ```text
  9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08  a.out
  ```

#### **5.2 沙箱中运行（安全测试）**
```bash
firejail --net=none ./a.out  # 禁用网络
```
- **用途**：防止恶意代码破坏系统。

---

### **总结**
| **工具**       | **用途**                          | **关键命令**                     |
|----------------|----------------------------------|----------------------------------|
| `file`         | 确认文件类型和架构                | `file a.out`                     |
| `ldd`          | 检查动态库依赖                    | `ldd a.out`                      |
| `nm`           | 查看符号表                        | `nm a.out`                       |
| `strings`      | 提取可打印字符串                  | `strings a.out`                  |
| `strace`       | 跟踪系统调用                      | `strace ./a.out`                 |
| `gdb`          | 动态调试                          | `gdb ./a.out`                    |
| `objdump`      | 反汇编代码                        | `objdump -d a.out`               |

通过组合这些工具，可以全面了解 `a.out` 的内部结构、依赖关系和运行时行为。如果是未知来源文件，建议在隔离环境中分析！


## 四、最小化的合法 MBR 文件

```
(printf "\xeb\xfe"; cat /dev/zero | head -c 508; printf "\x55\xaa") > a.img
```

该命令生成一个名为 `a.img` 的 **512 字节磁盘映像文件**，其内容符合 **MBR（主引导记录）** 的格式要求，但实际功能仅为一个 **无限循环的引导扇区**。以下是分步解析：

---

### **命令分解**
```bash
(
  printf "\xeb\xfe";          # 第1部分：写入 2 字节的机器码（无限循环）
  cat /dev/zero | head -c 508; # 第2部分：填充 508 字节的零（占位）
  printf "\x55\xaa"           # 第3部分：写入 2 字节的 MBR 结束标志
) > a.img                     # 输出到文件 a.img
```

---

### **文件结构（总 512 字节）**
| 偏移量 | 长度（字节） | 内容              | 作用                     |
|-------|------------|------------------|-------------------------|
| 0     | 2          | `EB FE`          | 引导代码（无限循环）       |
| 2     | 508        | `00 00 ... 00`   | 填充空字节（无意义数据）    |
| 510   | 2          | `55 AA`          | MBR 有效签名（必需）      |

---

### **详细说明**
1. **引导代码（EB FE）**  
   - **机器码**：`EB FE` 对应汇编指令 `jmp $`（无限循环跳转到自身地址）。  
   - **作用**：当 BIOS 加载此扇区到内存 `0x7C00` 并执行时，CPU 会卡在此循环，不进行后续操作。  
   - **用途**：常用于测试引导流程或占位。

2. **填充空字节（508 字节）**  
   - 传统 MBR 中，前 **446 字节** 为引导代码，随后 **64 字节** 为分区表，但此示例未严格遵循，直接填充零。  
   - 空字节无实际功能，仅用于凑满 512 字节的扇区大小。

3. **MBR 结束标志（55 AA）**  
   - 所有 MBR 的 **最后 2 字节** 必须为 `55 AA`，否则 BIOS 会认为该磁盘不可引导。

---

### **使用场景**
1. **引导扇区测试**  
   - 在虚拟机或真实硬件中测试 BIOS 是否能正确识别并加载此 MBR。  
   - 例如，用 QEMU 运行：  
     ```bash
     qemu-system-x86_64 -drive file=a.img,format=raw
     ```
   - 虚拟机将启动并卡在无限循环（无输出，需调试工具观察）。

2. **构建自定义引导程序**  
   - 可替换 `EB FE` 和填充区域为实际引导代码（如显示文本、加载内核）。

---

### **验证文件**
```bash
# 查看文件大小（应为 512 字节）
ls -l a.img

# 检查结尾是否为 55 AA（最后两字节）
hexdump -C a.img | tail -n 1
# 输出示例：00000200  eb fe 00 00 ... 00 55 aa
```

---

### **注意事项**
- **安全性**：此文件无破坏性，但若写入真实磁盘（如 `/dev/sda`）会覆盖原有 MBR，导致系统无法启动。
- **扩展性**：实际引导程序需在 `EB FE` 位置编写有效代码（如调用 BIOS 中断显示字符）。

---

## 五、Firmware

### **Firmware（固件）的功能**

**Firmware（固件）** 是直接嵌入硬件设备的底层软件，负责 **初始化硬件** 和 **提供硬件与操作系统之间的桥梁**。它的核心功能包括：

1. **硬件初始化（Power-On Self-Test, POST）**  
   - 计算机通电后，固件（如 BIOS/UEFI）首先执行自检（POST），检测 CPU、内存、存储设备等硬件是否正常。
   - 若检测到故障，会通过蜂鸣声或屏幕提示错误代码。

2. **提供硬件操作接口**  
   - 固件提供标准化的接口（如 BIOS 中断调用、UEFI 协议），供操作系统和应用程序访问硬件（如磁盘读写、键盘输入）。

3. **启动管理（Boot Manager）**  
   - 负责从存储设备（硬盘、UEFI 分区、USB）加载操作系统的引导程序（如 GRUB、Windows Boot Manager）。
   - 支持多系统启动（通过选择不同启动项）。

4. **硬件配置与安全功能**  
   - 提供硬件参数设置界面（如 BIOS Setup/UEFI 界面），配置 CPU 频率、启动顺序、安全启动（Secure Boot）等。
   - 支持固件级安全功能（如 TPM 芯片管理）。

---

### **Firmware 如何加载操作系统？**

操作系统加载是一个链式过程，以 **UEFI**（现代主流固件）为例，流程如下：

#### **1. 固件初始化阶段**
   - **Step 1**：通电后，UEFI 固件初始化 CPU、内存、主板芯片组等硬件。
   - **Step 2**：扫描所有存储设备（如 SSD、HDD），寻找 **EFI 系统分区**（FAT32 格式，通常包含引导文件）。

#### **2. 加载引导程序（Boot Loader）**
   - **Step 3**：在 EFI 分区中找到操作系统的引导程序（如 `\EFI\Microsoft\Boot\bootmgfw.efi`（Windows）或 `\EFI\ubuntu\grubx64.efi`（Linux））。
   - **Step 4**：固件将控制权交给引导程序。

#### **3. 引导程序加载操作系统内核**
   - **Step 5**：引导程序读取配置文件（如 GRUB 的 `grub.cfg`），显示启动菜单（选择操作系统或内核版本）。
   - **Step 6**：根据用户选择，加载操作系统内核（如 Linux 的 `vmlinuz`）和初始化内存盘（`initramfs`）。
   - **Step 7**：引导程序将控制权交给操作系统内核。

#### **4. 操作系统启动**
   - **Step 8**：内核初始化系统资源（驱动、文件系统、网络），启动用户空间的第一个进程（如 `systemd`（Linux）或 `wininit.exe`（Windows））。
   - **Step 9**：操作系统完成启动，进入登录界面。

---

### **传统 BIOS vs 现代 UEFI**

| **功能**           | **传统 BIOS**                      | **现代 UEFI**                     |
|--------------------|-----------------------------------|----------------------------------|
| **启动方式**        | 基于 MBR 分区（最大支持 2TB）      | 基于 GPT 分区（支持 >2TB 磁盘）    |
| **启动速度**        | 较慢（逐级加载）                   | 更快（直接加载 EFI 文件）          |
| **安全功能**        | 无 Secure Boot                    | 支持 Secure Boot（防恶意软件）     |
| **图形界面**        | 文本模式                          | 支持图形化界面和鼠标操作           |
| **多系统支持**      | 依赖第三方引导工具（如 GRUB）      | 原生支持多系统启动                 |


### **总结**
- **Firmware** 是硬件和操作系统之间的“翻译官”，负责初始化硬件并启动操作系统。
- **加载操作系统的关键**：固件找到引导程序 → 引导程序加载内核 → 内核接管系统。
- 现代 UEFI 提供了更快的启动速度和更强的安全性，逐步替代传统 BIOS。


## 六、GRUB 和 Linux 系统加载流程

以下是 **GRUB 和 Linux 系统加载流程**的详细步骤说明，涵盖从开机到用户空间初始化的完整过程：

---

### **1. 固件阶段（BIOS/UEFI）**
- **动作**：  
  计算机通电后，固件（BIOS 或 UEFI）执行硬件自检（POST），初始化关键硬件（CPU、内存、存储控制器等）。  
- **关键任务**：  
  - **BIOS**：寻找磁盘的 **MBR（主引导记录）**（第一个扇区，512字节）。  
  - **UEFI**：直接读取 **EFI 系统分区**（FAT32格式）中的 `.efi` 引导文件。

---

### **2. GRUB 引导加载阶段**
#### **2.1 GRUB 第一阶段（仅 BIOS 模式需要）**
- **MBR 中的引导代码**（`boot.img`，446字节）：  
  由 BIOS 加载到内存 `0x7C00` 并执行，负责加载 GRUB 的 **核心镜像**（`core.img`）。

- **core.img**（嵌入在 MBR 后的扇区）：  
  包含基础驱动（如访问文件系统），加载 `/boot/grub/grub.cfg` 配置文件。

#### **2.2 GRUB 第二阶段（图形化菜单）**
- **加载流程**：  
  1. 读取 `/boot/grub/grub.cfg`，显示启动菜单（选择内核版本或操作系统）。  
  2. 根据用户选择，加载 **Linux 内核**（`vmlinuz-xxx`）和 **initramfs**（初始内存文件系统）。  
  3. 将控制权交给 Linux 内核。

- **关键文件**：  
  ```bash
  /boot/grub/grub.cfg          # GRUB 配置文件（通常由 grub-mkconfig 生成）
  /boot/vmlinuz-5.15.0-78      # Linux 内核文件
  /boot/initramfs-5.15.0-78.img # 初始内存文件系统
  ```

---

### **3. Linux 内核初始化**
#### **3.1 内核解压与早期初始化**
- **动作**：  
  内核解压自身并初始化基本硬件（CPU、内存管理、中断控制器等）。  
- **关键任务**：  
  解析内核命令行参数（如 `root=/dev/sda1` 指定根文件系统）。

#### **3.2 挂载 initramfs**
- **作用**：  
  `initramfs` 是一个临时根文件系统，包含启动必需的驱动和工具（如磁盘驱动、LVM 支持）。  
- **流程**：  
  1. 内核解压 `initramfs` 到内存中的 `tmpfs`。  
  2. 执行 `/init` 脚本（位于 `initramfs` 中），加载实际根文件系统所需的模块（如 `ext4`、`nvme`）。  

#### **3.3 切换根文件系统**
- **动作**：  
  `initramfs` 的 `/init` 脚本挂载真正的根文件系统（如 `/dev/sda1`）到 `/sysroot`，并执行 `switch_root` 切换根目录。  
- **示例命令**：  
  ```bash
  mount /dev/sda1 /sysroot
  exec switch_root /sysroot /sbin/init
  ```

---

### **4. 用户空间初始化（systemd）**
#### **4.1 systemd 进程启动**
- **PID 1**：  
  `systemd` 作为第一个用户空间进程（PID 1）启动，接管后续初始化。  
- **关键任务**：  
  1. 解析 `/etc/fstab`，挂载所有文件系统。  
  2. 启动系统服务（如网络、日志、SSH）。  

#### **4.2 Target 单元与运行级别**
- **默认目标**：  
  通过 `systemctl get-default` 获取（如 `graphical.target` 或 `multi-user.target`）。  
- **流程**：  
  - 按依赖顺序启动目标（target）对应的服务单元（unit）。  
  - 最终进入登录界面（如 GDM 或 getty）。

---

### **流程图解**
```
固件 (BIOS/UEFI)
   ↓
GRUB 阶段1（加载 core.img）
   ↓
GRUB 阶段2（显示菜单，加载内核和 initramfs）
   ↓
Linux 内核（初始化硬件，加载 initramfs）
   ↓
initramfs（挂载根文件系统，切换根目录）
   ↓
systemd（挂载文件系统，启动服务）
   ↓
用户空间（登录界面或 Shell）
```

---

### **关键配置文件与工具**
| 组件      | 配置文件/工具                  | 作用                          |
|-----------|-------------------------------|-------------------------------|
| **GRUB**  | `/boot/grub/grub.cfg`         | 定义启动菜单和内核参数          |
|           | `grub-mkconfig`               | 生成 `grub.cfg`               |
| **内核**  | `/proc/cmdline`               | 查看启动时的内核参数            |
| **initramfs** | `dracut` 或 `mkinitcpio`    | 生成 `initramfs` 镜像          |
| **systemd** | `/etc/systemd/system`       | 自定义服务单元文件              |

---

### **调试启动问题**
- **查看内核日志**：  
  ```bash
  journalctl -k          # 内核日志
  journalctl -b          # 本次启动日志
  ```
- **修改 GRUB 参数**：  
  在 GRUB 菜单按 `e` 进入编辑模式，临时修改内核参数（如 `rd.break` 进入 initramfs 的 Shell）。

---

### **总结**
- **GRUB** 是引导加载程序，负责加载内核和 `initramfs`。  
- **Linux 内核** 初始化硬件并挂载根文件系统，依赖 `initramfs` 提供必要驱动。  
- **systemd** 管理用户空间服务，完成系统启动。  
- 整个流程通过多阶段协作，将硬件控制权逐步从固件转移到操作系统。


## 七、RISC-V 系统启动流程

以下是 **RISC-V 系统启动流程**的详细说明，涵盖复位机制、固件作用和操作系统加载步骤：

---

### **1. 复位（Reset）机制**
RISC-V 系统的复位流程由硬件设计决定，通常遵循以下步骤：

#### **1.1 复位信号触发**
- **复位源**：  
  系统复位可由多种条件触发，如：  
  - 上电复位（Power-On Reset, POR）  
  - 外部复位引脚（Reset Button）  
  - 看门狗超时（Watchdog Timeout）  
  - 软件触发（写特定控制寄存器）  

#### **1.2 复位后的初始状态**
- **核心行为**：  
  - 所有 CPU 核心（HART）的 **PC（程序计数器）** 跳转到 **复位向量地址**（通常为 `0x8000_0000`，具体由 SoC 设计决定）。  
  - 特权模式设置为 **机器模式（Machine Mode, M-Mode）**（最高权限）。  
  - 关键寄存器（如 `mstatus`、`mtvec`）被初始化为默认值。  

- **内存与设备**：  
  - 片上 ROM 或 Flash 中的 **引导代码（BootROM）** 被映射到复位向量地址，用于启动第一阶段的固件。

---

### **2. 执行的固件**
RISC-V 系统的启动通常依赖多级固件协作，常见组合如下：

#### **2.1 BootROM（一级固件）**
- **位置**：固化在芯片的只读存储器（ROM）或 Flash 中。  
- **功能**：  
  - 初始化最基础的硬件（如时钟、内存控制器）。  
  - 加载二级固件（如 OpenSBI 或 U-Boot）到内存。  
  - 示例：SiFive 芯片的 BootROM 会从 SPI Flash 加载下一阶段代码。

#### **2.2 OpenSBI（二级固件，可选）**
- **作用**：实现 **RISC-V 监督模式二进制接口（SBI）**，为操作系统提供底层服务（如定时器、IPI 中断）。  
- **启动流程**：  
  1. BootROM 将 OpenSBI 代码加载到内存并跳转执行。  
  2. OpenSBI 初始化中断控制器、串口等外设。  
  3. 通过 `sbi_hart_start` 启动其他 CPU 核心（多核系统）。  
  4. 最终跳转到操作系统内核或下一级引导程序（如 U-Boot）。

#### **2.3 U-Boot（三级固件，可选）**
- **角色**：作为 **引导加载程序（Bootloader）**，支持复杂启动逻辑（如网络启动、多重引导）。  
- **功能**：  
  - 从存储设备（SD 卡、eMMC、NVMe）或网络（TFTP）加载操作系统内核和设备树（DTB）。  
  - 解析环境变量（`bootcmd`、`bootargs`）配置启动参数。  
  - 示例命令：  
    ```bash
    # 从 SD 卡加载 Linux 内核
    load mmc 0:1 ${kernel_addr_r} /boot/Image
    booti ${kernel_addr_r} - ${fdt_addr_r}
    ```

---

### **3. 操作系统加载流程**
以 **Linux 内核** 为例，完整启动流程如下：

#### **3.1 内核加载**
- **固件传递控制权**：  
  OpenSBI 或 U-Boot 将内核镜像（如 `Image` 或 `vmlinux`）和设备树（`.dtb`）加载到指定内存地址，并跳转到内核入口点。  
- **内核要求**：  
  - RISC-V Linux 内核需支持 **PBL（Payload Boot Loader）** 格式或 Flat Image（直接可执行）。  
  - 设备树需描述硬件拓扑（如 UART 地址、内存布局）。

#### **3.2 内核初始化**
1. **早期初始化**：  
   - 设置虚拟内存（页表）、检测 CPU 核心数量。  
   - 解析设备树，初始化平台设备（如 PLIC 中断控制器、CLINT 定时器）。  
2. **用户空间启动**：  
   - 挂载根文件系统（通过 `root=` 参数指定，如 `root=/dev/mmcblk0p2`）。  
   - 启动第一个用户进程（如 `init` 或 `systemd`）。

#### **3.3 多核启动（SMP）**
- **主核（Boot HART）**：执行完整的内核初始化流程。  
- **从核（Secondary HARTs）**：  
  主核通过 SBI 的 `sbi_hart_start` 唤醒从核，从核直接跳转到内核指定的入口点，执行空闲循环或任务调度。

---

### **4. 典型启动代码示例**
#### **BootROM 伪代码**
```asm
reset_vector:
    li sp, 0x80010000  # 设置栈指针
    call init_clocks    # 初始化时钟
    call init_ddr       # 初始化内存
    load_fw_from_spi()  # 从 SPI Flash 加载 OpenSBI 到 0x80000000
    jr 0x80000000       # 跳转到 OpenSBI
```

#### **OpenSBI 启动 Linux**
```bash
# 启动单核 Linux
fw_dynamic info: booting hart 0 with entry point 0x80200000
```

---

### **5. 总结**
- **复位**：硬件强制跳转到复位向量，由 BootROM 执行最底层初始化。  
- **固件链**：BootROM → OpenSBI（可选）→ U-Boot（可选）→ Linux Kernel。  
- **操作系统加载**：依赖设备树描述硬件，通过固件传递参数并跳转执行。  
- **关键点**：确保复位地址、设备树和内核镜像的兼容性。  

实际实现需结合具体芯片手册（如 SiFive FU740、Allwinner D1）调整启动流程。

## 八、OpenSBI 原理详解

OpenSBI 是 RISC-V 架构的开源参考实现，属于 **监督模式二进制接口（SBI）** 的固件，专为 RISC-V 系统设计。它在硬件与操作系统之间扮演桥梁角色，负责 **底层硬件初始化** 和 **运行时服务提供**。以下是其核心原理的分层解析：

---

### **1. OpenSBI 的定位与架构**
#### **1.1 RISC-V 特权级模型**
- **M-Mode（机器模式）**：最高特权级，OpenSBI 运行于此模式，直接管理硬件。
- **S-Mode（监督模式）**：操作系统内核（如 Linux）运行层级。
- **U-Mode（用户模式）**：应用程序运行环境。

OpenSBI 作为 M-Mode 固件，**隔离操作系统与硬件**，提供标准服务接口。

#### **1.2 核心组件**
- **平台初始化代码**：处理特定硬件（如 UART、定时器）的初始化。
- **SBI 实现层**：响应来自 S-Mode 的 `ecall` 请求，执行 M-Mode 操作。
- **运行时服务**：如核间中断（IPI）、定时器管理、系统复位。

---

### **2. 启动流程解析**
以典型 RISC-V SoC（如 SiFive FU740）为例：

2.1 **复位阶段**
   - 处理器从 **复位向量地址**（如 `0x8000_0000`）执行 BootROM 代码。
   - BootROM 初始化基础硬件（DDR 内存、SPI 控制器），加载 OpenSBI 到内存。

2.2 **OpenSBI 初始化**
   ```c
   // 示例：启动主核（HART 0）
   sbi_init() {
       init_platform();      // 初始化串口、中断控制器
       sbi_hsm_init();       // 硬件状态管理
       sbi_scratch_init();   // 设置各核的暂存区
       sbi_trap_init();      // 配置异常处理
   }
   ```
   - 配置 **物理内存保护（PMP）**，隔离内核空间。
   - 初始化 **CLINT（核局部中断器）** 和 **PLIC（平台级中断控制器）**。

2.3 **启动操作系统**
   - 将设备树（DTB）和内核镜像地址传递给下一阶段（如 U-Boot 或 Linux）。
   - 通过 `sbi_hart_start()` 启动从核，执行 `_start` 函数跳转至内核入口。

---

### **3. SBI 接口工作机制**
#### **3.1 SBI 调用流程**
当操作系统（S-Mode）需执行特权操作时：
1. **触发 `ecall` 指令**  
   ```asm
   // 示例：获取 SBI 版本
   li a7, 0x10   // SBI_GET_SBI_SPEC_VERSION
   ecall         // 陷入 M-Mode
   mv version, a0
   ```
2. **OpenSBI 陷阱处理**  
   - 读取 `a7` 寄存器中的 **SBI 扩展 ID**（如 0x10 表示基础扩展）。
   - 根据 ID 调用对应的处理函数，结果通过 `a0` 返回。

#### **3.2 主要 SBI 扩展**
| **扩展名**         | **功能**                          | **示例调用**                     |
|--------------------|----------------------------------|----------------------------------|
| **Base**           | 查询 SBI 版本与实现信息           | `sbi_get_spec_version()`         |
| **Timer**          | 管理定时器中断                    | `sbi_set_timer()`                |
| **IPI**            | 核间中断控制                      | `sbi_send_ipi()`                 |
| **RFENCE**         | 内存屏障与缓存维护                | `sbi_remote_fence_i()`           |
| **HSM**            | 多核启动/停止管理                 | `sbi_hart_start()`               |

---

### **4. 关键代码结构**
```text
opensbi/
├── lib/             # 通用库（链表、字符串处理）
├── platform/        # 平台相关代码（如 qemu/, fu740/）
├── include/sbi/     # SBI 接口定义
├── fw_base.S        # 启动汇编入口
└── sbi_init.c       # 初始化主函数
```

- **平台移植**：需实现 `struct sbi_platform_operations`，定义硬件操作函数集。
- **设备树解析**：通过 `fdt_parse_` 系列函数获取内存布局、设备信息。

---

### **5. 与 U-Boot/Linux 的协作**
1. **直接引导 Linux**  
   ```
   BootROM → OpenSBI → Linux Kernel（带嵌入式 Initramfs）
   ```
   - OpenSBI 直接传递设备树给内核，适用于简单系统。

2. **通过 U-Boot 引导**  
   ```
   BootROM → OpenSBI → U-Boot → Linux Kernel
   ```
   - U-Boot 提供更复杂的引导菜单、文件系统支持。

---

### **6. 调试与定制**
- **日志查看**：通过串口输出，需在平台代码中启用 `CONFIG_PRINT`。
- **自定义 SBI 扩展**：
  ```c
  // 添加新扩展
  struct sbi_extension my_ext = {
      .extid_start = 0x09000000,
      .handle = my_ext_handler,
  };
  sbi_extension_register(&my_ext);
  ```
- **QEMU 测试**：
  ```bash
  qemu-system-riscv64 -M virt -kernel opensbi/build/platform/qemu/virt/firmware/fw_jump.bin -device loader,file=linux/arch/riscv/boot/Image,addr=0x80200000
  ```

---

### **7. 安全机制**
- **PMP 配置**：限制 S-Mode 内存访问范围。
- **Secure Boot**（依赖硬件）：验证下一阶段镜像签名。
- **隔离关键资源**：如定时器寄存器仅允许 M-Mode 访问。

---

**总结**  
OpenSBI 是 RISC-V 生态的基石，通过标准化 SBI 接口统一了硬件操作，使操作系统无需关注底层差异。其模块化设计支持灵活适配不同硬件平台，为构建安全可靠的 RISC-V 系统提供了关键支持。

## 九、极简操作系统

以下是一个使用 Python 实现的极简操作系统概念模型，模拟了 **进程调度**、**内存管理** 和 **文件系统** 的核心功能。该模型不涉及真实硬件交互，但能帮助理解操作系统基本原理：

---

### **1. 极简操作系统模型代码**
```python
import time
from collections import deque

# ==============================
# 核心组件定义
# ==============================

class Process:
    """模拟进程"""
    _pid_counter = 0

    def __init__(self, name, memory=128):
        self.pid = Process._pid_counter
        Process._pid_counter += 1
        self.name = name
        self.memory = memory  # 内存需求 (KB)
        self.state = "READY"  # 状态: READY/RUNNING/WAITING/TERMINATED

class MemoryManager:
    """模拟内存管理"""
    def __init__(self, total_memory=1024):  # 总内存 1024KB
        self.total = total_memory
        self.free = total_memory
        self.allocations = {}  # {pid: allocated_memory}

    def allocate(self, process):
        if self.free >= process.memory:
            self.allocations[process.pid] = process.memory
            self.free -= process.memory
            return True
        return False

    def deallocate(self, pid):
        if pid in self.allocations:
            self.free += self.allocations[pid]
            del self.allocations[pid]

class Scheduler:
    """模拟进程调度器 (轮转算法)"""
    def __init__(self):
        self.ready_queue = deque()

    def add_process(self, process):
        self.ready_queue.append(process)

    def schedule(self):
        if self.ready_queue:
            process = self.ready_queue.popleft()
            process.state = "RUNNING"
            print(f"Executing PID={process.pid} ({process.name})")
            time.sleep(1)  # 模拟时间片
            process.state = "TERMINATED"
            return process
        return None

class SimpleFS:
    """模拟文件系统"""
    def __init__(self):
        self.files = {}  # {filename: content}

    def create_file(self, name, content=""):
        self.files[name] = content

    def read_file(self, name):
        return self.files.get(name, "File not found")

# ==============================
# 操作系统运行逻辑
# ==============================
class MiniOS:
    def __init__(self):
        self.memory = MemoryManager()
        self.scheduler = Scheduler()
        self.fs = SimpleFS()

    def run(self):
        print("MiniOS 启动")
        # 初始化文件系统
        self.fs.create_file("welcome.txt", "Welcome to MiniOS!")
        # 用户进程示例
        processes = [
            Process("文本编辑器", 256),
            Process("音乐播放器", 512),
            Process("终端", 128)
        ]
        # 分配内存并加入调度队列
        for p in processes:
            if self.memory.allocate(p):
                self.scheduler.add_process(p)
                print(f"进程 {p.name} (PID={p.pid}) 已加载")
            else:
                print(f"内存不足，无法加载 {p.name}")
        # 开始调度
        while True:
            terminated = self.scheduler.schedule()
            if terminated:
                self.memory.deallocate(terminated.pid)
                print(f"进程 PID={terminated.pid} 已退出")
            if not self.scheduler.ready_queue:
                break
        print("所有进程执行完毕")

# ==============================
# 运行操作系统
# ==============================
if __name__ == "__main__":
    os = MiniOS()
    os.run()
```

---

### **2. 代码解析**

#### **核心组件**
| **组件**          | **功能**                          | **实现细节**                              |
|-------------------|-----------------------------------|------------------------------------------|
| `Process`         | 模拟进程                          | 包含 PID、名称、内存需求和状态            |
| `MemoryManager`   | 内存分配器                        | 跟踪空闲内存，处理分配/释放请求            |
| `Scheduler`       | 进程调度器（轮转算法）            | 管理就绪队列，按时间片调度进程             |
| `SimpleFS`        | 简易文件系统                      | 支持文件创建和读取                        |
| `MiniOS`          | 操作系统整合层                    | 协调各组件运行                            |

#### **执行流程**
1. **初始化操作系统**  
   - 创建内存管理器（1024KB）  
   - 初始化文件系统（创建欢迎文件）  
2. **加载进程**  
   - 创建三个示例进程（文本编辑器、音乐播放器、终端）  
   - 检查内存是否足够并加入调度队列  
3. **调度执行**  
   - 轮转执行就绪队列中的进程  
   - 每个进程模拟运行1秒后终止  
   - 释放进程占用的内存  

---

### **3. 运行示例输出**
```text
MiniOS 启动
进程 文本编辑器 (PID=0) 已加载
进程 音乐播放器 (PID=1) 已加载
进程 终端 (PID=2) 已加载
Executing PID=0 (文本编辑器)
进程 PID=0 已退出
Executing PID=1 (音乐播放器)
进程 PID=1 已退出
Executing PID=2 (终端)
进程 PID=2 已退出
所有进程执行完毕
```

---

### **4. 扩展建议**
要实现更接近真实操作系统的特性，可以添加以下功能：

#### **进程管理增强**
- **优先级调度**：在 `Process` 类中添加 `priority` 属性，实现优先级队列
- **进程状态切换**：增加 `WAITING` 状态和 I/O 模拟
```python
class Process:
    def __init__(self, name, memory=128, io_ops=0):
        self.io_ops = io_ops  # 模拟 I/O 操作次数
```

#### **内存管理升级**
- **分页机制**：将内存划分为固定大小的页（如4KB）
```python
class Page:
    def __init__(self, page_id):
        self.id = page_id
        self.owner = None  # 所属进程 PID
```

#### **文件系统完善**
- **目录结构**：使用树形结构代替扁平字典
- **权限管理**：添加读写权限位
```python
class File:
    def __init__(self, name, content="", permissions=0o644):
        self.name = name
        self.content = content
        self.permissions = permissions
```

---

### **5. 学习价值**
此模型虽简化，但完整呈现了操作系统的 **核心抽象概念**：
- **资源虚拟化**：通过软件模拟硬件资源（CPU时间片、内存分配）
- **并发管理**：通过调度器实现进程间的伪并行
- **持久化存储**：通过文件系统管理数据

可通过逐步扩展此模型，深入理解 **进程间通信（IPC）**、**虚拟内存** 和 **设备驱动** 等高级主题。