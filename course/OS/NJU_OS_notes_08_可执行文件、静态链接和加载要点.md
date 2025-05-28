本文主要整理可执行文件、静态链接和加载的要点。

## 一、可执行文件

在 C 语言中，可执行文件的生成需要包含以下几个核心部分。这些内容由编译器和链接器共同协作生成，最终形成一个可在操作系统上独立运行的程序。
ELF总结参考[ICS-PA2024 Lecture8-9 note](./course/Computer_System/notes/PA_ICS2024_notes_04_IO设备_链接与加载.md)。

---

### **1. 程序头（Program Headers）**
- **作用**：描述可执行文件的内存布局和加载方式，供操作系统加载器使用。
- **包含内容**：
  - **入口地址（Entry Point）**：程序开始执行的地址（通常是 `_start` 或 `main` 函数）。
  - **段（Segments）信息**：如代码段（`.text`）、数据段（`.data`、`.bss`）、堆栈段等。

```bash
# 查看程序头（ELF 格式）
readelf -l /path/to/executable
```

---

### **2. 代码段（.text Section）**
- **作用**：存储编译后的机器指令（即程序的代码逻辑）。
- **来源**：由编译器将 C 代码编译为汇编，再汇编为机器码生成。
- **示例**：
  ```c
  // C 代码
  int main() { return 0; }
  ```
  对应的 `.text` 段会包含 `main` 函数的机器指令。

---

### **3. 数据段（.data 和 .bss Sections）**
- **`.data` 段**：存储已初始化的全局变量和静态变量。
  ```c
  int global_var = 42;  // 存储在 .data 段
  ```
- **`.bss` 段**：存储未初始化的全局变量和静态变量（程序加载时清零）。
  ```c
  static int uninit_var; // 存储在 .bss 段
  ```

---

### **4. 符号表（Symbol Table）**
- **作用**：记录函数名、变量名等符号的地址信息，用于调试和动态链接。
- **查看方式**：
  ```bash
  nm /path/to/executable   # 查看符号表
  readelf -s /path/to/executable
  ```

---

### **5. 动态链接信息（仅动态链接程序）**
- **作用**：记录程序依赖的动态库（如 `libc.so`）及其函数符号。
- **包含内容**：
  - **动态段（.dynamic）**：列出依赖的共享库。
  - **重定位表（.rela.plt, .rela.dyn）**：记录运行时需要解析的符号地址。
- **查看方式**：
  ```bash
  ldd /path/to/executable   # 查看动态库依赖
  readelf -d /path/to/executable
  ```

---

### **6. 调试信息（可选）**
- **作用**：包含源代码行号、变量类型等调试数据（需编译时添加 `-g` 选项）。
- **查看方式**：
  ```bash
  objdump -S /path/to/executable  # 查看带源码的反汇编
  ```

---

### **7. 启动代码（Startup Code）**
- **作用**：初始化程序运行环境（如设置堆栈、处理命令行参数），并调用 `main` 函数。
- **示例**：
  - 在 Linux 中，由 `crt1.o` 或 `Scrt1.o` 提供 `_start` 入口。
  - 启动代码负责调用 `main` 并处理其返回值（通过 `exit` 系统调用）。

---

### **8. 运行时依赖（Runtime Dependencies）**
- **静态链接程序**：将 `libc` 等库代码直接嵌入可执行文件。
- **动态链接程序**：依赖系统中的共享库（如 `/lib/x86_64-linux-gnu/libc.so.6`）。

---

### **可执行文件生成流程**
```text
C 源代码 (.c) 
  → 预处理器（展开宏、头文件） → 预处理后的代码 (.i) 
  → 编译器（生成汇编代码） → 汇编文件 (.s) 
  → 汇编器（生成目标文件） → 目标文件 (.o) 
  → 链接器（合并目标文件、库、启动代码） → 可执行文件
```

---

### **示例：最小可执行文件**
```c
// main.c
int _start() {  // 直接定义入口函数（通常由启动代码调用 main）
    asm("mov $60, %eax\n"  // exit 系统调用号
        "mov $42, %edi\n"  // 返回值
        "syscall");
}
```
编译命令：
```bash
gcc -nostdlib -static -o minimal main.c  # 禁用标准库和动态链接
```

---

### **总结**
| **组成部分**        | **作用**                               | **是否必需**           |
|---------------------|---------------------------------------|------------------------|
| 程序头和段信息       | 定义内存布局和加载方式                 | 必需                   |
| 代码段（.text）      | 存储机器指令                           | 必需                   |
| 数据段（.data/.bss） | 存储全局/静态变量                      | 必需（若有此类变量）   |
| 符号表              | 调试和动态链接支持                     | 可选（可剥离）         |
| 动态链接信息         | 动态库依赖和符号解析                   | 仅动态链接程序需要     |
| 启动代码            | 初始化环境并调用 `main`                | 必需                   |
| 运行时库            | 提供标准函数（如 `printf`、`malloc`） | 依赖代码是否使用库函数 |

通过理解这些组成部分，可以更好地调试程序、优化二进制大小或分析底层行为。

## 二、Core Dump elf文件

在 Linux 系统中，当程序因段错误（Segmentation Fault）、非法指令或其他严重信号（如 `SIGSEGV`、`SIGABRT`）而崩溃时，操作系统会生成一个 **Core Dump 文件**（通常是 `core` 或 `core.<pid>`）。这个文件本质上是一个 **ELF 格式的二进制文件**，记录了程序崩溃时的完整内存状态和运行上下文。以下是其保存的核心内容：

---

### **1. 核心转储文件包含的内容**
| **内容**                | **作用**                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| **程序崩溃时的内存快照** | 包括代码段（`.text`）、数据段（`.data`、`.bss`）、堆（`heap`）、栈（`stack`）等内存区域的完整副本。 |
| **寄存器状态**           | 保存所有 CPU 寄存器的值（如 `RIP`/`EIP` 指向崩溃时的指令地址）。         |
| **程序头信息**           | ELF 程序头（Program Headers），描述内存段的加载方式和权限。               |
| **信号信息**             | 导致程序崩溃的信号类型（如 `SIGSEGV` 表示段错误）。                       |
| **线程和进程状态**       | 进程的线程状态、文件描述符表、环境变量、命令行参数等。                    |
| **共享库映射信息**       | 加载的共享库（如 `libc.so`）的内存地址和路径。                            |

---

### **2. 为什么 Core Dump 可以复现段错误问题**
Core Dump 文件通过以下机制帮助开发者复现和调试段错误：

#### **(1) 精确还原崩溃现场**
- **指令指针（RIP/EIP）**：直接指向触发段错误的机器指令地址，结合调试符号（`-g` 编译选项）可定位到源代码行。
- **内存访问地址**：通过寄存器（如 `RAX`、`RDX`）或堆栈中的指针值，确定程序试图访问的非法内存地址。
  ```bash
  # 使用 GDB 查看崩溃时的指令和内存地址
  gdb ./your_program core
  (gdb) bt          # 查看堆栈回溯
  (gdb) info reg    # 查看寄存器值
  (gdb) x/x 0xdeadbeef  # 检查非法地址内容
  ```

#### **(2) 堆栈回溯（Backtrace）**
- **调用链分析**：Core Dump 保存了完整的堆栈帧信息，可还原函数调用链，找到触发崩溃的代码路径。
  ```text
  # 示例堆栈回溯
  #0  0x0000555555555141 in crash_function () at main.c:10
  #1  0x000055555555516a in main () at main.c:15
  ```

#### **(3) 内存映射验证**
- **内存权限检查**：通过 `/proc/<pid>/maps` 的镜像信息，确认崩溃地址是否属于合法内存区域（如访问只读代码段或未映射区域）。
  ```bash
  # 查看崩溃时的内存映射
  (gdb) info proc mappings
  ```

#### **(4) 数据状态分析**
- **变量值检查**：查看崩溃时的全局变量、局部变量和堆内存内容，判断是否存在缓冲区溢出、空指针解引用等问题。
  ```bash
  # 查看变量值
  (gdb) print *ptr
  ```

---

### **3. 生成 Core Dump 的条件**
要生成 Core Dump 文件，需满足以下条件：
1. **系统配置允许生成 Core Dump**：
   ```bash
   ulimit -c unlimited  # 解除 Core Dump 大小限制
   ```
2. **程序崩溃时未被捕获信号**：例如未通过 `signal(SIGSEGV, handler)` 拦截 `SIGSEGV`。
3. **文件系统权限**：当前目录有写入权限，且磁盘空间充足。
4. **操作系统支持**：Linux 默认启用 Core Dump，路径可通过 `/proc/sys/kernel/core_pattern` 配置。

---

### **4. 分析 Core Dump 的步骤**
1. **编译程序时包含调试信息**：
   ```bash
   gcc -g -o your_program your_program.c
   ```
2. **加载 Core Dump 文件**：
   ```bash
   gdb ./your_program core
   ```
3. **定位问题**：
   - 使用 `bt` 查看堆栈回溯。
   - 使用 `info reg` 检查寄存器状态。
   - 使用 `x` 命令分析内存内容。

---

### **5. 经典段错误场景与 Core Dump 分析示例**

#### **场景：空指针解引用**
```c
// main.c
int main() {
    int *ptr = NULL;
    *ptr = 42;  // 触发 SIGSEGV
    return 0;
}
```

**分析步骤**：
1. 生成 Core Dump：
   ```bash
   ulimit -c unlimited
   ./main  # 崩溃后生成 core 文件
   ```
2. 使用 GDB 调试：
   ```bash
   gdb ./main core
   (gdb) bt
   # 输出：
   #0  0x0000555555555151 in main () at main.c:3
   (gdb) info reg rip
   rip   0x555555555151  0x555555555151 <main+9>
   ```
3. 结论：`main.c` 第 3 行尝试向空指针写入数据。

---

### **总结**
Core Dump 文件通过保存程序崩溃时的完整内存和寄存器状态，使得开发者能够精确复现段错误问题，结合调试工具（如 GDB）快速定位根本原因。这是解决内存相关错误（如段错误、堆溢出、非法指令）的终极调试手段。


## 三、R_X86_64_32, R_X86_64_PLT32区别

在 x86-64 架构的 ELF 文件（可执行文件或共享库）中，`R_X86_64_32` 和 `R_X86_64_PLT32` 是两种常见的 **重定位类型（Relocation Types）**，它们描述了链接器在符号解析时需要执行的不同操作。以下是它们的核心区别和应用场景：

---

### **1. `R_X86_64_32`**
#### **定义与用途**
- **类型**：绝对地址重定位（32 位）。
- **场景**：用于 **静态链接** 或 **非位置无关代码（Non-PIC）** 的场景。
- **行为**：  
  链接器直接将符号的 **绝对地址** 写入目标位置（例如全局变量的地址）。  
  要求符号地址在链接时确定，且最终地址在 32 位地址空间内（即 0x00000000 到 0xFFFFFFFF）。

#### **示例**
```c
// 全局变量（静态链接时可能触发 R_X86_64_32）
int global_var = 42;

// 代码中访问全局变量
int main() {
    return global_var;  // 编译后生成对 global_var 的绝对地址引用
}
```

#### **限制**
- **不支持动态链接**：若符号定义在共享库中，无法通过动态链接解析。
- **地址空间限制**：仅适用于 32 位地址（x86-64 的虚拟地址通常为 64 位，但低 32 位可能有效）。

---

### **2. `R_X86_64_PLT32`**
#### **定义与用途**
- **类型**：相对地址重定位（32 位偏移），通过 **过程链接表（PLT, Procedure Linkage Table）** 实现动态链接。
- **场景**：用于 **动态链接** 和 **位置无关代码（PIC）** 的场景。
- **行为**：  
  链接器生成对 PLT 项的引用，PLT 项在运行时通过 **延迟绑定（Lazy Binding）** 解析到实际函数地址。

#### **示例**
```c
// 调用共享库中的函数（触发 R_X86_64_PLT32）
#include <stdio.h>
int main() {
    printf("Hello, World!\n");  // 动态链接库函数
    return 0;
}
```

#### **优势**
- **动态链接支持**：允许运行时解析共享库中的符号。
- **位置无关性**：通过 PLT 的间接跳转，支持代码在内存中的任意位置加载（PIC）。

---

### **关键区别对比**
| **特性**               | `R_X86_64_32`                  | `R_X86_64_PLT32`               |
|------------------------|--------------------------------|--------------------------------|
| **重定位目标**         | 符号的绝对地址（32 位）         | 符号在 PLT 中的相对偏移（32 位） |
| **链接类型**           | 静态链接或非 PIC 代码          | 动态链接或 PIC 代码            |
| **运行时解析**         | 不支持（地址在链接时确定）      | 支持（通过 PLT 和 GOT 动态解析） |
| **适用符号类型**       | 全局变量、静态函数             | 动态库中的函数                 |
| **地址空间限制**       | 限于 32 位地址（可能不兼容 ASLR） | 无限制（支持 64 位地址空间）    |
| **生成条件**           | 编译时未启用 `-fPIC`           | 编译时启用 `-fPIC` 或动态链接   |

---

### **技术细节**
#### **`R_X86_64_32` 的链接过程**
- **输入**：符号的绝对地址（如 `0x601020`）。
- **操作**：将地址直接写入目标位置。
- **限制**：若符号位于共享库中，链接器无法确定其绝对地址，导致链接失败。

#### **`R_X86_64_PLT32` 的链接过程**
- **输入**：符号在 PLT 中的条目偏移（如 `printf@plt` 的地址）。
- **操作**：计算 `目标地址 = PLT条目地址 + 偏移`。
- **运行时行为**：  
  1. 首次调用函数时，PLT 条目跳转到动态链接器（`ld.so`），解析函数实际地址并更新 GOT（Global Offset Table）。
  2. 后续调用直接通过 GOT 跳转到实际地址。

---

### **示例：反汇编对比**
#### **`R_X86_64_32` 代码**
```asm
; 访问全局变量（绝对地址）
mov eax, DWORD PTR [rip + 0x200000]  # 假设 global_var 的地址为 0x601020
```

#### **`R_X86_64_PLT32` 代码**
```asm
; 调用动态库函数（通过 PLT）
call printf@plt
```

---

### **常见问题与解决方案**
#### **问题 1：链接时出现 `relocation R_X86_64_32 against symbol ... can not be used when making a shared object`**
- **原因**：在编译共享库（`.so`）时未启用 `-fPIC`，导致生成了 `R_X86_64_32` 重定位。
- **解决**：重新编译源码并添加 `-fPIC` 选项：
  ```bash
  gcc -fPIC -shared -o libfoo.so foo.c
  ```

#### **问题 2：动态库中函数调用性能问题**
- **原因**：PLT 的延迟绑定引入额外开销。
- **解决**：在程序启动时绑定所有符号（禁用延迟绑定）：
  ```bash
  gcc -Wl,-z,now -o program main.c -lfoo
  ```

---

### **总结**
- **`R_X86_64_32`**：用于静态地址引用，适合静态链接或绝对地址已知的场景，但无法支持动态库和 ASLR。
- **`R_X86_64_PLT32`**：通过 PLT 实现动态链接，是共享库和位置无关代码的基石，支持灵活的运行时地址解析。

理解这两者的区别有助于解决链接错误、优化代码性能，并正确设计跨模块的交互逻辑。

## 四、elf_to_fle源码解读

该函数将 ELF 二进制文件的特定 section 转换为自定义的 **FLE (Format for Linked Executables) JSON 格式**，包含：
1. 原始字节数据
2. 符号定义（全局/局部）
3. 重定位信息（支持特定类型）
4. 结构化数据表示

### **1. 提取 Section 原始数据**
```python
section_data = subprocess.check_output(
    ['objcopy', '--dump-section', f'{section}=/dev/stdout', binary],
    stderr=subprocess.DEVNULL,
)
```
- **工具**：`objcopy`
- **功能**：提取指定 section 的二进制内容
- **输出**：原始字节数据 (`bytes` 类型)

### **2. 获取重定位信息**
```python
relocs = subprocess.check_output(
    ['readelf', '-r', binary],
    stderr=subprocess.DEVNULL, text=True,
)
```
- **工具**：`readelf -r`
- **功能**：解析 ELF 的重定位表
- **输出**：文本格式的重定位信息

### **3. 获取符号表信息**
```python
names = subprocess.check_output(
    ['objdump', '-t', binary],
    stderr=subprocess.DEVNULL, text=True,
)
```
- **工具**：`objdump -t`
- **功能**：提取符号表（包括全局/局部符号）
- **输出**：文本格式的符号信息

### **4. 解析符号表**
```python
Symbol = namedtuple('Symbol', 'symb_type section offset name')
symbols = []
for line in names.splitlines():
    pattern = r'^([0-9a-fA-F]+)\s+(l|g)\s+(\w+)?\s+([.a-zA-Z0-9_]+)\s+([0-9a-fA-F]+)\s+(.*)$'
    if match := re.match(pattern, line):
        offset, symb_type, _, sec, _, name = match.groups()
        symbols.append(
            Symbol(symb_type, sec, int(offset, 16), name.replace('.', '_'))
        )
```
- **数据结构**：`Symbol` 命名元组（类型、节名、偏移量、名称）
- **正则模式**：解析 `objdump -t` 的每行输出
- **关键处理**：
  - 将符号名称中的 `.` 替换为 `_`（避免 JSON 冲突）
  - 十六进制偏移量转为整数

### **5. 解析重定位信息**
```python
relocations, enabled = {}, True
for line in relocs.splitlines():
    if 'Relocation section' in line:
        enabled = ('.rela' + section) in line
    elif enabled:
        pattern = r'^\s*([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+(\S+)\s+([0-9a-fA-F]+)\s+(.*)$'
        if match := re.match(pattern, line):
            offset, _, reloc, *_, expr = match.groups()
            # ...表达式处理...
            if reloc not in ['R_X86_64_PC32', 'R_X86_64_PLT32', 'R_X86_64_32']:
                raise Exception(f'Unsupported relocation {reloc}')
            relocations[int(offset, 16)] = (skip, expr.replace('.', '_'))
```
- **过滤机制**：只处理目标 section 的重定位（`.rela<section>`）
- **支持的重定位类型**：
  - `R_X86_64_PC32`（32位相对地址）
  - `R_X86_64_PLT32`（过程链接表）
  - `R_X86_64_32`（32位绝对地址）
- **表达式处理**：
  - 将数字转为十六进制格式（`0x...`）
  - 符号名中的 `.` 替换为 `_`
  - 添加 `i32()` 包装

### **6. 生成 FLE 格式**
```python
res, skip, holding = [], 0, []

def do_dump(holding: list):
    if holding:
        res.append(
            f'{BYTES}: ' +  # 假设 BYTES 是常量（如 "BYTES"）
            ' '.join([f'{x:02x}' for x in holding])
        )
        holding.clear()
    
dump = lambda holding=holding: do_dump(holding)

# 遍历 section 每个字节
for i, b in enumerate(section_data):
    # 1. 处理符号定义
    for sym in symbols:
        if sym.section == section and sym.offset == i:
            dump()
            prefix = LOCAL if sym.symb_type == 'l' else GLOBL  # 假设常量
            res.append(f'{prefix}: {sym.name}')
    
    # 2. 处理重定位
    if i in relocations:
        skip, reloc = relocations[i]
        dump()
        res.append(f'{RELOC}: {reloc}')  # 假设 RELOC 是常量
    
    # 3. 收集原始字节
    if skip > 0:  # 跳过重定位占用的字节
        skip -= 1
    else:
        holding.append(b)
        if len(holding) == 16:  # 每16字节输出一次
            dump()
dump()  # 输出剩余字节
```
- **输出格式**：
  - `BYTES: xx xx xx ...`（原始字节）
  - `LOCAL: symbol_name`（局部符号）
  - `GLOBL: symbol_name`（全局符号）
  - `RELOC: i32(0x... - CURRENT)`（重定位表达式）
- **缓冲机制**：
  - `holding` 列表缓存字节（每16字节刷新）
  - 遇到符号/重定位时强制刷新


## 五、FLE_ld源码解读

`FLE_ld` 是一个自定义链接器，它将多个 FLE 格式的目标文件链接成单个可执行文件。主要功能包括：
1. 解析多个 FLE 格式的目标文件
2. 处理符号解析（全局和局部符号）
3. 解析重定位表达式
4. 生成最终的可执行文件格式

### 1. 输入处理与初始化
```python
sections, fles = defaultdict(list), {}

# 解析命令行参数
for i, fle in enumerate(args):
    if fle == '-o': continue
    if i > 0 and args[i - 1] == '-o':
        dest = Path(args[i])  # 获取输出文件路径
    else:
        # 读取并解析 FLE 文件
        fle_content = json.loads(Path(fle).read_text())
        assert fle_content['type'] == '.obj'  # 验证文件类型
        
        fles[fle] = fle_content  # 存储文件内容
        # 收集所有段(section)
        for k, v in fle_content.items():
            if k.startswith('.'):
                sections[k] += [(fle, v)]  # 按文件名和段内容存储

assert dest  # 确保输出路径已设置
```

### 2. 符号表初始化
```python
symbols = dict(
    i32=lambda x: x.to_bytes(4, 'little', signed=True),  # 32位整数转换函数
)
```

### 3. 两遍链接过程
```python
for iter in range(2):  # 第一遍解析符号，第二遍处理重定位
    res = b''  # 最终二进制结果
    
    # 遍历所有段
    for sec_name, sec in sections.items():
        for fle, sec_body in sec:
            # 记录当前段在输出中的位置
            symbols[f'{fle}.{sec_name.replace(".", "_")}'] = len(res)
            
            # 处理段内容
            for item in sec_body:
                match item.split(': '):
                    # 处理不同类型的段项
```

### 4. 符号处理（📤 和 🏷️）
```python
case '📤', symb:  # 全局符号定义
    # 第一遍检查多重定义
    if iter == 0 and symb in symbols:
        raise Exception(f'Multiple definition of {symb}')
    # 记录符号位置
    symbols[symb] = len(res)

case '🏷️', symb:  # 局部符号定义
    # 格式: {文件名}.{符号名}
    symbols[f'{fle}.{symb}'] = len(res)
```

### 5. 字节数据处理（🔢）
```python
case '🔢', xs:  # 原始字节数据
    res += bytes.fromhex(xs)  # 添加十六进制字节
```

### 6. 重定位处理（❓）
```python
case '❓', expr:  # 重定位表达式
    # 自定义符号查找类
    class Symbols(dict):
        def __getitem__(self, key):
            if iter == 0:  # 第一遍：符号解析
                return symbols.get(key, 0)  # 未定义符号返回0
            else:  # 第二遍：实际重定位
                # 优先查找局部符号（带文件名前缀）
                if (key_l := f'{fle}.{key}') in symbols:
                    return symbols[key_l]
                # 查找全局符号
                elif key in symbols:
                    return symbols[key]
                else:
                    raise Exception(f'Undefined symbol: {key}')
    
    # 替换 CURRENT 为当前位置
    expr = expr.replace(CURRENT, f'({len(res)})')
    # 计算重定位值
    reloc = eval(expr, {}, Symbols())
    # 添加重定位结果
    res += reloc
```

### 7. 生成最终可执行文件
```python
# 提取全局符号
symbols_global = {
    k: v for k, v in symbols.items()
        if type(v) == int and '.' not in k  # 纯符号名（无文件名前缀）
}

# 将二进制数据分块（每16字节）
parts = [res[i:i+16] for i in range(0, len(res), 16)]

# 写入可执行文件
dest.write_text(
    '#!./exec\n\n' +  # Shebang 指定解释器
    json.dumps({
        'type': '.exe',  # 文件类型标识
        'symbols': symbols_global,  # 全局符号表
        '.load': [  # 二进制数据段
            f'{BYTES}: ' + ' '.join([f'{b:02x}' for b in part])
            for part in parts
        ]
    },
    ensure_ascii=False, indent=4))

# 设置可执行权限
dest.chmod(0o755)
```

### 8. 关键设计解析

#### 两遍链接 (Two-pass Linking)
- **第一遍 (iter=0)**:
  - 收集所有符号定义
  - 检查符号冲突（多重定义）
  - 建立符号地址映射
  - 忽略重定位计算（返回0）

- **第二遍 (iter=1)**:
  - 使用已解析的符号表
  - 计算重定位表达式
  - 生成最终二进制内容

#### 符号处理
- **全局符号 (📤)**: 
  - 跨文件可见
  - 名称直接存储在符号表中
  - 链接时检查多重定义

- **局部符号 (🏷️)**:
  - 文件内可见
  - 存储为 `{文件名}.{符号名}` 格式
  - 避免命名冲突

#### 重定位机制
- **表达式求值**:
  - 使用 Python 的 `eval()` 计算表达式
  - `CURRENT` 替换为当前位置
  - 符号查找通过自定义 `Symbols` 类完成

- **重定位类型**:
  - 支持 `i32()` 函数（32位整数转换）
  - 示例表达式: `i32(print - 0x4 - CURRENT)`

#### 输出格式
- **Shebang**: `#!./exec` 指定自定义解释器
- **JSON 结构**:
  - 类型标识: `.exe`
  - 全局符号表
  - 二进制数据块（十六进制格式）

### 9. 使用示例

#### 输入文件示例
```json
{
  "type": ".obj",
  ".text": [
    "📤: main",
    "🔢: 48 89 e5",
    "❓: i32(printf - 0x4 - CURRENT)"
  ]
}
```

#### 输出文件示例
```json
#!./exec

{
    "type": ".exe",
    "symbols": {
        "main": 0,
        "printf": 4195600
    },
    ".load": [
        "BYTES: 48 89 e5",
        "BYTES: 90 12 40 00"  // 重定位结果
    ]
}
```
