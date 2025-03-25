本文主要整理RISC-V指令集基础。

## 1 RISC-V相关资料

- The RISC-V Instruction Set Manual ([Volume I](https://github.com/riscv/riscv-isa-manual/releases/download/Priv-v1.12/riscv-privileged-20211203.pdf), [Volume II](https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf))
- [ABI for riscv](https://github.com/riscv-non-isa/riscv-elf-psabi-doc)
- [RISC-V Reference Card](https://www.cl.cam.ac.uk/teaching/1617/ECAD+Arch/files/docs/RISCVGreenCardv8-20151013.pdf)

## 2 RISC-V通用寄存器

RISC-V 定义了 **32 个通用寄存器**（x0-x31），每个寄存器有对应的 **ABI 名称**（Application Binary Interface），用于规范函数调用和系统编程。以下是详细列表：

| 寄存器编号 | ABI 名称 | 用途描述 | 保存责任 |
|-----------|----------|---------|----------|
| **x0**    | **zero** | 硬编码为 0，写入无效，常用于常数生成或丢弃结果。 | 无需保存 |
| **x1**    | **ra**   | **返回地址**（Return Address），保存函数调用后的返回地址（如 `jal` 指令写入）。 | 调用者保存 |
| **x2**    | **sp**   | **栈指针**（Stack Pointer），指向当前栈顶，向下增长。 | 被调用者保存 |
| **x3**    | **gp**   | **全局指针**（Global Pointer），指向全局数据区（可选优化，减少地址计算）。 | 无需保存 |
| **x4**    | **tp**   | **线程指针**（Thread Pointer），用于线程局部存储（TLS）。 | 由系统管理 |
| **x5**    | **t0**   | **临时寄存器**，用于临时值或链接跳转（如 `auipc` 结合使用）。 | 调用者保存 |
| **x6-x7** | **t1-t2** | 临时寄存器，用于中间计算。 | 调用者保存 |
| **x8**    | **s0/fp** | **保存寄存器**或**帧指针**（Frame Pointer，可选），用于保存函数帧的基地址。 | 被调用者保存 |
| **x9**    | **s1**   | 保存寄存器，需在函数调用间保留值。 | 被调用者保存 |
| **x10-x11** | **a0-a1** | **函数参数/返回值**，传递前两个参数，返回时存储结果。 | 调用者保存 |
| **x12-x17** | **a2-a7** | 函数参数寄存器，传递第 3 到第 8 个参数。 | 调用者保存 |
| **x18-x27** | **s2-s11** | 保存寄存器，需在函数调用间保留值。 | 被调用者保存 |
| **x28-x31** | **t3-t6** | 临时寄存器，用于复杂计算或长跳转。 | 调用者保存 |

### **2.1 关键寄存器详解**
#### **(1) 零寄存器（x0/zero）**
- **只读**，任何写入操作均被忽略。
- 用途示例：
  ```asm
  addi x5, x0, 100   # x5 = 0 + 100（等同于 li x5, 100）
  beq  x6, x0, label # 若 x6 == 0，跳转到 label
  ```

#### **(2) 栈指针（x2/sp）**
- **栈向下增长**，函数调用时需手动调整栈空间：
  ```asm
  func:
    addi sp, sp, -16   # 分配 16 字节栈空间
    sw   ra, 12(sp)    # 保存返回地址
    ...
    lw   ra, 12(sp)    # 恢复返回地址
    addi sp, sp, 16    # 释放栈空间
    ret
  ```

#### **(3) 参数与返回值寄存器（a0-a7）**
- **a0-a1** 同时用于返回值，若返回值超过 64 位（如结构体），需通过内存传递。
- 示例：
  ```asm
  # 调用函数 sum(a, b)
  addi a0, x0, 10    # a0 = 10（第一个参数）
  addi a1, x0, 20    # a1 = 20（第二个参数）
  jal  ra, sum       # 调用函数
  ```

#### **(4) 保存寄存器（s0-s11）**
- **被调用者必须保存**：若函数内修改了这些寄存器，需在栈中保存原始值。
- 示例：
  ```asm
  func:
    addi sp, sp, -8
    sw   s0, 4(sp)    # 保存 s0
    sw   s1, 0(sp)    # 保存 s1
    ...
    lw   s1, 0(sp)
    lw   s0, 4(sp)
    addi sp, sp, 8
    ret
  ```

### **2.2 特殊寄存器**
#### **(1) 程序计数器（PC）**
- 存储当前指令地址，不可直接修改，通过跳转指令（`jal`, `jalr`, `beq` 等）间接控制。

#### **(2) 浮点寄存器（f0-f31）**
- 若支持 **F/D 扩展**（单/双精度浮点），浮点寄存器用于浮点运算。
- 命名规则：
  - `f0-f7`: 临时浮点寄存器（调用者保存）。
  - `f8-f9`: 保存浮点寄存器（被调用者保存）。
  - `f10-f11`: 浮点参数/返回值。
  - `f12-f17`: 浮点参数寄存器。

### **2.3 寄存器使用规范**
#### **(1) 调用者保存（Caller-Saved）**
- **临时寄存器（t0-t6, a0-a7）**：调用函数前，若需保留这些寄存器的值，调用者需自行保存。
- 示例：
  ```asm
  # 调用函数前保存 t0
  addi sp, sp, -4
  sw   t0, 0(sp)
  jal  ra, func
  lw   t0, 0(sp)
  addi sp, sp, 4
  ```

#### **(2) 被调用者保存（Callee-Saved）**
- **保存寄存器（s0-s11, sp）**：若函数内修改了这些寄存器，被调用者需在栈中保存并恢复原值。


### **2.4 常见编程场景**
#### **(1) 函数调用**
```asm
# 调用函数 func(a0, a1)
addi a0, x0, 42     # 参数1 = 42
addi a1, x0, 100    # 参数2 = 100
jal  ra, func       # 跳转并保存返回地址到 ra
...

func:
  addi sp, sp, -16
  sw   ra, 12(sp)   # 保存返回地址
  sw   s0, 8(sp)    # 保存 s0
  ...
  lw   s0, 8(sp)
  lw   ra, 12(sp)
  addi sp, sp, 16
  ret               # 等同于 jalr x0, 0(ra)
```

#### **(2) 系统调用（ecall）**
```asm
# 输出字符串（RARS 模拟器示例）
.data
msg: .asciz "Hello, RISC-V!\n"

.text
main:
  la a0, msg        # 加载字符串地址到 a0
  li a7, 4          # 系统调用号 4（打印字符串）
  ecall
  li a7, 10         # 系统调用号 10（退出程序）
  ecall
```

### **2.5 注意事项**
1. **零寄存器（x0）的只读性**：不可用于存储中间值，但可简化指令（如清零操作 `mv x5, x0`）。
2. **栈对齐**：某些环境要求栈指针 `sp` 按特定字节对齐（如 16 字节）。
3. **浮点寄存器**：需确保目标平台支持 F/D 扩展，否则无法使用。

## 3 RISC-V指令格式

**R、I、B、U、J** 是 RISC-V 指令集的 **指令编码格式**，定义了指令在机器码中的位布局。这些格式决定了操作码（opcode）、寄存器索引、立即数等的编码方式。

### **3.1 R 型指令（Register-Register）**
- **用途**：寄存器到寄存器的算术/逻辑运算（如 `add`, `sub`, `sll`）。
- **字段布局**：
  ```
  | funct7 (7 bits) | rs2 (5 bits) | rs1 (5 bits) | funct3 (3 bits) | rd (5 bits) | opcode (7 bits) |
  ```
- **特点**：
  - 无立即数，操作数全部来自寄存器。
  - `funct7` 和 `funct3` 组合确定具体操作（如区分 `add` 和 `sub`）。
- **示例**：
  ```asm
  add x3, x1, x2   # x3 = x1 + x2
  ```
  对应机器码布局：
  - `funct7=0x00`, `rs2=x2`, `rs1=x1`, `funct3=0x0`, `rd=x3`, `opcode=0x33`（整数运算）。


### **3.2 I 型指令（Immediate）**
- **用途**：立即数操作（如 `addi`, `lw`, `jalr`）。
- **字段布局**：
  ```
  | imm[11:0] (12 bits) | rs1 (5 bits) | funct3 (3 bits) | rd (5 bits) | opcode (7 bits) |
  ```
- **特点**：
  - 12 位立即数（符号扩展为 32/64 位）。
  - 用于加载指令（`lw`, `lb`）、立即数运算（`addi`）和跳转（`jalr`）。
- **示例**：
  ```asm
  addi x5, x6, -42  # x5 = x6 + (-42)
  lw   x7, 8(x8)    # 从地址 x8+8 加载字到 x7
  ```
  - `addi` 的 `opcode=0x13`，`funct3=0x0`。
  - `lw` 的 `opcode=0x03`，`funct3=0x2`。

### **3.3 B 型指令（Branch）**
- **用途**：条件分支（如 `beq`, `bne`, `blt`）。
- **字段布局**：
  ```
  | imm[12|10:5] (7 bits) | rs2 (5 bits) | rs1 (5 bits) | funct3 (3 bits) | imm[4:1|11] (5 bits) | opcode (7 bits) |
  ```
- **特点**：
  - **12 位立即数**，但编码时拆分为多个字段（目标地址需按 2 字节对齐）。
  - 立即数计算方式：  
    `offset = imm[12] << 12 | imm[11] << 11 | imm[10:5] << 5 | imm[4:1] << 1`.
- **示例**：
  ```asm
  beq x1, x2, label  # 若 x1 == x2，跳转到 label
  ```
  - `opcode=0x63`，`funct3=0x0`（`beq`）。

### **3.4 U 型指令（Upper Immediate）**
- **用途**：加载大立即数到高位（如 `lui`, `auipc`）。
- **字段布局**：
  ```
  | imm[31:12] (20 bits) | rd (5 bits) | opcode (7 bits) |
  ```
- **特点**：
  - 20 位立即数，加载到目标寄存器的高 20 位，低 12 位填充 0。
  - 常用于构建 32/64 位地址或大常数（结合 `addi`）。
- **示例**：
  ```asm
  lui x5, 0x12345    # x5 = 0x12345 << 12 = 0x12345000
  auipc x6, 0x1000   # x6 = PC + (0x1000 << 12)
  ```

### **3.5 J 型指令（Jump）**
- **用途**：长跳转（如 `jal`）。
- **字段布局**：
  ```
  | imm[20|10:1|11|19:12] (20 bits) | rd (5 bits) | opcode (7 bits) |
  ```
- **特点**：
  - **20 位立即数**，编码时拆分为多个字段（目标地址按 2 字节对齐）。
  - 立即数计算方式：  
    `offset = imm[20] << 20 | imm[19:12] << 12 | imm[11] << 11 | imm[10:1] << 1`.
- **示例**：
  ```asm
  jal ra, func       # 跳转到 func，返回地址存入 ra
  ```
  - `opcode=0x6F`，`rd=ra`（x1）。

### **3.6 S 型指令（Store）**
- **用途**：存储指令（如 `sw`, `sh`, `sb`）。
- **字段布局**：
  ```
  | imm[11:5] (7 bits) | rs2 (5 bits) | rs1 (5 bits) | funct3 (3 bits) | imm[4:0] (5 bits) | opcode (7 bits) |
  ```
- **特点**：
  - 12 位立即数拆分为 `imm[11:5]` 和 `imm[4:0]`，组合后符号扩展。
- **示例**：
  ```asm
  sw x10, 16(x11)   # 将 x10 的值存储到地址 x11+16
  ```

### **3.7 指令格式对比表**
| 类型 | 指令示例           | 立即数位数 | 操作数来源             | 典型用途               |
|------|--------------------|------------|------------------------|------------------------|
| **R** | `add`, `sub`      | 无         | rs1, rs2 → rd          | 寄存器间算术运算       |
| **I** | `addi`, `lw`      | 12 位      | rs1 + imm → rd         | 立即数运算、加载操作   |
| **S** | `sw`, `sh`        | 12 位      | rs1 + imm → 存储 rs2   | 存储到内存             |
| **B** | `beq`, `bne`      | 12 位      | rs1, rs2 → 跳转目标    | 条件分支               |
| **U** | `lui`, `auipc`    | 20 位      | 高位立即数 → rd        | 构建大地址/常数        |
| **J** | `jal`             | 20 位      | PC + 偏移 → rd         | 长跳转（函数调用）     |

## 4 RISC-V 控制与状态寄存器

RISC-V 的 **控制与状态寄存器（Control and Status Registers, CSR）** 是用于管理处理器核心行为、特权模式、中断/异常处理、性能监控等系统级功能的寄存器。它们通过专用指令（如 `csrrw`, `csrrs`）访问，并遵循严格的权限控制。

### **4.1 CSR 基础**
#### **(1) 访问指令**
- **原子操作指令**：
  ```asm
  csrrw  rd, csr, rs   # 原子交换：rd = CSR[csr], CSR[csr] = rs
  csrrs  rd, csr, rs   # 原子置位：rd = CSR[csr], CSR[csr] |= rs
  csrrc  rd, csr, rs   # 原子清除：rd = CSR[csr], CSR[csr] &= ~rs
  ```
  - 若不需要返回值，可用 `rs = x0` 或 `rd = x0` 简化操作：
    ```asm
    csrw mstatus, a0  # 等价于 csrrw x0, mstatus, a0
    csrr a0, mie      # 等价于 csrrs a0, mie, x0
    ```

#### **(2) CSR 地址编码**
- CSR 地址为 **12 位**，范围 `0x000–0xFFF`，分为：
  - **标准 CSR**：由 RISC-V 特权架构定义（如 `mstatus`, `mtvec`）。
  - **自定义 CSR**：地址 `0x7C0–0x7FF` 保留供厂商扩展。

### **4.2 主要 CSR 分类**
根据 RISC-V 特权架构（Privileged Architecture），CSR 分为以下几类：

#### **(1) 机器模式（Machine Mode）CSR**
机器模式（M-mode）是最高特权级别，用于处理硬件中断和异常。

| CSR 名称  | 地址  | 功能描述                                                                 |
|-----------|-------|--------------------------------------------------------------------------|
| **mstatus**   | 0x300 | 全局状态寄存器，包含中断使能位（MIE）、特权模式（MPP）等。               |
| **misa**      | 0x301 | 标识支持的指令集扩展（如 M、A、F、D 扩展）。                             |
| **mie**       | 0x304 | 中断使能寄存器，控制定时器、外部、软件中断的启用。                      |
| **mtvec**     | 0x305 | 陷阱向量基地址，指定中断/异常处理程序的入口地址。                        |
| **mepc**      | 0x341 | 异常程序计数器（PC），保存触发异常的指令地址。                           |
| **mcause**    | 0x342 | 异常原因编码（如中断类型、异常类型）。                                  |
| **mtval**     | 0x343 | 异常附加信息（如非法指令的编码或访问的非法地址）。                      |
| **mip**       | 0x344 | 中断挂起寄存器，记录当前待处理的中断。                                  |
| **mcycle**    | 0xB00 | 时钟周期计数器（低 64 位）。                                            |
| **mcycleh**   | 0xB80 | 时钟周期计数器（高 64 位，仅 RV32 需要）。                              |
| **minstret**  | 0xB02 | 指令执行计数器（低 64 位）。                                            |

#### **(2) 监管者模式（Supervisor Mode）CSR**
监管者模式（S-mode）用于操作系统内核，依赖 M-mode 授权。

| CSR 名称  | 地址  | 功能描述                                                                 |
|-----------|-------|--------------------------------------------------------------------------|
| **sstatus**   | 0x100 | S-mode 状态寄存器（类似 `mstatus`，但权限更低）。                       |
| **stvec**     | 0x105 | S-mode 陷阱向量基地址。                                                 |
| **sepc**      | 0x141 | S-mode 异常 PC。                                                        |
| **scause**    | 0x142 | S-mode 异常原因。                                                       |
| **stval**     | 0x143 | S-mode 异常附加信息。                                                   |
| **satp**      | 0x180 | 页表基地址寄存器，用于虚拟内存管理。                                    |

#### **(3) 用户模式（User Mode）CSR**
用户模式（U-mode）是应用程序运行的最低特权级别。

| CSR 名称  | 地址  | 功能描述                                                                 |
|-----------|-------|--------------------------------------------------------------------------|
| **ustatus**   | 0x000 | U-mode 状态寄存器。                                                     |
| **utvec**     | 0x005 | U-mode 陷阱向量基地址（需硬件支持）。                                   |
| **uepc**      | 0x041 | U-mode 异常 PC。                                                        |
| **ucause**    | 0x042 | U-mode 异常原因。                                                       |
| **utval**     | 0x043 | U-mode 异常附加信息。                                                   |

#### **(4) 通用 CSR**
| CSR 名称  | 地址  | 功能描述                                                                 |
|-----------|-------|--------------------------------------------------------------------------|
| **cycle**     | 0xC00 | 用户态可读的时钟周期计数器（低 64 位）。                                |
| **time**      | 0xC01 | 实时时钟计数器（通常映射到外部时钟源）。                                |
| **instret**   | 0xC02 | 用户态可读的指令执行计数器（低 64 位）。                                |

### **4.3 关键 CSR 详解**
#### **(1) mstatus（Machine Status Register）**
- **字段**：
  - **MIE**（Machine Interrupt Enable）：全局中断使能位（1=启用）。
  - **MPP**（Machine Previous Privilege）：异常前的特权模式（M/S/U）。
  - **FS**（Floating-Point Status）：浮点单元状态（初始/干净/脏）。
- **用途**：控制中断和特权模式切换。

#### **(2) mtvec（Machine Trap Vector）**
- **模式**：
  - **Direct 模式**：所有陷阱跳转到基地址 `mtvec.base`。
  - **Vectored 模式**：中断跳转到 `mtvec.base + 4 × cause`，异常跳转到基地址。
- **配置示例**：
  ```asm
  # 设置陷阱处理程序为 direct 模式
  la   t0, trap_handler
  csrw mtvec, t0
  ```

#### **(3) mie 与 mip（Interrupt Enable/Pending）**
- **mie**：控制中断是否被响应。
  - **MTIE**（Machine Timer Interrupt Enable）：定时器中断。
  - **MEIE**（Machine External Interrupt Enable）：外部中断。
  - **MSIE**（Machine Software Interrupt Enable）：软件中断。
- **mip**：记录当前待处理的中断（硬件自动更新）。

#### **(4) mepc（Machine Exception PC）**
- 保存触发异常的指令地址（或中断时的下一条指令地址，取决于异常类型）。
- **恢复执行**：通过 `mret` 指令跳转回 `mepc`。

### **4.4 CSR 使用示例**
#### **(1) 启用定时器中断**
```asm
# 设置定时器中断处理程序
la   t0, timer_handler
csrw mtvec, t0

# 启用机器模式定时器中断
li   t0, 0x80       # MTIE 位（mie[7]）
csrw mie, t0

# 全局中断使能
csrsi mstatus, 0x8  # 设置 MIE 位（mstatus[3]）
```

#### **(2) 异常处理框架**
```asm
trap_handler:
  # 保存上下文
  csrrw t0, mscratch, t0  # 使用 mscratch 作为临时存储
  # ... 保存其他寄存器到栈

  # 读取异常原因
  csrr  a0, mcause
  csrr  a1, mepc
  csrr  a2, mtval

  # 根据 mcause 处理异常
  bgez  a0, handle_exception  # 若最高位为 0，表示异常
  j     handle_interrupt      # 否则为中断

handle_exception:
  # 处理非法指令、缺页等异常
  ...

handle_interrupt:
  # 处理定时器、外部中断等
  ...

  # 恢复上下文并返回
  csrrw t0, mscratch, t0
  mret
```

### **4.5 权限与安全**
- **特权级别**：CSR 的访问权限由当前特权模式（M/S/U）控制。
  - 例如，用户模式无法直接访问 `mstatus` 或 `satp`。
- **陷阱委托**：通过 `medeleg` 和 `mideleg` CSR，可将特定异常/中断委托给 S-mode 处理。

### **4.6 扩展与标准**
- **Zicsr 扩展**：基础 CSR 操作指令（`csrrw`, `csrrs` 等）属于该扩展。
- **特权架构文档**：详见 [RISC-V Privileged Specification](https://riscv.org/technical/specifications/)。
