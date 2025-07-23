本文主要整理Chapter 4 Traps and system calls的要点。

## 4.3 Code: System call arguments

本章描述用户程序（`initcode.S`）如何通过系统调用机制调用 `exec`，并最终在 `sys_exec` 中执行的过程。重点在系统调用的参数传递、陷阱处理、系统调用分发和返回值处理机制。

**关键要点总结：**

1.  **用户空间发起系统调用 (initcode.S):**
    *   `initcode.S`（用户空间汇编代码）通过将 `exec` 的参数放入寄存器 `a0` 和 `a1` 来准备调用。
    *   它将系统调用号 `SYS_exec` 放入寄存器 `a7`。
    *   执行 `ecall` 指令：该指令触发一个从用户模式（User Mode）到内核模式（Supervisor Mode）的陷阱（Trap）。

2.  **陷阱处理流程:**
    *   `ecall` 指令导致硬件控制权转移到内核预设的陷阱处理程序。
    *   依次执行的内核函数链是：`uservec` -> `usertrap` -> `syscall`。
    *   `syscall` 是实际处理系统调用的核心入口函数。

3.  **系统调用分发 (syscall):**
    *   `syscall` (kernel/syscall.c:132) 从保存的陷阱帧（`trapframe`）中获取用户传入的系统调用号（即之前放入 `a7` 的值）。
    *   它使用这个数字作为索引，在系统调用表 `syscalls[]` (kernel/syscall.c:107) 中查找对应的内核实现函数。
    *   系统调用表 `syscalls[]` 是一个函数指针数组，每个系统调用号对应一个具体的实现函数（如 `sys_exec`）。
    *   对于第一个系统调用 (`SYS_exec`), `a7` 的值会使 `syscall` 调用函数 `sys_exec`。

4.  **内核函数执行:**
    *   控制权转移到具体的系统调用实现函数 `sys_exec`。

5.  **返回值处理 (syscall):**
    *   当 `sys_exec` 函数执行完毕返回时，`syscall` 将其返回值记录到当前进程的陷阱帧的 `a0` 字段 (`p->trapframe->a0`)。
    *   **RISC-V C调用约定：** 系统调用在用户空间看起来像函数调用，其返回值遵循C调用约定，存储在寄存器 `a0` 中。
    *   **返回值的约定：**
        *   负数：表示发生错误。
        *   零或正数：表示成功。

6.  **错误处理:**
    *   如果系统调用号 (`a7`) 无效（不在 `syscalls[]` 表的有效范围内），`syscall` 会打印错误信息并返回 `-1` 到用户空间。

**概述流程图:**

> **用户空间 (`initcode.S`)**：
>   (1) 参数放 `a0`, `a1`
>   (2) 系统调用号 `SYS_exec` 放 `a7`
>   (3) 执行 `ecall`
> ↓
> **内核陷阱处理**：
>   (4) 硬件陷阱 → `uservec` → `usertrap` → `syscall`
> ↓
> **内核 `syscall` 函数**：
>   (5) 从陷阱帧取 `a7` (系统调用号)
>   (6) 查表 `syscalls[a7]` → 找到 `sys_exec`
>   (7) 调用 `sys_exec`
>   (8) 接收 `sys_exec` 的返回值
>   (9) 设置 `p->trapframe->a0` = 返回值
>   (10) 返回用户空间 (或处理无效调用号错误)
> ↓
> **返回用户空间**：
>   (11) 用户进程恢复执行，`a0` 寄存器保存着系统调用 `exec()` 的返回值（可能成功或包含错误码）。

## 4.4 Code: System call arguments

本章描述内核系统调用实现如何安全地获取用户空间传递的参数（特别是数据指针），以及如何在保证安全的前提下访问用户空间内存。

**关键要点总结：** 主要解决 **参数定位** 和 **用户内存安全访问** 两大问题。

1.  **定位用户传递的参数：**
    *   **根源：** 用户程序通过C库的包装函数调用系统调用，遵循 **RISC-V C调用约定** (ABI)，参数最初存储在特定寄存器中（如 a0, a1, a2...）。
    *   **保存：** 当用户程序执行 `ecall` 陷入内核时，**陷阱处理代码（`uservec`, `usertrap`）** 将用户寄存器完整地保存到**当前进程的陷阱帧（`trapframe`）** 中。
    *   **检索：** 内核的系统调用实现函数 (如 `sys_exec`) 使用一组辅助函数来从陷阱帧中提取参数：
        *   **`argraw(n)`** (kernel/syscall.c:34): 核心函数，直接从保存的陷阱帧中获取第 `n` 个系统调用参数寄存器（如 `a0`, `a1`）的值。
        *   **`argint(n, &ip)`**: 调用 `argraw` 获取第 `n` 个参数并将其视为整数，存入 `ip` 指向的位置。
        *   **`argaddr(n, &ap)`**: 调用 `argraw` 获取第 `n` 个参数并将其视为地址（指针），存入 `ap` 指向的位置。
        *   **`argfd(n, &fd, &f)`**: 特殊化的函数，用于获取文件描述符参数（第 `n` 个），返回文件描述符号 (`fd`) 和对应的内核文件结构指针 (`f`)。


```c
// Copy a null-terminated string from user to kernel.
// Copy bytes to dst from virtual address srcva in a given page table,
// until a '\0', or max.
// Return 0 on success, -1 on error.
int
copyinstr(pagetable_t pagetable, char *dst, uint64 srcva, uint64 max)
{
  uint64 n, va0, pa0;
  int got_null = 0;

  while(got_null == 0 && max > 0){
    va0 = PGROUNDDOWN(srcva);       // 对齐到当前页起始地址
    pa0 = walkaddr(pagetable, va0); // 查询物理地址
    if(pa0 == 0)
      return -1;
    // 例：页大小 4KB，srcva=0x1002 → 可复制 4096-2=4094 字节
    n = PGSIZE - (srcva - va0);     // 当前页剩余字节
    if(n > max)
      n = max;

    char *p = (char *) (pa0 + (srcva - va0));
    while(n > 0){
      if(*p == '\0'){
        *dst = '\0';
        got_null = 1;
        break;
      } else {
        *dst = *p;
      }
      --n;
      --max;
      p++;
      dst++;
    }

    srcva = va0 + PGSIZE;
  }
  if(got_null){
    return 0;
  } else {
    return -1;
  }
}
```

2.  **处理用户空间指针参数与安全访问用户内存：**
    *   **挑战：**
        *   **无效/恶意指针：** 用户程序可能传递无效指针或意图欺骗内核访问内核内存（而非用户内存）。
        *   **页表隔离：** 内核使用**自己的页表**（`kernel_pagetable`），其映射与发起调用的用户进程的页表 (`p->pagetable`) **不同**。内核不能直接使用用户虚拟地址（`va_user`）进行访存操作。
    *   **解决方案：内核提供安全的用户空间数据拷贝函数。**
        *   **示例：`fetchstr` (kernel/syscall.c:25):** 由系统调用使用（如 `exec` 获取文件名字符串）。它封装了对 `copyinstr` 的调用。
        *   **核心函数：`copyinstr(pagetable, srcva, dst, max)` (kernel/vm.c:415):**
            *   **作用：** 从用户进程页表 `pagetable` 中的虚拟地址 `srcva` 处，安全地拷贝一个以 `\0` 结尾的字符串到内核缓冲区 `dst`，最多拷贝 `max` 字节。
            *   **关键步骤：**
                1.  **地址转换与权限验证 (核心：`walkaddr`):** `copyinstr` 调用 `walkaddr(pagetable, srcva)` (kernel/vm.c:109) 来查询用户虚拟地址 `srcva` 对应的物理地址 `pa0`。
                    *   `walkaddr` (通过 `walk`) 遍历用户页表 `pagetable` 进行地址转换。
                    *   **安全检查：** `walkaddr` **验证地址 `srcva` 是否在该进程合法的用户地址空间范围内**，防止内核访问其他内存（内核空间、其他进程空间）。
                2.  **利用内核恒等映射（Identity Mapping）:** 内核页表 (`kernel_pagetable`) 将**所有物理内存 (RAM) 按物理地址 1:1 地映射到内核虚拟地址空间**（即 `va_kernel = pa`）。
                3.  **直接拷贝：** 由于获得了合法的物理地址 `pa0` 且内核可以直接访问任意物理地址（通过恒等映射），`copyinstr` 可以直接将字符串字节从物理内存位置 `pa0` 拷贝到内核目标 `dst`。
        *   **对称操作：`copyout`:** 提及存在一个类似的函数，用于将数据从内核安全地拷贝到用户空间地址（与 `copyinstr` 方向相反）。

**概述流程图 (针对处理用户空间指针):**

> **系统调用实现 (e.g., `sys_exec`)**:
>   (1) 需要访问用户空间字符串参数指针 (`char __user *argv[]`)
>   (2) 使用 `argaddr(n, &user_pointer)` 从陷阱帧获取用户空间指针值 `user_pointer`
>   (3) 调用 `fetchstr` 或类似辅助函数访问该地址指向的数据
>
> **访问用户空间数据 (`fetchstr` -> `copyinstr`)**:
>   (4) `copyinstr` 被调用: `copyinstr(p->pagetable, user_pointer, kernel_buffer, max)`
>   ↓
>   **地址转换与验证 (`walkaddr`)**:
>   (5) `walkaddr(p->pagetable, user_pointer)`:
>       *   遍历 `p->pagetable` (用户页表) 转换 `user_pointer` (虚拟地址) 为物理地址 `pa0`
>       *   **关键: 检查 `user_pointer` 是否属于有效用户地址范围 (防止非法访问)**
>       *   若检查失败，返回错误；若成功，返回 `pa0`
>   ↓
>   **利用内核恒等映射**:
>   (6) 内核知道物理地址 `pa0`。
>   (7) 内核页表 (`kernel_pagetable`) 已将 `pa0` 映射到内核虚拟地址 `kva = pa0` (恒等映射)。
>   ↓
>   **安全数据拷贝**:
>   (8) `copyinstr` 直接使用内核虚拟地址 `kva` (等于 `pa0`) 作为源地址，进行字节拷贝到 `kernel_buffer`。