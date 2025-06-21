本文主要整理NJU-OS多处理器编程章节的要点。

## 一、多线程栈4kB内存

在多线程环境中，线程栈之间的 4kB 内存区域是操作系统为保护线程栈安全而设置的**保护页（Guard Page）**。这是一个重要的内存保护机制，用于防止栈溢出导致的安全问题。以下是详细解释：

### 1. 保护页的作用

| 功能 | 描述 |
|------|------|
| **栈溢出检测** | 当线程尝试访问保护页时会触发段错误（SIGSEGV） |
| **内存隔离** | 防止一个线程的栈溢出破坏相邻线程的栈数据 |
| **安全防护** | 阻止恶意或错误代码通过栈溢出进行攻击 |

### 2. 内存布局示例

典型的线程栈内存布局（Linux x86-64 系统）：
```
高地址
+-------------------+ 
| 线程2的栈底        | 0x7FFF0000A000
| ...               |
| 线程2的栈数据      |
|-------------------|
| 4kB 保护页         | ← 不可访问区域 (触发 SIGSEGV)
|-------------------|
| 线程1的栈顶        | 0x7FFF00006000
| ...               |
| 线程1的栈数据      |
| 线程1的栈底        | 0x7FFF00001000
低地址
```

### 3. 技术实现细节

1. **内存分配**：
   ```c
   // 伪代码：线程栈分配
   void* create_thread_stack() {
       size_t stack_size = 2 * 1024 * 1024; // 2MB 栈大小
       size_t total_size = stack_size + 4096; // 额外 4kB 保护页
       
       void* stack = mmap(NULL, total_size, 
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         -1, 0);
       
       // 设置顶部 4kB 为不可访问
       mprotect(stack, 4096, PROT_NONE);
       
       return stack + 4096; // 返回可用的栈起始地址
   }
   ```

2. **工作流程**：
   ```mermaid
   graph TD
       A[线程创建] --> B[分配栈内存]
       B --> C[设置顶部4kB为PROT_NONE]
       C --> D[线程使用栈空间]
       D -->|栈溢出| E[访问保护页]
       E --> F[触发SIGSEGV]
       F --> G[操作系统终止程序]
   ```

### 4. 为什么是 4kB？

| 原因 | 说明 |
|------|------|
| **内存页大小** | x86/x86-64 架构的标准内存页大小是 4kB |
| **硬件支持** | MMU（内存管理单元）以 4kB 为单位管理内存权限 |
| **效率考量** | 单页保护在性能和内存开销间取得平衡 |

### 5. 实际观察示例

使用 GDB 查看线程栈布局：
```gdb
(gdb) info threads
  Id   Target Id         Frame 
* 1    Thread 0x7ffff7da2740 (LWP 1234) main () at main.c:10
  2    Thread 0x7ffff75fe700 (LWP 1235) thread_func () at thread.c:5

(gdb) thread 2
(gdb) info proc mappings
...
0x7ffff75fe000     0x7ffff77ff000    0x21000     0x0 [stack:1235]
```

计算栈空间：
```
栈大小 = 0x7ffff77ff000 - 0x7ffff75fe000 = 0x21000 (135168 bytes = 132KB)
实际可用栈 = 132KB - 4KB = 128KB
```

### 6. 编程注意事项

1. **避免栈溢出**：
   ```c
   // 危险：大数组可能溢出
   void risky_function() {
       char buffer[1024*1024]; // 1MB 栈分配 → 可能溢出!
       // ...
   }
   
   // 安全：改用堆分配
   void safe_function() {
       char* buffer = malloc(1024*1024);
       // ...
       free(buffer);
   }
   ```

2. **调整保护页大小**：
   ```c
   // 设置线程属性 (Linux)
   pthread_attr_t attr;
   pthread_attr_init(&attr);
   pthread_attr_setguardsize(&attr, 8192); // 8kB 保护页
   ```

### 7. 不同系统的实现差异

| 系统 | 保护页实现 | 默认大小 |
|------|------------|----------|
| **Linux** | 使用 `mprotect()` 设置 | 4kB |
| **Windows** | `PAGE_GUARD` 标志位 | 4kB |
| **macOS** | `VM_FLAGS_GUARD` 标志 | 4kB |
| **FreeBSD** | `MAP_GUARD` 标志 | 4kB |

### 总结

线程栈之间的 4kB 内存是：
1. 操作系统强制设置的**内存保护区域**
2. 基于硬件页大小（4kB）的**安全边界**
3. 用于检测和阻止**栈溢出错误**的关键机制
4. 多线程编程中**内存安全**的基础保障

理解这个机制有助于：
- 诊断栈溢出导致的段错误（SIGSEGV）
- 优化线程栈内存使用
- 设计更安全的并发程序
- 理解操作系统内存保护原理

## 二、T_sum编译器行为

```c
#define N 100000000
long sum = 0;

void T_sum() { for (int i = 0; i < N; i++) sum++; }

int main() {
    create(T_sum);
    create(T_sum);
    join();
    printf("sum = %ld\n", sum);
} 
```

这段代码在多线程环境下存在严重的数据竞争问题。编译器在不同优化级别（`-O1`和`-O2`）下会进行不同的优化，这会导致程序行为发生显著变化。以下是详细分析：

### 1. 代码问题分析
```c
#define N 100000000
long sum = 0;  // 全局共享变量

void T_sum() {
    for (int i = 0; i < N; i++) 
        sum++;  // 非原子操作，存在数据竞争
}
```
关键问题：
1. **数据竞争**：两个线程同时修改共享变量 `sum`
2. **非原子操作**：`sum++` 不是原子操作，实际包含：
   - 从内存加载值到寄存器
   - 寄存器加1
   - 结果存回内存
3. **编译器优化**：不同优化级别会改变代码行为

---

### 2. `-O1` 优化级别下的行为
```assembly
# gcc -S -O1 -fno-inline 生成的汇编代码
T_sum:
.LFB1:
    movq    sum(%rip), %rax  # 加载sum到rax
    movl    $100000000, %edx # edx = N
.L2:
    addq    $1, %rax         # rax += 1
    subl    $1, %edx         # edx -= 1
    jne     .L2              # 循环直到edx=0
    movq    %rax, sum(%rip)  # 存储结果回sum
    ret
```

#### 优化特点：
1. **循环优化**：
   - 将 `sum++` 优化为寄存器累加
   - 循环结束后才将结果写回内存
2. **实际执行逻辑**：
   ```c
   void T_sum() {
       long temp = sum;
       for (int i = 0; i < N; i++) {
           temp++;
       }
       sum = temp;  // 最后一次性写回
   }
   ```

#### 多线程影响：
- **数据竞争结果**：
  - 两个线程都读取初始值 `sum=0`
  - 每个线程在寄存器中完成1亿次加法
  - 最后都尝试将1亿写回内存
  - **实际结果≈1亿**（后写入的线程覆盖前一个线程的结果）

---

### 3. `-O2` 优化级别下的行为
```assembly
# gcc -S -O2 生成的汇编代码
T_sum:
.LFB1:
    addq    $100000000, sum(%rip)  # 直接加1亿
    ret
```

#### 优化特点：
1. **循环消除**：
   - 编译器识别出循环是简单累加
   - 直接优化为单条加法指令
2. **实际执行逻辑**：
   ```c
   void T_sum() {
       sum += 100000000;  // 直接加1亿
   }
   ```

#### 多线程影响：
- **数据竞争结果**：
  - 两个线程都执行 `sum += 100000000`
  - 该操作在汇编层面仍不是原子的：
    ```mermaid
    sequenceDiagram
        participant T1 as Thread1
        participant T2 as Thread2
        participant MEM as Memory(sum)
        
        T1->>MEM: 读取sum值(0)
        T2->>MEM: 读取sum值(0)
        T1->>T1: 计算0+1亿=1亿
        T2->>T2: 计算0+1亿=1亿
        T1->>MEM: 写入1亿
        T2->>MEM: 写入1亿(覆盖)
    ```
  - **最终结果≈1亿**（后写入的线程覆盖前一个线程的结果）

---

### 4. 不同优化级别结果对比

| 优化级别 | 循环优化 | 实际操作 | 预期结果 | 实际结果 |
|---------|----------|----------|----------|----------|
| **-O0** (无优化) | 无 | 1亿次内存访问 | 2亿 | ≈1亿 (CPU缓存竞争) |
| **-O1** | 寄存器累加 | 1次内存写入 | 2亿 | ≈1亿 (写回覆盖) |
| **-O2** | 循环消除 | 单次加法 | 2亿 | ≈1亿 (加法覆盖) |

#### 实测结果示例：
```bash
# 编译运行
$ gcc -O1 -pthread test.c && ./a.out
sum = 100000000

$ gcc -O2 -pthread test.c && ./a.out
sum = 100000000
```

---

### 5. 为什么结果总是≈1亿？

1. **内存访问模式**：
   - 两个线程几乎同时开始执行
   - 都读取到初始值 `sum=0`
   - 都计算相同的结果（1亿）
   - 最后写回时互相覆盖

2. **时间窗口**：
   - 线程切换时间 >> 加法操作时间
   - 写回操作几乎同时发生
   - 很难出现交替执行的情况

---

### 6. 正确实现方案

#### 方案1：使用原子操作 (C11)
```c
#include <stdatomic.h>
atomic_long sum = 0;

void T_sum() {
    for (int i = 0; i < N; i++) 
        atomic_fetch_add(&sum, 1);
}
```

#### 方案2：使用互斥锁
```c
#include <pthread.h>

long sum = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void T_sum() {
    for (int i = 0; i < N; i++) {
        pthread_mutex_lock(&lock);
        sum++;
        pthread_mutex_unlock(&lock);
    }
}
```

#### 方案3：线程本地累加
```c
long sum = 0;

void T_sum() {
    long local_sum = 0;
    for (int i = 0; i < N; i++) 
        local_sum++;
    
    // 合并结果（需要同步）
    pthread_mutex_lock(&lock);
    sum += local_sum;
    pthread_mutex_unlock(&lock);
}
```

---

### 7. 编译器优化启示

1. **优化风险**：
   - 编译器不知道多线程语义
   - 激进的优化会暴露数据竞争问题
   - 使用 `volatile` **不能** 解决数据竞争

2. **正确姿势**：
   ```c
   // 错误：volatile 不保证原子性
   volatile long sum = 0; 
   
   // 正确：使用原子类型
   _Atomic long atomic_sum = 0;
   ```

3. **编译选项**：
   - 使用 `-fsanitize=thread` 检测数据竞争
   ```bash
   gcc -fsanitize=thread -O2 -pthread test.c
   ```

---

### 总结
1. **`-O1`优化**：将循环优化为寄存器累加，最后写回内存
2. **`-O2`优化**：直接消除循环，优化为单条加法指令
3. **两种优化**都会导致最终结果≈1亿而非2亿
4. **根本原因**：非原子操作 + 编译器优化暴露数据竞争
5. **解决方案**：使用原子操作或互斥锁保护共享变量

多线程编程黄金法则：**所有共享变量的访问都需要同步**，编译器优化会使未同步的代码表现出更难以预测的行为。
