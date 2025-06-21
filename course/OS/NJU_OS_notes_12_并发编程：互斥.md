本文主要整理NJU-OS并发控制（互斥）章节的要点。

## 一、lock/unlock

# C++ 中 lock/unlock 的使用指南

在 C++ 中，`lock()` 和 `unlock()` 是用于多线程同步的基本操作，主要通过 `<mutex>` 头文件提供的类来实现。以下是详细的使用方法和最佳实践：

### 1. 基本锁操作：std::mutex

#### 直接使用 lock()/unlock()

```cpp
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx; // 创建互斥锁
int shared_data = 0;

void increment() {
    for (int i = 0; i < 100000; ++i) {
        mtx.lock();   // 获取锁
        ++shared_data;
        mtx.unlock(); // 释放锁
    }
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);
    
    t1.join();
    t2.join();
    
    std::cout << "Final value: " << shared_data << std::endl;
    // 正确输出: 200000
}
```

⚠️ **注意问题**：
- 容易忘记 unlock() 导致死锁
- 异常发生时可能无法解锁
- 不推荐直接使用

### 2. 推荐方式：RAII 锁管理

#### 2.1 std::lock_guard（C++11）
```cpp
void safe_increment() {
    for (int i = 0; i < 100000; ++i) {
        std::lock_guard<std::mutex> lock(mtx); // 自动加锁
        ++shared_data; 
        // 离开作用域时自动解锁
    }
}
```

#### 2.2 std::unique_lock（C++11）
```cpp
void flexible_increment() {
    for (int i = 0; i < 100000; ++i) {
        std::unique_lock<std::mutex> lock(mtx); // 自动加锁
        
        // 可以手动控制
        // lock.unlock();
        // ... 非关键区操作 ...
        // lock.lock();
        
        ++shared_data;
    }
}
```

### 3. 高级锁类型

#### 3.1 std::recursive_mutex（递归锁）
```cpp
std::recursive_mutex rec_mtx;

void recursive_function(int count) {
    std::lock_guard<std::recursive_mutex> lock(rec_mtx);
    if (count > 0) {
        recursive_function(count - 1); // 可重入
    }
}
```

#### 3.2 std::timed_mutex（超时锁）
```cpp
std::timed_mutex timed_mtx;

void try_lock_example() {
    if (timed_mtx.try_lock_for(std::chrono::milliseconds(100))) {
        // 成功获取锁
        timed_mtx.unlock();
    } else {
        // 超时处理
    }
}
```

### 4. 多锁管理

#### 4.1 同时锁定多个互斥量
```cpp
std::mutex mtx1, mtx2;

void multi_lock_example() {
    // 避免死锁的方式
    std::lock(mtx1, mtx2); // 同时锁定两个互斥量
    
    // 使用lock_guard管理（adopt_lock表示已锁定）
    std::lock_guard<std::mutex> lock1(mtx1, std::adopt_lock);
    std::lock_guard<std::mutex> lock2(mtx2, std::adopt_lock);
    
    // 安全操作共享资源
}
```

#### 4.2 使用 std::scoped_lock（C++17）
```cpp
void scoped_lock_example() {
    std::scoped_lock lock(mtx1, mtx2); // C++17推荐方式
    // 自动管理多个锁
}
```

### 5. 条件变量配合使用
```cpp
#include <condition_variable>

std::mutex cv_mtx;
std::condition_variable cv;
bool data_ready = false;

void producer() {
    {
        std::lock_guard<std::mutex> lock(cv_mtx);
        // 生产数据...
        data_ready = true;
    }
    cv.notify_one(); // 通知消费者
}

void consumer() {
    std::unique_lock<std::mutex> lock(cv_mtx);
    cv.wait(lock, []{ return data_ready; }); // 等待条件
    
    // 消费数据...
    data_ready = false;
}
```

### 6. 最佳实践与注意事项

1. **优先使用RAII对象**：
   - 推荐：`lock_guard` (简单场景)
   - 推荐：`unique_lock` (需要灵活控制)
   - 推荐：`scoped_lock` (多锁场景)

2. **避免直接使用lock()/unlock()**：
   - 异常安全无法保证
   - 容易遗漏unlock()

3. **锁粒度控制**：
   ```cpp
   void good_practice() {
       // 非关键区操作...
       
       {
           std::lock_guard<std::mutex> lock(mtx);
           // 最小化临界区
       }
       
       // 更多非关键区操作...
   }
   ```

4. **死锁预防**：
   - 固定锁的获取顺序
   - 使用std::lock()同时获取多个锁
   - 设置超时机制

5. **性能考虑**：
   - 避免在锁内进行I/O操作
   - 考虑读写锁（std::shared_mutex）优化
   ```cpp
   #include <shared_mutex>
   
   std::shared_mutex rw_mtx;
   
   void reader() {
       std::shared_lock lock(rw_mtx); // 共享锁
       // 读取操作...
   }
   
   void writer() {
       std::unique_lock lock(rw_mtx); // 独占锁
       // 写入操作...
   }
   ```

### 7. 错误示例与修正

#### 错误1：忘记解锁
```cpp
void danger_function() {
    mtx.lock();
    if (error_condition) {
        return; // 直接返回，忘记解锁!
    }
    mtx.unlock();
}
```

**修正**：使用RAII对象

#### 错误2：异常安全问题
```cpp
void unsafe_function() {
    mtx.lock();
    throw std::runtime_error("Oops"); // 异常抛出!
    mtx.unlock(); // 永远不会执行
}
```

**修正**：使用lock_guard保证异常安全

#### 错误3：锁顺序不一致导致死锁
```cpp
void thread1() {
    mtx1.lock();
    mtx2.lock(); // 可能死锁点
    // ...
}

void thread2() {
    mtx2.lock();
    mtx1.lock(); // 相反的顺序!
    // ...
}
```

**修正**：
```cpp
void safe_thread() {
    std::scoped_lock lock(mtx1, mtx2); // C++17
    // 或者
    std::lock(mtx1, mtx2);
    std::lock_guard lock1(mtx1, std::adopt_lock);
    std::lock_guard lock2(mtx2, std::adopt_lock);
}
```

### 总结

| 技术 | 适用场景 | C++版本 | 特点 |
|------|----------|---------|------|
| `lock()/unlock()` | 底层控制 | C++11 | 不推荐直接使用 |
| `lock_guard` | 简单临界区 | C++11 | 轻量级，自动解锁 |
| `unique_lock` | 灵活控制 | C++11 | 可转移，支持延迟锁定 |
| `scoped_lock` | 多锁管理 | C++17 | 多锁防死锁 |
| `shared_mutex` | 读写分离 | C++17 | 读多写少优化 |

**核心原则**：
1. 优先使用RAII风格的锁管理对象
2. 保持临界区尽可能小
3. 避免嵌套锁或确保一致的锁定顺序
4. 为读写场景选择适当的锁类型
5. 使用标准库而非平台特定实现

正确使用锁机制可以确保多线程程序的正确性和高效性，避免数据竞争和死锁问题。


## 二、pthread_mutex_t 

`pthread_mutex_t` 是 POSIX 线程库中用于实现互斥锁的核心数据类型，用于保护共享资源在多线程环境中的安全访问。下面从基础到高级全面介绍其使用方法。

### 1. 基本使用流程

#### 包含头文件
```c
#include <pthread.h>
```

#### 声明和初始化互斥锁
```c
// 静态初始化 (编译时初始化)
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// 动态初始化 (运行时初始化)
pthread_mutex_t mutex;
pthread_mutex_init(&mutex, NULL); // 第二个参数为属性，NULL表示默认
```

#### 使用互斥锁保护临界区
```c
void* thread_function(void* arg) {
    // 加锁
    pthread_mutex_lock(&mutex);
    
    // === 临界区开始 ===
    // 访问共享资源
    shared_counter++;
    // === 临界区结束 ===
    
    // 解锁
    pthread_mutex_unlock(&mutex);
    
    return NULL;
}
```

#### 销毁互斥锁
```c
pthread_mutex_destroy(&mutex);
```

### 2. 互斥锁属性设置

#### 互斥锁类型属性
```c
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);

// 设置互斥锁类型
pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK); 

pthread_mutex_t mutex;
pthread_mutex_init(&mutex, &attr);

// 使用后销毁属性
pthread_mutexattr_destroy(&attr);
```

#### 支持的互斥锁类型

| 类型 | 常量 | 描述 |
|------|------|------|
| 标准锁 | `PTHREAD_MUTEX_NORMAL` | 默认类型，无死锁检测 |
| 错误检查锁 | `PTHREAD_MUTEX_ERRORCHECK` | 提供错误检查 |
| 递归锁 | `PTHREAD_MUTEX_RECURSIVE` | 允许同一线程多次加锁 |
| 自适应锁 | `PTHREAD_MUTEX_ADAPTIVE_NP` | 自旋后阻塞 (Linux特有) |

#### 进程共享属性
```c
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);

// 设置进程间共享
pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);

pthread_mutex_t *mutex = mmap(NULL, sizeof(pthread_mutex_t), 
                            PROT_READ|PROT_WRITE,
                            MAP_SHARED|MAP_ANONYMOUS, -1, 0);
                            
pthread_mutex_init(mutex, &attr);
```

### 3. 高级锁定技术

#### 尝试加锁
```c
if (pthread_mutex_trylock(&mutex) == 0) {
    // 成功获取锁
    // ... 临界区操作 ...
    pthread_mutex_unlock(&mutex);
} else {
    // 锁已被占用，执行其他操作
    printf("Mutex is busy, doing alternative work\n");
}
```

#### 超时加锁
```c
struct timespec ts;
clock_gettime(CLOCK_REALTIME, &ts);
ts.tv_sec += 2; // 设置2秒超时

int result = pthread_mutex_timedlock(&mutex, &ts);
if (result == 0) {
    // 成功获取锁
    // ... 临界区操作 ...
    pthread_mutex_unlock(&mutex);
} else if (result == ETIMEDOUT) {
    // 超时处理
    printf("Failed to acquire mutex within 2 seconds\n");
}
```

#### 递归锁使用
```c
void recursive_function(int count) {
    pthread_mutex_lock(&recursive_mutex);
    
    if (count > 0) {
        recursive_function(count - 1);
    }
    
    pthread_mutex_unlock(&recursive_mutex);
}

// 初始化递归锁
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);
pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
pthread_mutex_init(&recursive_mutex, &attr);
```

### 4. 完整示例程序

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 5

int shared_counter = 0;
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

void* increment_counter(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < 10; i++) {
        // 加锁
        pthread_mutex_lock(&counter_mutex);
        
        // 临界区开始
        int current = shared_counter;
        printf("Thread %d: current value = %d\n", thread_id, current);
        usleep(1000); // 模拟处理时间
        shared_counter = current + 1;
        // 临界区结束
        
        // 解锁
        pthread_mutex_unlock(&counter_mutex);
        
        usleep(10000); // 等待一段时间
    }
    
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    
    // 创建线程
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, increment_counter, &thread_ids[i]) != 0) {
            perror("Failed to create thread");
            return 1;
        }
    }
    
    // 等待所有线程完成
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // 销毁互斥锁
    pthread_mutex_destroy(&counter_mutex);
    
    printf("Final counter value: %d (expected: %d)\n", 
           shared_counter, NUM_THREADS * 10);
    
    return 0;
}
```

## 三、自旋锁源码解读

### 1. 简单自旋锁实现

```C
typedef struct {
    atomic_int locked;  // 0=未锁定, 1=已锁定
} spinlock_t;

void spin_lock(spinlock_t *lock) {
    while (atomic_exchange(&lock->locked, 1) == 1) {
        // 自旋等待 (CPU忙等)
        while (atomic_load(&lock->locked) == 1);
    }
    // 获取锁后插入内存屏障
    memory_barrier();
}

void spin_unlock(spinlock_t *lock) {
    memory_barrier();
    atomic_store(&lock->locked, 0);
}
```
### 2. 自旋锁：API 与实现

```C
void lock() {
    int expected;
    do {
        // Try compare status with expected.
        // If the comparison succeeded, perform
        // an exchange.
        expected = UNLOCKED;
        asm volatile (
            "lock cmpxchgl %2, %1"
            : "+a" (expected) // Value for comparison.
                              // x86 uses eax/rax.
            : "m" (status),   // Memory location.
              "r" (LOCKED)    // Value to be written if
                              // status == expected
            : "memory", "cc"
        );
    } while (expected != UNLOCKED);
}

void unlock() {
    status = UNLOCKED;
    asm volatile("" ::: "memory");
    __sync_synchronize();
}
```

这段代码实现了一个基于自旋锁的原子累加操作。下面是对源码的详细解析：

#### 2.1. 实现原理
核心是使用x86的`lock cmpxchgl`指令实现原子比较交换操作：

```c
void lock() {
    int expected;
    do {
        expected = UNLOCKED;  // 期望锁处于未占用状态
        asm volatile (
            "lock cmpxchgl %2, %1"  // 原子比较交换指令
            : "+a" (expected)       // %0: eax寄存器(输入输出)
            : "m" (status),         // %1: 锁状态的内存地址
              "r" (LOCKED)          // %2: 要写入的锁定值
            : "memory", "cc"        // 告知编译器内存和标志寄存器被修改
        );
    } while (expected != UNLOCKED);  // 当expected值改变时说明获取锁失败
}
```

**指令执行逻辑**：
1. 比较`status`和`expected`(UNLOCKED)的值
2. 如果相等：将`LOCKED`写入`status`，同时将`expected`设为原`status`值(UNLOCKED)
3. 如果不相等：将当前`status`值(LOCKED)加载到`expected`
4. `lock`前缀确保整个操作的原子性

**循环退出条件**：
- 只有当`expected`保持UNLOCKED（说明成功获取锁）时才退出循环
- 否则持续自旋重试

#### 2.2. 解锁实现
```c
void unlock() {
    status = UNLOCKED;         // 直接修改锁状态
    asm volatile("" ::: "memory");  // 编译器级内存屏障
    __sync_synchronize();      // 硬件级内存屏障
}
```
双重内存屏障保证：
1. `asm volatile("" ::: "memory")`：防止编译器指令重排
2. `__sync_synchronize()`：插入CPU内存屏障指令（如mfence），确保：
   - 临界区内的操作在解锁前完成
   - 锁状态变更立即对所有CPU可见

#### 2.3. 线程安全累加
```c
void T_sum() {
    for (int i = 0; i < N; i++) {
        lock();    // 进入临界区
        sum++;     // 受保护的全局变量操作
        unlock();  // 退出临界区
    }
}
```
- 每个线程执行N次累加
- 通过`lock()`/`unlock()`确保`sum++`的原子性
- 实际相当于实现了一个原子加法操作

#### 2.4. 关键设计点
1. **自旋锁特性**：
   - 忙等待（busy-waiting）机制
   - 适用于短临界区场景
   - 避免上下文切换开销

2. **内存序保证**：
   - 解锁时的双重内存屏障确保：
     ```mermaid
     graph LR
     A[临界区写操作] --> B[内存屏障]
     B --> C[锁状态变更]
     C --> D[其他线程可见]
     ```

3. **x86特定实现**：
   - 依赖`cmpxchgl`指令的原子性
   - 使用`lock`前缀缓存一致性协议(MESI)
   - 隐含使用eax寄存器

## 四、sum-atomic源码解读

```c
void T_sum() {
    for (int i = 0; i < N; i++) {
        asm volatile(
            "lock addq $1, %0" : "+m"(sum)
        );
    }
}
```

这段代码实现了一个高效的多线程累加操作，直接使用x86汇编指令实现原子加法，避免了锁的开销。下面是对源码的详细解析：

### 1. 核心实现原理
```c
asm volatile(
    "lock addq $1, %0"  // 原子加1指令
    : "+m"(sum)         // 内存操作数约束
);
```
- **`lock`前缀**：确保指令执行的原子性
  - 锁定内存总线（或通过缓存一致性协议）
  - 防止其他处理器同时访问同一内存地址
- **`addq $1`**：64位整数加法（quad word）
  - `$1`表示立即数1
  - `%0`指向操作数（sum的地址）
- **内存约束`"+m"(sum)`**：
  - `+`表示读写操作数
  - `m`表示内存操作数
- **`volatile`**：禁止编译器优化（确保指令不被重排或消除）

### 2. 与锁版本的对比
原始锁版本：
```c
lock();     // 可能自旋等待
sum++;      // 非原子操作
unlock();   // 内存屏障开销
```
新版本：
```c
// 单条指令完成原子加法
lock addq $1, [sum]
```
优势：
1. **零等待**：无锁竞争时的自旋开销
2. **无上下文切换**：纯硬件原子操作
3. **更低开销**：
   - 省去锁获取/释放操作
   - 省去显式内存屏障

### 3. 关键技术点
1. **原子性保证机制**：
   ```mermaid
   graph LR
   A[CPU Core] -->|lock前缀| B[缓存一致性协议]
   B --> C[MESI状态转换]
   C --> D[原子内存访问]
   ```

2. **内存序语义**：
   - `lock`指令隐含完整内存屏障（mfence）
   - 保证：
     - 指令前的读写不会重排到lock之后
     - 指令后的读写不会重排到lock之前

3. **指令选择**：
   - `addq`：针对64位操作数（如long/long long）
   - 32位需改用`addl`：
     ```c
     "lock addl $1, %0"  // 用于32位整型
     ```

### 4. 性能优化分析
假设4线程各执行1亿次累加（N=100,000,000）：

| 方案          | 时钟周期(估计) | 潜在加速比 |
|---------------|----------------|------------|
| 自旋锁版本    | ~30周期/次     | 1x         |
| 原子指令版本  | ~1-3周期/次    | 10-30x     |

实际测试中可观察到：
- 无竞争时：原子指令快20倍以上
- 高竞争时：原子指令仍快5-10倍

### 5. 潜在问题及解决方案
1. **缓存行颠簸**：
   - 问题：多核同时写同一缓存行
   - 优化：线程本地计数+定期汇总
     ```c
     __thread int local_sum;  // 线程局部存储
     
     void T_sum() {
         for(int i=0; i<N; i++) local_sum++;
         asm volatile("lock addl %1, %0"
             : "+m"(global_sum)
             : "r"(local_sum)
         );
     }
     ```

2. **ABA问题规避**：
   - 当前场景无ABA风险（纯加法操作）
   - 若需CAS操作应使用：
     ```c
     asm volatile("lock cmpxchgq %2, %1"
         : "+a"(expected)
         : "m"(ptr), "r"(new)
         : "memory"
     );
     ```

### 6. 可移植性改进
C11标准原子操作（跨平台）：
```c
#include <stdatomic.h>
atomic_llong sum = ATOMIC_VAR_INIT(0);

void T_sum() {
    for(int i=0; i<N; i++) {
        atomic_fetch_add_explicit(&sum, 1, memory_order_relaxed);
    }
}
```
- 编译器会自动选择最优指令
- memory_order_relaxed：仅需原子性，无需严格内存序

### 结论
这种原子指令实现：
1. 性能远超基于锁的方案
2. 是操作系统/数据库等系统软件中常见的优化手段
3. 体现了"机制与策略分离"思想：
   - 硬件提供原子指令（机制）
   - 开发者选择同步策略（如本例的累加）

## 五、Futex: Fast Userspace muTexes

Futex（Fast Userspace Mutex）是Linux内核提供的一种高效同步原语，用于实现用户空间的锁和条件变量。其核心思想是**在无竞争情况下完全在用户空间操作，仅在需要等待时才进入内核**。

### **核心设计思想**
1. **混合模式同步**：
   - 无竞争时：用户空间原子操作（快速路径）
   - 有竞争时：内核辅助阻塞/唤醒（慢速路径）
2. **最小化内核交互**：
   - 避免每次锁操作都陷入内核
   - 仅当线程需要休眠或唤醒时调用系统调用
