本文主要整理NJU-OS并发控制（信号量）章节的要点。

## 一、使用互斥锁实现计算图

```C
struct Edge {
    int from, to;
    mutex_t mutex;
} edges[] = {
    {1, 2, MUTEX_INIT()},
    {2, 3, MUTEX_INIT()},
    {2, 4, MUTEX_INIT()},
    {2, 5, MUTEX_INIT()},
    {4, 6, MUTEX_INIT()},
    {5, 6, MUTEX_INIT()},
    {4, 7, MUTEX_INIT()},
};

void T_worker(int id) {
    for (int i = 0; i < LENGTH(edges); i++) {
        struct Edge *e = &edges[i];
        if (e->to == id) {
            mutex_lock(&e->mutex);
        }
    }

    printf("Start %d\n", id);
    sleep(1);
    printf("End %d\n", id);
    sleep(1);

    for (int i = 0; i < LENGTH(edges); i++) {
        struct Edge *e = &edges[i];
        if (e->from == id) {
            // Well... This is undefined behavior
            // for POSIX threads. This is just a
            // hack for demonstration.
            mutex_unlock(&e->mutex);
        }
    }
}

int main() {
    for (int i = 0; i < LENGTH(edges); i++) {
        struct Edge *e = &edges[i];
        mutex_lock(&e->mutex);
    }

    for (int i = 0; i < N; i++) {
        create(T_worker);
    }
}
```

### 1. 锁作为依赖信号
- **入边锁**：任务获取所有入边的锁，确保所有前驱任务已完成
- **出边锁**：任务完成后释放所有出边的锁，通知后继任务

### 2. 执行顺序控制
```c
// 任务开始时：获取所有入边锁
for (each incoming edge) 
    lock(edge.mutex);

// 任务结束时：释放所有出边锁
for (each outgoing edge)
    unlock(edge.mutex);
```

### 3. 节点执行特性
- **节点1**：无入边，直接执行
- **节点2**：依赖节点1，等待边1→2解锁
- **节点3/4/5**：并行执行（都依赖节点2）
- **节点6**：依赖节点4和5，需等待两条边解锁
- **节点7**：只依赖节点4，可与节点5并行执行


## 二、使用信号量实现生产者-消费者问题

```C
sem_t fill, empty;

void T_produce() {
    while (1) {
        // Needs an empty slot for producing.
        P(&empty);

        printf("(");

        // Creates a filled slot.
        V(&fill);
    }
}

void T_consume() {
    while (1) {
        // Needs a filled slot for consuming.
        P(&fill);

        printf(")");
        
        // Creates an empty slot.
        V(&empty);
    }
}
```

1. **线程安全**：
   - 信号量操作是原子的
   - 不需要额外的互斥锁

2. **流量控制**：
   - `empty` 信号量限制生产速度
   - `fill` 信号量限制消费速度

3. **阻塞机制**：
   - 当缓冲区满时，生产者自动阻塞
   - 当缓冲区空时，消费者自动阻塞

4. **唤醒机制**：
   - `V(fill)` 唤醒阻塞的消费者
   - `V(empty)` 唤醒阻塞的生产者

通过调整缓冲区大小和线程数量，可以平衡生产者和消费者的速度差异，实现高效的系统资源利用。

## 三、信号量与条件变量对比

信号量和条件变量都是并发编程中的核心同步原语，但它们在设计理念、使用场景和实现机制上有显著差异。下面从多个维度进行详细对比：

### 1. 基本概念对比

| **特性**         | **信号量 (Semaphore)**                          | **条件变量 (Condition Variable)**               |
|------------------|------------------------------------------------|-----------------------------------------------|
| **本质**         | 带计数器的同步对象                             | 无状态的等待/通知机制                          |
| **核心操作**     | P/wait (减少计数), V/post (增加计数)           | wait (等待), signal/broadcast (通知)           |
| **状态保持**     | 内部维护计数器值                               | 不保存状态，依赖外部条件                       |
| **所有权概念**   | 无所有权，任何线程可操作                       | 必须与互斥锁配合，有锁所有权概念               |
| **标准实现**     | C++20 (`std::counting_semaphore`), POSIX       | C++11 (`std::condition_variable`), POSIX       |

### 2. 技术特性对比

| **特性**         | **信号量**                     | **条件变量**                     |
|------------------|--------------------------------|----------------------------------|
| **状态管理**     | 内置计数器                     | 依赖外部条件变量                 |
| **虚假唤醒**     | 不会发生                       | 可能发生，需要循环检查           |
| **唤醒范围**     | V操作自动唤醒等待线程          | 需显式调用notify                 |
| **多条件支持**   | 需创建多个信号量               | 单一条件变量支持多条件           |
| **跨线程释放**   | 支持（任意线程可V）            | 需在持有锁时通知                 |
| **性能**         | 通常更高效（内核优化）         | 稍慢（需维护等待队列）           |
| **可移植性**     | 标准统一                       | 不同系统实现差异较大             |
| **死锁风险**     | 低（无嵌套要求）               | 中（需正确管理锁顺序）           |

### 3. 性能与特性深度对比

| **维度**         | **信号量**                                     | **条件变量**                                   |
|------------------|-----------------------------------------------|-----------------------------------------------|
| **内存占用**     | 小（仅计数器+等待队列）                       | 中（等待队列+锁关联）                         |
| **系统调用**     | 通常1次（futex）                              | 至少2次（锁+等待）                            |
| **唤醒精度**     | 精确（计数器控制）                            | 可能过度唤醒（需重新检查条件）                |
| **嵌套支持**     | 支持                                          | 需谨慎处理锁重入                              |
| **超时处理**     | 原生支持 (try_acquire_for)                    | 需额外实现 (wait_for)                         |
| **多条件关联**   | 需创建多个信号量                              | 单一条件变量可处理多条件                      |
| **优先级反转**   | 可能发生                                      | 可通过优先级继承避免                          |
| **调试难度**     | 简单（状态可见）                              | 复杂（状态分散）                              |