本文主要整理Chapter 7 Sleep and wakeup的要点。

## 7.5 Sleep and wakeup

### 内容概括  
Xv6 中的 `sleep` 和 `wakeup` 机制是用于线程间主动交互的底层同步原语，主要解决**忙等待（Busy Waiting）**的低效问题，并通过协调线程的执行状态（睡眠与唤醒）实现条件同步。文中以信号量（Semaphore）为例，详细阐述了 `sleep` 和 `wakeup` 的设计动机、关键挑战（如“丢失唤醒”）及最终解决方案。

### 要点总结  

#### **1. 背景与需求**  
线程间主动交互是操作系统的基础需求（如管道读写等待、父进程等待子进程退出、磁盘 IO 完成等待）。传统的锁和调度器虽能隐藏线程行为，但无法直接解决“线程需主动等待特定事件”的场景。为此，Xv6 引入 `sleep`（让线程睡眠等待事件）和 `wakeup`（唤醒等待事件的线程）机制，属于**序列协调（Sequence Coordination）**或**条件同步（Conditional Synchronization）**原语。  

#### **2. 信号量的示例与初始问题**  
信号量（Semaphore）是典型的条件同步工具，维护一个计数（`count`），支持两个操作：  
- **V 操作（生产者）**：增加计数（`count += 1`）。  
- **P 操作（消费者）**：等待计数非零后减少计数（`count -= 1`）。  

最初的 P 操作实现（忙等待）存在效率问题：若计数长期为 0，消费者会循环检查（`while(s->count == 0);`），空转浪费 CPU 资源。  

```c
 100 structsemaphore{
 101 structspinlocklock;
 102 intcount;
 103 };
 104
 105 void
 106 V(structsemaphore *s)
 107 {
 108 acquire(&s->lock);
 109 s->count+=1;
 110 release(&s->lock);
 111 }
 112
 113 void
 114 P(structsemaphore *s)
 115 {
 116 while(s->count==0)
 117 ;
 118 acquire(&s->lock);
 119 s->count-=1;
 120 release(&s->lock);
 121 }
```

#### **3. 初步改进：引入 `sleep` 避免忙等待**  
为解决忙等待，修改 P 操作：当计数为 0 时调用 `sleep(chan)` 让线程睡眠，释放 CPU；V 操作调用 `wakeup(chan)` 唤醒等待该通道（`chan`）的线程。此设计虽避免了空转，但引入新问题——**丢失唤醒（Lost Wakeup）**：  
- 若 P 检查到 `count == 0`（行 212）后，V 操作（另一 CPU）执行 `count += 1` 并调用 `wakeup`，此时无线程睡眠（P 尚未调用 `sleep`），`wakeup` 无操作。  
- P 继续执行 `sleep` 后进入睡眠，但 V 已完成计数增加，导致 P 永远等待（即使后续 V 再次调用，可能无法匹配）。  

```c
200 void
 201 V(structsemaphore *s)
 202 {
 203 acquire(&s->lock);
 204 s->count+=1;
 205 wakeup(s);
 206 release(&s->lock);
 207 }
 208
 209 void
 210 P(structsemaphore *s)
 211 {
 212 while(s->count==0)
 213 sleep(s);
 214 acquire(&s->lock);
 215 s->count-=1;
 216 r
```

#### **4. 错误尝试：加锁保护条件检查**  
为修复丢失唤醒，尝试用锁保护 P 操作的条件检查与 `sleep` 调用（即 P 在锁内检查 `count` 并调用 `sleep`）。但此方法导致**死锁**：  
- P 持有锁时调用 `sleep`，锁未释放，V 操作因无法获取锁而阻塞，无法执行 `wakeup`。  

```c
 300 void
 301 V(structsemaphore *s)
 302 {
 303 acquire(&s->lock);
 304 s->count+=1;
 305 wakeup(s);
 306 release(&s->lock);
 307 }
 308
 309 void
 310 P(structsemaphore *s)
 311 {
 312 acquire(&s->lock);
 313 while(s->count==0)
 314 sleep(s);
 315 s->count-=1;
 316 release(&s->lock);
 317 }
```

#### **5. 正确设计：`sleep` 接口传递条件锁**  
最终解决方案修改 `sleep` 接口，要求调用者传递**条件锁**（即保护 `count` 的锁）。关键逻辑如下：  
- **P 操作**：  
  1. 获取条件锁（`acquire(&s->lock)`），确保原子性检查 `count`。  
  2. 若 `count == 0`，调用 `sleep(s, &s->lock)`：`sleep` 内部释放锁，并将线程标记为睡眠（关联通道 `s`）。  
  3. 被唤醒后，`sleep` 重新获取锁，继续执行后续逻辑（`count -= 1`）。  
- **V 操作**：  
  1. 获取条件锁，增加 `count`。  
  2. 调用 `wakeup(s)`，唤醒所有等待通道 `s` 的睡眠线程。  
  3. 释放条件锁。  

```c
400 void
 401 V(structsemaphore *s)
 402 {
 403 acquire(&s->lock);
 404 s->count+=1;
 405 wakeup(s);
 406 release(&s->lock);
 407 }
 408
 409 void
 410 P(structsemaphore *s)
 411 {
 412 acquire(&s->lock);
 413 while(s->count==0)
 414 sleep(s,&s->lock);
 415 s->count-=1;
 416 release(&s->lock);
 417 }
```

此设计的核心是：  
- **原子性保护**：条件锁确保 P 的“检查 `count`”与“进入睡眠”操作的原子性，避免 V 在中间插入执行导致唤醒丢失。  
- **无死锁**：`sleep` 内部释放锁后再睡眠，V 可正常获取锁并执行 `wakeup`，唤醒已准备好的睡眠线程。  

#### **6. 总结**  
`sleep` 和 `wakeup` 的设计通过**条件锁传递**和**原子性操作保护**，解决了忙等待、丢失唤醒和死锁问题，是 Xv6 中线程间主动交互的高效同步机制。其核心思想是：在检查等待条件（如 `count == 0`）与进入睡眠状态之间保持原子性，确保唤醒操作不会因竞争条件而失效。

## 7.6 Code: Sleep and wakeup

### 内容概括  
Xv6 的 `sleep` 和 `wakeup` 机制是实现线程间条件同步的核心原语，用于解决线程主动等待特定事件（如 I/O 完成、资源可用）的问题。其核心逻辑通过**锁机制**（进程锁 `p->lock` 和条件锁）和**进程状态管理**（`SLEEPING` 与 `RUNNABLE`）实现，确保线程在等待事件时高效释放 CPU，并在事件发生时被准确唤醒。


### 要点总结  

#### **1. `sleep` 函数的核心流程**  
`sleep(chan, lk)` 用于让当前进程进入睡眠状态，等待通道 `chan` 上的事件，关键步骤如下：  
- **获取进程锁**：首先获取当前进程的 `p->lock`（保护进程状态和上下文），确保后续操作的原子性。  
- **释放条件锁**：若调用 `sleep` 前持有条件锁 `lk`（如保护共享资源的锁），则释放 `lk`（避免阻塞其他线程修改条件）。  
- **标记睡眠状态**：将进程状态从 `RUNNING` 改为 `SLEEPING`，记录等待通道 `chan`。  
- **触发调度**：调用 `sched()` 释放 CPU，使当前进程进入睡眠，调度器后续会选择其他进程运行。  

```c
void
sleep(void *chan, struct spinlock *lk)
{
  struct proc *p = myproc();
  
  // Must acquire p->lock in order to
  // change p->state and then call sched.
  // Once we hold p->lock, we can be
  // guaranteed that we won't miss any wakeup
  // (wakeup locks p->lock),
  // so it's okay to release lk.

  acquire(&p->lock);  //DOC: sleeplock1
  release(lk);

  // Go to sleep.
  p->chan = chan;
  p->state = SLEEPING;

  sched();

  // Tidy up.
  p->chan = 0;

  // Reacquire original lock.
  release(&p->lock);
  acquire(lk);
}
```

#### **2. `wakeup` 函数的核心流程**  
`wakeup(chan)` 用于唤醒所有等待通道 `chan` 的睡眠进程，关键步骤如下：  
- **遍历进程表**：遍历全局进程表（`proc[]`），检查每个进程的状态和等待通道。  
- **获取进程锁**：对每个进程，获取其 `p->lock`（确保检查状态和修改状态的原子性）。  
- **唤醒符合条件的进程**：若进程状态为 `SLEEPING` 且等待通道匹配 `chan`，则将其状态改为 `RUNNABLE`（可运行）。  
- **调度器重新调度**：被唤醒的进程在调度器下次运行时会被选中，重新获得 CPU 执行权。  

```c
void
wakeup(void *chan)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    if(p != myproc()){
      acquire(&p->lock);
      if(p->state == SLEEPING && p->chan == chan) {
        p->state = RUNNABLE;
      }
      release(&p->lock);
    }
  }
}
```

#### **3. 锁机制的关键作用**  
`sleep` 和 `wakeup` 的正确性依赖**锁的互斥与协作**：  
- **`sleep` 持锁**：`sleep` 执行期间始终持有 `p->lock`（或条件锁 `lk`），确保在检查条件（如资源是否就绪）和标记睡眠状态的过程中，不会被其他线程（如 `wakeup`）打断。  
- **`wakeup` 持锁**：`wakeup` 遍历进程表时，对每个目标进程获取 `p->lock`，确保在检查其状态（是否 `SLEEPING`）和修改状态（设为 `RUNNABLE`）时的原子性，避免竞态条件。  


#### **4. 避免“丢失唤醒”（Lost Wakeup）的设计**  
“丢失唤醒”指：线程 `P` 检查到条件不满足（如 `count == 0`）后调用 `sleep`，但在 `sleep` 标记为 `SLEEPING` 前，线程 `V` 修改条件（如 `count += 1`）并调用 `wakeup`，导致 `wakeup` 未找到睡眠线程，`P` 最终永远等待。  

Xv6 通过以下机制避免此问题：  
- **锁的持有顺序**：`sleep` 在检查条件（如 `count == 0`）后、标记 `SLEEPING` 前，始终持有 `p->lock` 或条件锁 `lk`。  
- **`wakeup` 的原子检查**：`wakeup` 必须在持有条件锁的情况下调用（由调用者保证），确保在 `sleep` 标记 `SLEEPING` 前，条件已被修改，或 `wakeup` 检查时 `sleep` 已完成标记，从而保证唤醒不会遗漏。  


#### **5. 处理多个睡眠进程与虚假唤醒**  
- **多进程等待同一通道**：若多个进程等待同一通道（如多个线程读取同一管道），`wakeup` 会唤醒所有符合条件的进程。但只有第一个获取条件锁的进程能处理事件（如读取数据），其他进程会因无数据可处理而需重新检查条件（可能被标记为 `RUNNABLE` 后再次睡眠，即“虚假唤醒”）。  
- **循环检查条件**：因此，`sleep` 必须在循环中调用（如 `while (条件不满足) sleep(chan)`），确保进程在被虚假唤醒后重新检查条件，避免无限等待。  


#### **6. 通道的灵活性**  
通道 `chan` 可以是任意方便的数值（如内核数据结构的地址），Xv6 不强制要求特定格式。这一设计增加了灵活性，允许调用者根据具体场景选择有意义的通道标识（如管道的缓冲区地址）。  


### 总结  
Xv6 的 `sleep` 和 `wakeup` 机制通过**锁保护状态一致性**、**原子性操作避免竞态**和**循环检查处理虚假唤醒**，高效实现了线程间的条件同步。其核心思想是：在检查等待条件与进入睡眠状态之间保持锁的持有，确保唤醒操作不会因竞争条件而失效，同时通过灵活的通道设计降低使用门槛。