本文主要整理Chapter 7 Pipes / Wait, exit, and kill / Process Locking的要点。

## 7.7 Code: Pipes

### 内容概括  
Xv6 中管道（pipe）的实现是生产者-消费者同步问题的典型案例，通过 `sleep` 和 `wakeup` 机制协调读（`piperead`）写（`pipewrite`）操作的同步。管道核心由 `struct pipe` 结构体表示，包含锁、环形数据缓冲区及读写字节计数器（`nread`、`nwrite`）。读写操作通过锁保护共享状态，并利用 `sleep`（等待缓冲区空间/数据）和 `wakeup`（通知对方操作完成）实现高效同步。


### 要点总结  

#### **1. 管道的数据结构**  
管道由 `struct pipe` 表示，核心字段包括：  
- **锁（`lock`）**：保护缓冲区、计数器及内部不变式（如缓冲区满/空的判断条件）。  
- **环形缓冲区（`buf`）**：固定大小（`PIPESIZE`），支持循环写入（`nwrite` 到达末尾后回到起始位置）。  
- **读写字节计数器（`nread`、`nwrite`）**：记录总读取/写入字节数（不循环），用于判断缓冲区状态：  
  - 空：`nwrite == nread`；  
  - 满：`nwrite == nread + PIPESIZE`（通过计数器差值而非模运算判断）。  

```c
struct pipe {
  struct spinlock lock;
  char data[PIPESIZE];
  uint nread;     // number of bytes read
  uint nwrite;    // number of bytes written
  int readopen;   // read fd is still open
  int writeopen;  // write fd is still open
};
```

#### **2. 同步机制的核心逻辑**  
管道的读写操作需严格同步，避免缓冲区溢出或读空数据，依赖 `sleep` 和 `wakeup` 实现：  

##### **(1) `pipewrite`（写操作）**  
- **获取锁**：首先获取管道锁（`pi->lock`），确保对缓冲区和计数器的原子访问。  
- **检查缓冲区状态**：循环写入数据时，若缓冲区已满（`nwrite == nread + PIPESIZE`），则：  
  - 调用 `wakeup(&pi->nread)`：唤醒等待读取的进程（通知缓冲区有新数据）。  
  - 调用 `sleep(&pi->nwrite, &pi->lock)`：释放锁并进入睡眠（`sleep` 自动释放锁，避免阻塞其他操作），等待读者读取数据释放空间。  
- **写入数据**：若缓冲区未满，将数据逐字节写入 `buf[nwrite % PIPESIZE]`，更新 `nwrite`。  

```c
int
pipewrite(struct pipe *pi, uint64 addr, int n)
{
  int i = 0;
  struct proc *pr = myproc();

  acquire(&pi->lock);
  while(i < n){
    // 检查管道是否关闭或进程是否被杀死
    if(pi->readopen == 0 || killed(pr)){
      release(&pi->lock);
      return -1;
    }
    if(pi->nwrite == pi->nread + PIPESIZE){ //DOC: pipewrite-full
      wakeup(&pi->nread);  // 唤醒等待读取的进程（通知有空间）
      sleep(&pi->nwrite, &pi->lock);
    } else {
      char ch;
      if(copyin(pr->pagetable, &ch, addr + i, 1) == -1)
        break;
      pi->data[pi->nwrite++ % PIPESIZE] = ch; // 环形写入
      i++;
    }
  }
  wakeup(&pi->nread);
  release(&pi->lock);

  return i;
}
```

##### **(2) `piperead`（读操作）**  
- **获取锁**：同样先获取管道锁，确保原子访问共享状态。  
- **检查缓冲区状态**：若缓冲区非空（`nread != nwrite`），则：  
  - 循环读取数据（最多 `n` 字节），从 `buf[nread % PIPESIZE]` 读取，更新 `nread`。  
  - 调用 `wakeup(&pi->nwrite)`：唤醒等待写入的进程（通知缓冲区有空间可用）。  
- **处理空缓冲区**：若缓冲区为空（`nread == nwrite`），则 `sleep` 会在 `pipewrite` 写入数据并调用 `wakeup` 后被唤醒，重新检查条件后继续读取。  

```c
int
piperead(struct pipe *pi, uint64 addr, int n)
{
  int i;
  struct proc *pr = myproc();
  char ch;

  acquire(&pi->lock);
  while(pi->nread == pi->nwrite && pi->writeopen){  //DOC: pipe-empty
    if(killed(pr)){
      release(&pi->lock);
      return -1;
    }
    sleep(&pi->nread, &pi->lock); //DOC: piperead-sleep
  }
  for(i = 0; i < n; i++){  //DOC: piperead-copy
    if(pi->nread == pi->nwrite)
      break;
    ch = pi->data[pi->nread++ % PIPESIZE];
    if(copyout(pr->pagetable, addr + i, &ch, 1) == -1)
      break;
  }
  wakeup(&pi->nwrite);  //DOC: piperead-wakeup
  release(&pi->lock);
  return i;
}
```

#### **3. 关键设计细节**  

##### **(1) 环形缓冲区的索引计算**  
由于 `nread` 和 `nwrite` 是累加计数器（不循环），实际访问缓冲区时需通过 `nread % PIPESIZE` 和 `nwrite % PIPESIZE` 计算偏移量，确保环形结构的正确性。  


##### **(2) 睡眠通道的分离**  
读写操作使用不同的睡眠通道：  
- 写操作睡眠在 `&pi->nwrite`（等待读者释放空间）；  
- 读操作睡眠在 `&pi->nread`（等待写者写入数据）。  
此设计避免读写操作的睡眠相互干扰，提升多读者/写者场景下的效率（仅需唤醒目标方向的进程）。  


##### **(3) 循环检查条件的必要性**  
`sleep` 始终在循环中调用（如 `while (缓冲区满) sleep(...)`），原因有二：  
- **虚假唤醒**：即使无明确 `wakeup`，进程也可能被内核错误唤醒（概率低但需容忍）；  
- **多进程竞争**：若多个写者同时被唤醒，仅第一个能成功写入（后续写者发现缓冲区仍满，重新睡眠）。  


##### **(4) 锁与 `sleep`/`wakeup` 的协作**  
- `sleep` 在释放 CPU 前会释放管道锁（通过 `sched()`），避免阻塞其他操作（如 `piperead` 获取锁）；  
- `wakeup` 遍历进程表时需获取目标进程的锁（由 `sleep` 持有），确保状态检查和修改的原子性；  
- 锁的保护范围覆盖整个读写流程，确保 `nread`、`nwrite` 和缓冲区的一致性。  


### 总结  
Xv6 管道的实现通过**环形缓冲区**管理数据，利用**锁**保护共享状态，结合 `sleep`（等待条件）和 `wakeup`（通知条件满足）实现生产者-消费者的精确同步。其核心设计（如分离睡眠通道、循环检查条件）确保了多进程并发读写时的正确性和效率，是操作系统同步原语（`sleep`/`wakeup`）的典型应用场景。

## 7.8 Code: Wait, exit, and kill

### 内容概况  
Xv6 中的 `wait`、`exit` 和 `kill` 函数是进程管理的核心原语，分别用于**父进程等待子进程退出**、**进程主动终止自身**和**强制终止其他进程**。这些函数通过**锁机制**（如 `wait_lock` 和进程锁 `p->lock`）、**状态标记**（如 `ZOMBIE` 状态）和 `sleep`/`wakeup` 协作，确保进程终止与清理的安全性和正确性，避免竞态条件和死锁。


### 要点总结  

#### **1. `wait` 函数：父进程等待子进程退出**  
`wait` 的核心职责是**阻塞父进程，直到其子进程退出**，并清理子进程资源。关键逻辑如下：  

- **锁保护**：  
  首先获取全局 `wait_lock`（条件锁），确保在扫描进程表和检查子进程状态时，不会被其他进程（如子进程退出）的 `wakeup` 干扰，避免丢失唤醒。  

- **扫描进程表**：  
  遍历全局进程表（`proc[]`），查找当前进程的子进程（通过 `pp` 指针关联）。  

- **处理僵尸进程（ZOMBIE）**：  
  若找到状态为 `ZOMBIE` 的子进程（已退出但未被父进程清理）：  
  - 释放子进程的资源（如内存、文件描述符）。  
  - 将子进程的 `proc` 结构体标记为未使用（`UNUSED`）。  
  - 将子进程的退出状态复制到父进程提供的地址（若非零）。  
  - 返回子进程的 PID，父进程继续执行。  

- **无僵尸进程时等待**：  
  若所有子进程均未退出（非 `ZOMBIE` 状态），调用 `sleep(&wait_lock)` 释放 `wait_lock` 并进入睡眠，等待子进程退出后 `wakeup` 唤醒。  

- **锁顺序**：  
  `wait` 持锁顺序为 `wait_lock` → 子进程的 `p->lock`（若需要操作子进程），避免死锁（与 `exit` 函数的锁顺序一致）。  

```c
int
wait(uint64 addr)
{
  struct proc *pp;
  int havekids, pid;
  struct proc *p = myproc();

  acquire(&wait_lock);

  for(;;){
    // Scan through table looking for exited children.
    havekids = 0;
    for(pp = proc; pp < &proc[NPROC]; pp++){
      if(pp->parent == p){
        // make sure the child isn't still in exit() or swtch().
        acquire(&pp->lock);

        havekids = 1;
        if(pp->state == ZOMBIE){
          // Found one.
          pid = pp->pid;
          if(addr != 0 && copyout(p->pagetable, addr, (char *)&pp->xstate,
                                  sizeof(pp->xstate)) < 0) {
            release(&pp->lock);
            release(&wait_lock);
            return -1;
          }
          freeproc(pp);
          release(&pp->lock);
          release(&wait_lock);
          return pid;
        }
        release(&pp->lock);
      }
    }

    // No point waiting if we don't have any children.
    if(!havekids || killed(p)){
      release(&wait_lock);
      return -1;
    }
    
    // Wait for a child to exit.
    sleep(p, &wait_lock);  //DOC: wait-sleep
  }
}
```

#### **2. `exit` 函数：进程主动终止自身**  
`exit` 处理进程的退出流程，包括资源释放、状态标记和父进程通知。关键逻辑如下：  

- **资源释放与状态标记**：  
  - 记录退出状态（`exitstatus`）。  
  - 释放部分资源（如文件描述符、内存映射）。  

- **重新父进程（Reparent）**：  
  调用 `reparent` 将当前进程的子进程转移给 `init` 进程（`init` 进程持续调用 `wait`，确保所有孤儿进程被清理）。  

- **唤醒父进程**：  
  若父进程正在 `wait`（睡眠等待子进程退出），调用 `wakeup(p->parent)` 唤醒父进程，使其能及时清理当前进程。  

- **标记为僵尸（ZOMBIE）**：  
  将当前进程状态设为 `ZOMBIE`（表示已退出但未被父进程清理），并永久让出 CPU（`sched()`）。  

- **锁保护**：  
  整个流程持有 `wait_lock`（条件锁）和当前进程的 `p->lock`（保护状态修改），确保：  
  - `wakeup(p->parent)` 不会被父进程的 `wait` 竞争丢失。  
  - 父进程的 `wait` 扫描进程表时，不会在 `exit` 完成状态标记前看到 `ZOMBIE` 状态（避免提前清理）。  

```c
void
exit(int status)
{
  struct proc *p = myproc();

  // ...

  begin_op();
  iput(p->cwd);
  end_op();
  p->cwd = 0;

  acquire(&wait_lock);

  // Give any children to init.
  reparent(p);

  // Parent might be sleeping in wait().
  wakeup(p->parent);
  
  acquire(&p->lock);

  p->xstate = status;
  p->state = ZOMBIE;

  release(&wait_lock);

  // Jump into the scheduler, never to return.
  sched();
  panic("zombie exit");
}
```

#### **3. `kill` 函数：强制终止其他进程**  
`kill` 用于强制终止目标进程（如子进程或无关进程），通过设置标志位间接触发退出。关键逻辑如下：  

- **设置终止标志**：  
  直接修改目标进程的 `p->killed` 标志（无需直接销毁进程），表示该进程应终止。  

- **唤醒目标进程**：  
  若目标进程正在睡眠（`SLEEPING` 状态），调用 `wakeup` 唤醒它，使其尽快进入内核态检查 `killed` 标志。  

- **进程自行退出**：  
  目标进程被唤醒后（或在用户态执行系统调用时），会检查 `p->killed` 标志（如通过 `killed()` 函数）。若标志被设置：  
  - 若在用户态，继续执行当前系统调用或响应中断，最终通过 `usertrap` 进入内核态。  
  - 若在 kernel 态（如 `sleep` 返回后），调用 `exit` 清理资源并终止。  

- **安全设计**：  
  `kill` 不直接终止进程，而是通过标志位让进程自行退出，避免在敏感操作（如磁盘 I/O）中途终止导致数据不一致。例如，`virtio` 驱动在磁盘操作中不检查 `killed`，确保操作原子性完成后才退出。  

```c
int
kill(int pid)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++){
    acquire(&p->lock);
    if(p->pid == pid){
      p->killed = 1;
      if(p->state == SLEEPING){
        // Wake process from sleep().
        p->state = RUNNABLE;
      }
      release(&p->lock);
      return 0;
    }
    release(&p->lock);
  }
  return -1;
}
```

#### **4. 关键同步机制与竞态避免**  
- **锁顺序一致性**：  
  `wait` 和 `exit` 均遵循 `wait_lock` → `p->lock` 的持锁顺序，避免死锁（如 `wait` 持 `wait_lock` 等待 `p->lock`，而 `exit` 持 `p->lock` 等待 `wait_lock`）。  

- **状态转换的原子性**：  
  `exit` 在设置 `ZOMBIE` 状态前唤醒父进程，但父进程的 `wait` 扫描进程表时，需等待 `scheduler` 释放当前进程的 `p->lock` 后才能访问，确保不会在 `exit` 完成状态标记前误判。  

- **`killed` 标志的延迟检查**：  
  `sleep` 循环中不直接检查 `killed`（避免打断原子操作），但通过 `while` 循环重新测试条件（如管道读写的 `copyin` 失败或 `killed` 标志），确保进程在安全点退出。  


#### **5. 总结**  
`wait`、`exit` 和 `kill` 共同构成了 Xv6 进程生命周期管理的核心机制：  
- `wait` 通过锁和 `sleep` 实现父进程对子进程退出的阻塞等待。  
- `exit` 通过状态标记和锁保护，确保进程退出时的资源清理与父进程通知。  
- `kill` 通过标志位间接终止进程，避免强制中断敏感操作，确保系统稳定性。  

这些设计通过严格的锁协议、状态管理和延迟检查，解决了进程终止与清理中的竞态条件问题，是操作系统进程管理的经典实践。

## 7.9 Process Locking

### 内容概括  
Xv6 中的进程锁（`p->lock`）是保护进程结构体（`struct proc`）核心字段的最复杂锁机制，用于协调多进程/线程对进程状态的访问，确保原子性和正确性。它保护了进程状态（`p->state`）、通道（`p->chan`）、终止标志（`p->killed`）等关键字段，并参与了进程创建、销毁、调度、父子交互（如 `wait`/`exit`）及强制终止（`kill`）等核心操作，是 Xv6 进程管理的核心同步原语之一。


### 要点总结  

#### **1. `p->lock` 保护的核心字段**  
`p->lock` 用于保护以下可能被其他进程或调度器线程访问的进程字段：  
- **状态相关**：`p->state`（进程状态，如 `RUNNING`、`SLEEPING`、`ZOMBIE`）、`p->chan`（睡眠通道）。  
- **终止相关**：`p->killed`（进程终止标志）。  
- **调度相关**：`p->xstate`（调度器私有状态）、`p->pid`（进程 ID）。  


#### **2. `p->lock` 的核心功能**  
`p->lock` 的设计目标是解决多进程/线程并发访问进程状态时的竞态条件，具体功能包括：  

##### **(1) 进程创建与销毁的原子性**  
- 防止分配 `proc[]` 槽位时的竞态条件（如多个进程同时申请新进程槽位）。  
- 隐藏进程在创建或销毁过程中的中间状态（如 `ZOMBIE` 状态未完全生效时），避免其他进程误判。  


##### **(2) 协调父子进程的 `wait` 操作**  
- 防止父进程的 `wait` 收集未完全退出的 `ZOMBIE` 进程（即进程已设为 `ZOMBIE` 但未释放 CPU 前，父进程无法通过 `wait` 感知）。  
- 确保退出进程在 `wait` 唤醒父进程并释放 CPU 后，父进程才能正确清理其资源。  


##### **(3) 调度器决策的原子性**  
- 防止其他 CPU 的调度器在进程设为 `RUNNABLE` 但未完成 `swtch`（上下文切换）前，错误地选择该进程运行。  
- 确保同一时间只有一个 CPU 的调度器决定运行某个 `RUNNABLE` 进程（避免多 CPU 竞争同一进程）。  


##### **(4) 中断与 `sleep` 的安全交互**  
- 防止定时器中断导致进程在 `swtch`（上下文切换）过程中被强制 `yield`（让出 CPU），避免切换失败或状态不一致。  


##### **(5) 配合条件锁防止唤醒丢失**  
- 与条件锁（如 `wait_lock`）协作，确保 `wakeup` 不会遗漏调用 `sleep` 但未完成 `yield` 的进程（如进程在 `sleep` 返回前被 `wakeup` 标记）。  


##### **(6) `kill` 操作的原子性保护**  
- 防止 `kill` 操作中，受害进程在 `kill` 检查 `p->pid` 后、设置 `p->killed` 前退出或被重新分配（导致 `kill` 目标失效）。  
- 确保 `kill` 对 `p->state` 的检查（如是否 `RUNNING`）和修改（设为 `ZOMBIE`）是原子操作。  


#### **3. `p->lock` 与 `wait_lock` 的区别**  
- **保护对象不同**：`p->lock` 是**每个进程的私有锁**，保护进程自身的状态字段（如 `p->state`、`p->killed`）；`wait_lock` 是**全局锁**，保护父进程与子进程的 `wait` 协作逻辑（如 `p->parent` 字段的访问）。  
- **功能侧重不同**：`p->lock` 聚焦进程状态的原子性修改；`wait_lock` 聚焦父子进程间的事件通知（如子进程退出时唤醒父进程的 `wait`）。  


#### **4. 总结**  
`p->lock` 是 Xv6 进程管理的核心锁机制，通过保护进程关键状态字段，协调多进程/线程的并发操作，确保进程创建、销毁、调度、父子交互及强制终止的正确性。其设计覆盖了从进程生命周期到调度策略的全场景，是操作系统实现多任务并发的重要基础。