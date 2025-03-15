
## 3 - NEMU框架选讲(1): 编译运行

### 3.1 Git，GitHub与代码仓库

[A Visual Git Reference](https://marklodato.github.io/visual-git-guide/index-en.html)
[Visualizing Git Concepts with D3](http://onlywei.github.io/explain-git-with-d3)

#### Basic Usage

![basic-usage](https://marklodato.github.io/visual-git-guide/basic-usage.svg)

The four commands above copy files between the working directory, the stage (also called the index), and the history (in the form of commits).
- git add files copies files (at their current state) to the stage.
- git commit saves a snapshot of the stage as a commit.
- git reset -- files unstages files; that is, it copies files from the latest commit to the stage. Use this command to "undo" a git add files. You can also - git reset to unstage everything.
- git checkout -- files copies files from the stage to the working directory. Use this to throw away local changes.

![basic-usage-2](https://marklodato.github.io/visual-git-guide/basic-usage-2.svg)

- git commit -a is equivalent to running git add on all filenames that existed in the latest commit, and then running git commit.
- git commit files creates a new commit containing the contents of the latest commit, plus a snapshot of files taken from the working directory. Additionally, files are copied to the stage.
- git checkout HEAD -- files copies files from the latest commit to both the stage and the working directory.

### .gitignore

基本原则：一切生成的文件都不放在Git仓库中。

```
* # 忽略一切文件
!*/ # 除了目录
!*.c # .c
!*.h # ...
!Makefile*
!.gitignore
```

### Git追踪

```makefile
define git_soft_checkout
	git checkout --detach -q && git reset --soft $(1) -q -- && git checkout $(1) -q --
endef
```

git checkout --detach -q:
git checkout --detach: 这会使 Git 进入“分离头指针”状态。在这种状态下，你可以检出（checkout）到某个特定的提交，而不是分支。这意味着，后续的提交不会被任何分支所追踪。
-q: 安静模式，减少输出的冗余信息。

git reset --soft $(1) -q --:
git reset --soft: 这个命令会移动 HEAD 到指定的状态，但不会改变工作目录或暂存区的文件。--soft 选项意味着只改变 HEAD 的位置，不改变索引或工作目录。这常常用于撤销之前的提交，但保留这些更改在工作区中。
$(1): 这是一个占位符，代表宏的第一个参数。在实际使用时，这个参数应该是一个具体的提交哈希值或分支名。
-q: 同样是安静模式。
--: 这通常用于区分命令行选项和文件名，但在这里可能是为了确保后续可能的文件名参数不会被误解为命令行选项。

git checkout $(1) -q --:
git checkout: 切换分支或检出文件。
$(1): 与之前一样，代表宏的第一个参数，表示要检出的提交或分支。
-q: 安静模式。
--: 同上。

总的来说，git_soft_checkout 宏的作用是：首先进入分离头指针状态，然后重置到指定的提交（但仍保留工作区的更改），最后再次检出到那个提交。这个过程可能用于某些特定的 Git 工作流，比如当你想要临时切换到某个特定的提交，但还想保留当前工作区的更改时。

```makefile
.git_commit:
	@echo ".git_commit"
	-@while (test -e .git/index.lock); do sleep 0.1; done;               `# wait for other git instances`
	-@git branch $(TRACER_BRANCH) -q 2>/dev/null || true                 `# create tracer branch if not existent`
	-@cp -a .git/index $(WORK_INDEX)                                     `# backup git index`
	-@$(call git_soft_checkout, $(TRACER_BRANCH))                        `# switch to tracer branch`
	-@git add . -A --ignore-errors                                       `# add files to commit`
	-@(echo "> $(MSG)" && echo $(STUID) $(STUNAME) && uname -a && uptime `# generate commit msg`) \
	                | git commit -F - $(GITFLAGS)                        `# commit changes in tracer branch`
	-@$(call git_soft_checkout, $(WORK_BRANCH))                          `# switch to work branch`
	-@mv $(WORK_INDEX) .git/index                                        `# restore git index`
```

使用 echo 命令生成提交信息，包括自定义的消息 $(MSG)、用户 ID 和用户名 $(STUID) 和 $(STUNAME)、系统信息和系统运行时间。这些信息通过管道传递给 git commit 命令作为提交信息。

### 3.2 项目构建

Makefile描述了构建目标之间的依赖关系和更新方法，本质就是一堆gcc –c（编译）和一个gcc（链接）

#### **1. Makefile 核心结构**

```
# 注释以 `#` 开头
目标(target): 依赖项(prerequisites)
<Tab>命令(recipe)   # 必须以 Tab 开头
```

```c
hello: hello.c
    gcc -o hello hello.c
```

#### **2. 基本语法与规则**
**定义变量**
```makefile
CC = gcc
CFLAGS = -Wall -O2
TARGET = myapp
SRC = main.c utils.c
OBJ = $(SRC:.c=.o)  # 将 .c 替换为 .o

$(TARGET): $(OBJ)
    $(CC) $(CFLAGS) -o $@ $^
```

**通配符与模式规则**
```makefile
# 编译所有 .c 文件到 .o
%.o: %.c
    $(CC) $(CFLAGS) -c $< -o $@
```

**自动变量**
| 变量 | 说明 |
|------|------|
| `$@` | 当前目标名（如 `myapp`） |
| `$<` | 第一个依赖项（如 `main.c`） |
| `$^` | 所有依赖项（如 `main.c utils.c`） |
| `$?` | 比目标更新的依赖项 |

#### **3. 常用指令**
**伪目标（Phony Targets）**
```makefile
.PHONY: clean install

clean:
    rm -f $(TARGET) *.o

install: $(TARGET)
    cp $(TARGET) /usr/local/bin
```

**条件判断**
```makefile
DEBUG ?= 0
ifeq ($(DEBUG), 1)
    CFLAGS += -g
else
    CFLAGS += -DNDEBUG
endif
```

**包含其他 Makefile**
```makefile
include config.mk  # 包含配置文件
```

#### **4. 示例：完整 C 项目**
```makefile
# 定义变量
CC = gcc
CFLAGS = -Wall -O2
TARGET = myapp
SRC_DIR = src
OBJ_DIR = obj
SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

# 默认目标
all: $(TARGET)

# 链接目标文件
$(TARGET): $(OBJ)
    $(CC) $(CFLAGS) -o $@ $^

# 编译 .c 到 .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
    $(CC) $(CFLAGS) -c $< -o $@

# 创建 obj 目录
$(OBJ_DIR):
    mkdir -p $@

# 清理
clean:
    rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
```

#### **5. 常用命令**
**执行目标**
```bash
make          # 执行第一个目标
make clean    # 执行 clean 目标
make -j4      # 使用 4 个线程并行构建
```

**调试选项**
```bash
make -n       # 显示但不执行命令（dry-run）
make --debug  # 打印详细调试信息
```

#### **6. 高级技巧**
**函数**
```makefile
# 获取目录下所有 .c 文件
SRC = $(wildcard src/*.c)

# 替换文件后缀
OBJ = $(patsubst %.c,%.o,$(SRC))

# 生成依赖文件
DEP = $(OBJ:.o=.d)
-include $(DEP)

%.d: %.c
    $(CC) -MM $< > $@
```

**多目标规则**
```makefile
# 同时生成多个可执行文件
APPS = app1 app2

all: $(APPS)

$(APPS): %: %.c
    $(CC) $(CFLAGS) -o $@ $<
```

#### **7. 注意事项**
1. **缩进**：命令必须使用 **Tab** 缩进（不能是空格）。
2. **变量赋值**：
   - `=`：延迟赋值（使用时展开）
   - `:=`：立即赋值
   - `?=`：条件赋值（未定义时生效）
3. **避免文件名冲突**：使用 `.PHONY` 标记非文件目标（如 `clean`）。

## 4 - NEMU 框架选讲：代码导读

### 4.1 parse_args

#### **1. 函数原型与头文件**
```c
#include <getopt.h>  // 必须包含此头文件

int getopt_long(int argc, char *const argv[],
                const char *short_options,
                const struct option *long_options,
                int *long_index);
```

- `argc`, `argv`：命令行参数（与 `main` 函数参数一致）。
- `short_options`：短选项字符串（与 `getopt` 的 `optstring` 格式相同）。
- `long_options`：长选项结构体数组（见下方定义）。
- `long_index`：输出参数，返回匹配的长选项在数组中的索引（可为 `NULL`）。

#### **2. 长选项结构体定义**
```c
struct option {
    const char *name;    // 长选项名称（如 "help"）
    int has_arg;         // 是否带参数（见下方宏）
    int *flag;           // 标志位（通常设为 NULL）
    int val;             // 返回值（对应短选项字符或自定义值）
};
```

**`has_arg` 的取值**
| 宏 | 值 | 说明 |
|----|----|------|
| `no_argument`       | 0 | 不带参数（如 `--help`） |
| `required_argument` | 1 | 必须带参数（如 `--file=name`） |
| `optional_argument` | 2 | 可选参数（如 `--log[=level]`） |

#### **3. 完整示例代码**

```c
#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int opt;
    int verbose = 0;
    char *output_file = NULL;
    char *input_file = NULL;

    // 定义短选项和长选项的映射关系
    const char *short_options = "hvo:i:";
    const struct option long_options[] = {
        {"help",    no_argument,       0, 'h'},  // --help 对应短选项 -h
        {"verbose", no_argument,       0, 'v'},  // --verbose 对应 -v
        {"output",  required_argument, 0, 'o'},  // --output=file 对应 -o file
        {"input",   required_argument, 0, 'i'},  // --input=file 对应 -i file
        {0, 0, 0, 0}  // 结束标记
    };

    // 解析命令行参数
    while ((opt = getopt_long(argc, argv, short_options, long_options, NULL)) != -1) {
        switch (opt) {
            case 'h':
                printf("用法: %s [选项] [文件...]\n", argv[0]);
                printf("选项:\n");
                printf("  -h, --help           显示帮助信息\n");
                printf("  -v, --verbose        启用详细输出\n");
                printf("  -o <文件>, --output=<文件>  指定输出文件\n");
                printf("  -i <文件>, --input=<文件>   指定输入文件\n");
                exit(0);
            case 'v':
                verbose = 1;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'i':
                input_file = optarg;
                break;
            case '?':  // 处理未知选项或参数错误
                fprintf(stderr, "错误: 未知选项或缺少参数\n");
                exit(1);
            default:
                fprintf(stderr, "未知错误\n");
                exit(1);
        }
    }

    // 输出解析结果
    printf("Verbose 模式: %s\n", verbose ? "启用" : "禁用");
    printf("输入文件: %s\n", input_file ? input_file : "未指定");
    printf("输出文件: %s\n", output_file ? output_file : "未指定");

    // 处理非选项参数（例如输入文件）
    if (optind < argc) {
        printf("其他参数:\n");
        for (int i = optind; i < argc; i++) {
            printf("  %s\n", argv[i]);
        }
    }

    return 0;
}
```

#### **4. 运行示例**
**命令行输入**
```bash
./program --input=input.txt -o output.txt --verbose file1.txt file2.txt
```

**输出结果**
```
Verbose 模式: 启用
输入文件: input.txt
输出文件: output.txt
其他参数:
  file1.txt
  file2.txt
```

### 4.2 static inline

`static inline` 的组合用于定义**仅在当前文件可见的内联函数**，既能实现函数内联优化，又能避免多文件编译时的链接冲突。  
**`static inline`** = **文件作用域** + **内联优化**。

#### **1. 关键字拆解**
**`static` 的作用**

- **限制作用域**：函数/变量仅在当前编译单元（文件）内可见，避免与其他文件的同名符号冲突。
- **示例**：
  ```c
  static void func() {}  // 只能在当前文件调用
  ```

**`inline` 的作用**
- **内联优化建议**：提示编译器将函数代码直接嵌入调用处，减少函数调用开销。
- **非强制**：编译器可能忽略此提示（如函数体过大或递归调用时）。
- **示例**：
  ```c
  inline int add(int a, int b) { return a + b; }
  ```

#### **2. `static inline` 的组合意义**
**典型使用场景**
在头文件（`.h`）中定义函数，既保证内联优化，又避免多文件包含时的重复定义错误。

**示例代码**
```c
// utils.h
#ifndef UTILS_H
#define UTILS_H

static inline int max(int a, int b) {
    return (a > b) ? a : b;
}
#endif
```

#### **3. 关键优势**
- **文件作用域**：每个包含此头文件的源文件（`.c`）会生成独立的 `max` 函数副本。
- **内联优化**：编译器尝试将函数代码直接插入调用位置，提升性能。
- **无链接冲突**：`static` 确保不同文件的同名函数互不影响。

#### **4. 对比其他场景**
**仅使用 `inline`（无 `static`）**
- **问题**：若头文件中的函数被多个源文件包含，链接时会因重复定义报错。
- **错误示例**：
  ```c
  // 错误写法：inline 函数未标记为 static
  inline int min(int a, int b) { return (a < b) ? a : b; }
  ```
  - 多个 `.c` 文件包含此头文件后，链接时出现 `multiple definition of 'min'`。

**仅使用 `static`（无 `inline`）**
- **安全但低效**：函数在每个文件内独立存在，但无法享受内联优化。
- **示例**：
  ```c
  static int min(int a, int b) { return (a < b) ? a : b; }  // 普通静态函数
  ```

### 4.3 init_monitor

**1. init_log**

```c
void init_log(const char *log_file) {
  log_fp = stdout;
  if (log_file != NULL) {
// 20241033 CONFIG_SIM_LOG情况输出二进制文件
#ifdef CONFIG_SIM_LOG 
    FILE *fp = fopen(log_file, "wb");
#else
    FILE *fp = fopen(log_file, "w");
#endif
    Assert(fp, "Can not open '%s'", log_file);
    log_fp = fp;
  }
  Log("Log is written to %s", log_file ? log_file : "stdout");
}

#define Log(format, ...) \
    _Log(ANSI_FMT("[%s:%d %s] " format, ANSI_FG_BLUE) "\n", \
        __FILE__, __LINE__, __func__, ## __VA_ARGS__)
```

- 自动记录**文件名**（`__FILE__`）、**行号**（`__LINE__`）、**函数名**（`__func__`）。
- 支持用户自定义格式（`format`）和可变参数（`...`）。
- 通过 ANSI 转义码为日志添加颜色（如蓝色），并自动换行。

```c
// 示例
Log("Error: %s", "file not found");

// 输出
[main.c:42 main] Error: file not found
```

**2. init_mem**

- 将mem设置为随机数。
- 可以暴露访问未初始化内存的UB。

```
void init_mem() {
#if   defined(CONFIG_PMEM_MALLOC)
  pmem = malloc(CONFIG_MSIZE);
  assert(pmem);
#endif
  IFDEF(CONFIG_MEM_RANDOM, memset(pmem, rand(), CONFIG_MSIZE));
  Log("physical memory area [" FMT_PADDR ", " FMT_PADDR "]", PMEM_LEFT, PMEM_RIGHT);
}
```

**3. init_isa**

- 设置状态机的初始状态，程序img拷贝进内存。
- cpu.pc默认为0x80000000。

```
void init_isa() {
  /* Load built-in image. */
  memcpy(guest_to_host(RESET_VECTOR), img, sizeof(img));

  /* Initialize this virtual computer system. */
  restart();
}

static void restart() {
  /* Set the initial program counter. */
  cpu.pc = RESET_VECTOR;

  /* The zero register is always 0. */
  cpu.gpr[0] = 0;
}
```

**4. init_sdb**

GDB简化版本，PA1要求实现。

```
void init_sdb() {
  /* Compile the regular expressions. */
  init_regex();

  /* Initialize the watchpoint pool. */
  init_wp_pool();
}
```