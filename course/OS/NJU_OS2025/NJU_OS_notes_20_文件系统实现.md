本文主要整理文件系统实现、崩溃一致性章节的要点。

## 一、FAT文件系统原理

### 1. FAT 文件系统概述

FAT（File Allocation Table）文件系统是微软设计的简单文件系统，广泛应用于各类存储设备：

```mermaid
graph TD
    A[FAT文件系统] --> B[FAT12]
    A --> C[FAT16]
    A --> D[FAT32]
    A --> E[exFAT]
    B -->|最大32MB| F[软盘]
    C -->|最大4GB| G[早期硬盘]
    D -->|最大16TB| H[U盘/SD卡]
    E -->|最大128PB| I[现代闪存]
```

### 2. 磁盘结构布局

#### 2.1. FAT 分区结构
```mermaid
graph LR
    A[引导扇区] --> B[保留区]
    B --> C[FAT1]
    C --> D[FAT2]
    D --> E[根目录区]
    E --> F[数据区]
```

#### 2.2. 关键区域说明
| **区域**       | **大小**        | **内容**                     | **重要性**        |
|---------------|----------------|-----------------------------|------------------|
| 引导扇区       | 512字节         | 启动代码+分区参数            | ★★★★★           |
| 保留扇区       | 可变(通常32扇区)| 扩展引导记录                | ★★★☆☆           |
| FAT1          | 可变            | 文件分配表主副本             | ★★★★★           |
| FAT2          | 与FAT1相同      | 文件分配表备份               | ★★★★☆           |
| 根目录区       | FAT12/16固定<br>FAT32可变 | 目录条目数组               | ★★★★☆           |
| 数据区         | 剩余空间         | 文件实际数据                | ★★★★★           |

### 3.核心数据结构

#### 3.1. 引导扇区结构 (512字节)
```c
struct BootSector {
    uint8_t  jump[3];        // 跳转指令 EB 3C 90
    char     oem[8];         // OEM标识 "MSDOS5.0"
    uint16_t bytes_per_sector; // 512
    uint8_t  sectors_per_cluster; // 1,2,4,8...
    uint16_t reserved_sectors; // 保留扇区数
    uint8_t  fat_copies;     // FAT表数量(通常2)
    uint16_t root_entries;   // 根目录条目数(FAT32=0)
    uint16_t total_sectors16;// 总扇区数(<32MB)
    uint8_t  media_descriptor; // 介质类型 F0=软盘/F8=硬盘
    uint16_t sectors_per_fat;// FAT表大小(扇区)
    // ... FAT32扩展字段
    uint32_t sectors_per_fat32;
    uint32_t root_cluster;   // 根目录起始簇
    uint16_t signature;      // 0xAA55
};
```

#### 3.2. 目录条目结构 (32字节)
```c
struct DirEntry {
    char     filename[8];    // 主文件名
    char     extension[3];   // 扩展名
    uint8_t  attributes;     // 属性字节
    uint8_t  reserved;       // 保留
    uint8_t  create_time_ms; // 创建时间(毫秒)
    uint16_t create_time;    // 创建时间
    uint16_t create_date;    // 创建日期
    uint16_t access_date;    // 访问日期
    uint16_t cluster_high;   // 起始簇号高16位
    uint16_t mod_time;       // 修改时间
    uint16_t mod_date;       // 修改日期
    uint16_t cluster_low;    // 起始簇号低16位
    uint32_t file_size;      // 文件大小(字节)
};
```

#### 3.3. 属性字节解析
| **位** | 掩码    | 含义         |
|--------|--------|--------------|
| 0      | 0x01   | 只读         |
| 1      | 0x02   | 隐藏         |
| 2      | 0x04   | 系统         |
| 3      | 0x08   | 卷标         |
| 4      | 0x10   | 子目录       |
| 5      | 0x20   | 归档         |
| 6      | 0x40   | 设备(内部用) |
| 7      | 0x80   | 未使用       |

### 4. 文件分配表(FAT)工作原理

#### 4.1. FAT 表项含义
| **值(16位)** | **含义**           |
|-------------|-------------------|
| 0x0000      | 空闲簇             |
| 0x0002-0xFFEF| 下一个簇号         |
| 0xFFF7      | 坏簇              |
| 0xFFF8-0xFFFF| 文件结束簇(EOF)    |

#### 4.2. 文件存储原理
```mermaid
graph LR
    A[目录条目] -->|起始簇号| B[FAT表]
    B -->|簇链| C[簇2]
    C -->|下一个簇| D[簇5]
    D -->|下一个簇| E[簇9]
    E -->|EOF| F[结束]
```

#### 4.3. 簇链查找算法
```python
def read_file(first_cluster):
    current = first_cluster
    data = b''
    
    while not is_eof(current):
        # 计算簇的物理位置
        sector = data_start + (current - 2) * sectors_per_cluster
        # 读取簇数据
        data += read_sectors(sector, sectors_per_cluster)
        # 查找FAT表获取下一簇
        current = fat_table[current]
    
    return data
```

### 5. FAT 文件操作详解

#### 5.1. 文件创建流程
```mermaid
sequenceDiagram
    App->>FAT: 创建文件请求
    FAT->>根目录: 查找空闲目录项
    根目录-->>FAT: 返回目录项位置
    FAT->>FAT表: 查找空闲簇链
    FAT表-->>FAT: 返回起始簇号
    FAT->>目录项: 写入文件名/簇号/大小
    FAT->>FAT表: 设置簇链EOF标记
    FAT-->>App: 返回成功
```

#### 5.2. 文件删除实现
```c
void delete_file(DirEntry* entry) {
    // 1. 标记目录项为删除
    entry->filename[0] = 0xE5; // 删除标记
    
    // 2. 释放簇链
    uint32_t cluster = GET_CLUSTER(entry);
    while (!IS_EOF(cluster)) {
        uint32_t next = fat[cluster];
        fat[cluster] = 0; // 标记为空闲
        cluster = next;
    }
}
```

#### 5.3. 碎片化问题
FAT 文件系统的簇分配策略导致严重碎片化：
```mermaid
graph LR
    File1 -->|占用| Cluster1
    File2 -->|占用| Cluster2
    File1 -->|占用| Cluster3
    File3 -->|占用| Cluster4
    File1 -->|占用| Cluster5
```

### 6. FAT 类型对比

| **特性**         | FAT12       | FAT16       | FAT32         | exFAT        |
|------------------|-------------|-------------|---------------|--------------|
| **最大卷大小**   | 32MB        | 4GB         | 16TB          | 128PB        |
| **最大文件大小** | 32MB        | 4GB         | 4GB-1         | 16EB         |
| **簇大小范围**   | 512B-4KB    | 2KB-32KB    | 4KB-32KB      | 512B-32MB    |
| **根目录位置**   | 固定        | 固定        | 数据区可变     | 数据区可变    |
| **目录项数限制** | 固定        | 固定        | 无限          | 无限         |
| **创建日期**     | 不支持       | 不支持       | 支持           | 支持         |


## 二、 File Allocation Table

### 1. FAT 表核心机制

#### 1.1. FAT 表数据结构
```c
// FAT32 表项结构 (32位)
typedef struct {
    uint32_t value : 28;   // 实际使用的28位
    uint32_t reserved : 4; // 保留位
} FAT32_Entry;

// FAT 表项含义
#define FREE_CLUSTER   0x00000000
#define RESERVED_CLUSTER 0x00000001
#define BAD_CLUSTER    0x0FFFFFF7
#define EOF_CLUSTER    0x0FFFFFFF  // FAT32结束标志
```

#### 1.2. 簇链工作原理
```mermaid
graph LR
    A[目录条目] -->|起始簇号| B[簇2]
    B -->|FAT[2]=5| C[簇5]
    C -->|FAT[5]=9| D[簇9]
    D -->|FAT[9]=EOF| E[文件结束]
```

#### 1.3. 簇号到物理地址转换
```
物理扇区 = 数据区起始扇区 + (簇号 - 2) * 每簇扇区数
```

### 2. 关键磁盘结构详解

#### 2.1. 引导扇区 (512字节)
```c
struct BootSector {
    uint8_t  jump[3];        // EB 3C 90
    char     oem[8];         // "MSDOS5.0"
    uint16_t bytes_per_sector; // 512
    uint8_t  sectors_per_cluster;
    uint16_t reserved_sectors;
    uint8_t  fat_copies;     // 通常为2
    uint16_t root_entries;   // FAT32=0
    uint16_t total_sectors16;
    uint8_t  media_descriptor; // F8=硬盘
    uint16_t sectors_per_fat;
    // FAT32扩展字段
    uint32_t sectors_per_fat32;
    uint32_t root_cluster;   // 根目录起始簇
    uint16_t signature;      // 0xAA55
};
```

#### 2.2. 目录条目 (32字节)
```c
struct DirEntry {
    char     name[8];        // 主文件名
    char     ext[3];         // 扩展名
    uint8_t  attributes;     // 属性字节
    uint8_t  reserved;
    uint8_t  create_time_ms; // 10ms单位
    uint16_t create_time;    // 小时(5)分(6)秒(5)
    uint16_t create_date;    // 年(7)月(4)日(5)
    uint16_t access_date;
    uint16_t cluster_high;   // 起始簇高16位
    uint16_t mod_time;
    uint16_t mod_date;
    uint16_t cluster_low;    // 起始簇低16位
    uint32_t file_size;      // 文件实际大小
};
```

#### 2.3. 长文件名实现 (VFAT)
```mermaid
graph LR
    A[长文件名] --> B[拆分多个目录项]
    B --> C[特殊属性0x0F]
    C --> D[校验和字段]
    D --> E[Unicode编码]
    E --> F[主目录项]
```

### 3. 文件操作核心流程

#### 3.1. 文件读取过程
```mermaid
sequenceDiagram
    App->>FAT: 读取请求
    FAT->>根目录: 查找文件名
    根目录-->>FAT: 返回起始簇号
    FAT->>FAT表: 获取簇链
    FAT表-->>FAT: 返回簇序列
    FAT->>磁盘: 读取簇数据
    磁盘-->>FAT: 返回数据
    FAT-->>App: 文件内容
```

#### 3.2. 文件写入过程
```python
def write_file(filename, data):
    # 1. 查找或创建目录项
    entry = find_or_create_entry(filename)
    
    # 2. 计算所需簇数
    clusters_needed = ceil(len(data) / cluster_size)
    
    # 3. 分配簇链
    cluster_chain = allocate_clusters(clusters_needed)
    
    # 4. 更新FAT表
    update_fat_chain(cluster_chain)
    
    # 5. 写入数据
    for i, cluster in enumerate(cluster_chain):
        write_cluster(cluster, data[i*cluster_size: (i+1)*cluster_size])
    
    # 6. 更新目录项
    entry.file_size = len(data)
    entry.mod_time = current_time()
```

#### 3.3. 文件删除机制
```c
void delete_file(DirEntry* entry) {
    // 1. 标记目录项
    entry->name[0] = 0xE5; // 删除标记
    
    // 2. 释放簇链
    uint32_t cluster = GET_CLUSTER(entry);
    while (!IS_EOF(cluster)) {
        uint32_t next = fat[cluster];
        fat[cluster] = FREE_CLUSTER;
        cluster = next;
    }
    
    // 3. 不更新文件大小!
}
```

## 三、Ext2 文件系统原理

Ext2 (Second Extended File System) 是 Linux 最经典的文件系统，尽管已被 Ext3/4 取代，但其核心设计仍是理解现代文件系统的基础：

```mermaid
graph TD
    A[Ext2 设计理念] --> B[高性能]
    A --> C[简洁性]
    A --> D[可扩展性]
    B --> E[块组结构]
    C --> F[无日志设计]
    D --> G[动态inode分配]
```

### 1. 磁盘结构布局

#### 1.1. Ext2 分区结构
```mermaid
graph LR
    A[引导块] --> B[超级块]
    B --> C[块组0]
    C --> D[块组1]
    D --> E[...]
    E --> F[块组N]
    
    subgraph 块组结构
        C --> G[超级块副本]
        C --> H[块组描述符]
        C --> I[块位图]
        C --> J[inode位图]
        C --> K[inode表]
        C --> L[数据块]
    end
```

#### 1.2. 关键区域功能
| **区域**         | **大小**       | **作用**                     | **重要性** |
|------------------|---------------|-----------------------------|-----------|
| 引导块           | 1KB           | 系统启动代码                 | ★★☆☆☆     |
| 超级块           | 1024字节      | 文件系统全局元数据           | ★★★★★     |
| 块组描述符       | 32字节/组     | 块组详细信息                 | ★★★★☆     |
| 块位图           | 1块/块组      | 数据块分配状态               | ★★★★☆     |
| inode位图        | 1块/块组      | inode分配状态                | ★★★★☆     |
| inode表          | 多个块        | 存储所有inode                | ★★★★★     |
| 数据块           | 剩余空间      | 实际文件内容                 | ★★★★★     |

### 2. 核心数据结构

#### 2.1. 超级块结构 (1024字节)
```c
struct ext2_super_block {
    __le32  s_inodes_count;        // 总inode数
    __le32  s_blocks_count;        // 总块数
    __le32  s_r_blocks_count;      // 保留块数
    __le32  s_free_blocks_count;   // 空闲块数
    __le32  s_free_inodes_count;   // 空闲inode数
    __le32  s_first_data_block;    // 首个数据块号
    __le32  s_log_block_size;      // 块大小指数 (0=1KB,1=2KB...)
    __le32  s_log_frag_size;       // 片段大小(未使用)
    __le32  s_blocks_per_group;    // 每组块数
    __le32  s_frags_per_group;     // 每组片段数
    __le32  s_inodes_per_group;    // 每组inode数
    __le32  s_mtime;               // 最后挂载时间
    __le32  s_wtime;               // 最后写入时间
    // ... 共40多个字段
};
```

#### 2.2. inode 结构 (128字节)
```c
struct ext2_inode {
    __le16  i_mode;         // 文件类型+权限
    __le16  i_uid;          // 用户ID低16位
    __le32  i_size;         // 文件大小(字节)
    __le32  i_atime;        // 最后访问时间
    __le32  i_ctime;        // inode变更时间
    __le32  i_mtime;        // 最后修改时间
    __le32  i_dtime;        // 删除时间
    __le16  i_gid;          // 组ID低16位
    __le16  i_links_count;  // 硬链接计数
    __le32  i_blocks;       // 占用512B块数
    __le32  i_flags;        // 扩展标志
    __le32  i_block[15];    // 数据块指针
    // ... 其他字段
};
```

#### 2.3. 块指针结构
```mermaid
graph TD
    A[inode] --> B[直接块0-11]
    A --> C[一级间接块]
    A --> D[二级间接块]
    A --> E[三级间接块]
    
    C --> F[间接块]
    F --> G[数据块]
    
    D --> H[二级间接块]
    H --> I[间接块]
    I --> J[数据块]
    
    E --> K[三级间接块]
    K --> L[二级间接块]
    L --> M[间接块]
    M --> N[数据块]
```

### 3. 数据存储机制

#### 3.1. 块分配策略
```python
def allocate_block(group):
    # 1. 在块组内查找空闲块
    free_block = find_free_bit(block_bitmap)
    
    # 2. 优先使用相同块组
    if free_block != -1:
        return free_block
    
    # 3. 搜索其他块组
    for other_group in groups:
        free_block = find_free_bit(other_group.block_bitmap)
        if free_block != -1:
            return free_block
    
    # 4. 空间不足
    return -1
```

#### 3.2. 文件读取流程
```mermaid
sequenceDiagram
    App->>VFS: read(file, offset, size)
    VFS->>Ext2: 调用文件操作表
    Ext2->>inode: 计算逻辑块号
    alt 小文件(直接块)
        inode-->>Ext2: 直接返回块号
    else 大文件(间接块)
        Ext2->>Indirect: 读取间接块
        Indirect-->>Ext2: 返回目标块号
    end
    Ext2->>Block: 读取数据块
    Block-->>Ext2: 返回数据
    Ext2-->>VFS: 返回数据
    VFS-->>App: 复制到用户空间
```

#### 3.3. 最大文件尺寸计算
| **块大小** | **直接块** | **一级间接** | **二级间接** | **三级间接** | **最大文件** |
|------------|------------|--------------|--------------|--------------|--------------|
| 1KB        | 12KB       | 256KB        | 64MB         | 16GB         | ~16GB        |
| 2KB        | 24KB       | 2MB          | 1GB          | 512GB        | ~512GB       |
| 4KB        | 48KB       | 8MB          | 4GB          | 4TB          | ~4TB         |

### 4. 目录结构实现

#### 4.1. 目录项结构
```c
struct ext2_dir_entry {
    __le32  inode;          // inode号
    __le16  rec_len;        // 目录项长度
    __le8   name_len;       // 文件名长度
    __le8   file_type;      // 文件类型
    char    name[255];      // 文件名
};
```

#### 4.2. 目录树示例
```
inode table:
inode 2: / (目录)
  ├─ dir_entry: "home" -> inode 100
  ├─ dir_entry: "etc" -> inode 101
  └─ dir_entry: "bin" -> inode 102

inode 100: /home (目录)
  ├─ dir_entry: "user1" -> inode 200
  └─ dir_entry: "user2" -> inode 201

inode 200: /home/user1 (目录)
  ├─ dir_entry: "file.txt" -> inode 300
  └─ dir_entry: "photo.jpg" -> inode 301
```

#### 4.3. 目录查找优化
```c
// 线性查找算法
struct ext2_dir_entry *find_entry(struct inode *dir, const char *name) {
    char *buf = kmalloc(dir->i_size);
    read_blocks(dir, buf);  // 读取所有目录块
    
    struct ext2_dir_entry *de = (struct ext2_dir_entry *)buf;
    char *limit = buf + dir->i_size;
    
    while ((char *)de < limit) {
        if (de->inode && memcmp(name, de->name, de->name_len) == 0) {
            return de;  // 找到匹配项
        }
        de = (struct ext2_dir_entry *)((char *)de + de->rec_len);
    }
    return NULL;  // 未找到
}
```

### 5. Ext2 文件操作

#### 5.1. 文件创建流程
```mermaid
sequenceDiagram
    App->>VFS: creat("/path/file")
    VFS->>Ext2: ext2_create()
    Ext2->>Ext2: 路径解析
    Ext2->>Ext2: 分配空闲inode
    Ext2->>Ext2: 初始化inode
    Ext2->>Ext2: 添加到父目录
    Ext2->>Ext2: 分配初始块
    Ext2-->>VFS: 返回成功
    VFS-->>App: 文件描述符
```

#### 5.2. 文件写入流程
```c
ssize_t ext2_write(struct file *file, const char *buf, size_t count) {
    // 1. 检查文件大小限制
    if (file->f_pos + count > MAX_FILE_SIZE)
        return -EFBIG;
    
    // 2. 分配必要的数据块
    while (needed_blocks > allocated_blocks) {
        new_block = allocate_block(inode);
        add_block_to_inode(inode, new_block);
    }
    
    // 3. 写入数据
    write_blocks(inode, file->f_pos, buf, count);
    
    // 4. 更新元数据
    inode->i_size = max(inode->i_size, file->f_pos + count);
    inode->i_mtime = CURRENT_TIME;
    
    return count;
}
```

## 四、i-bmap d-bmap inode关系

在 Linux 文件系统（如 ext2/3/4）中，**`inode`、`i_bmap`** 和 **`d_bmap`** 是管理磁盘空间（特别是数据块和 inode 块）的核心元数据结构，它们之间的关系非常密切。以下是它们的详细解释和关系：

---

### 1. **Inode**
* **职责**：存储文件或目录的 **元数据**（metadata），如文件大小、权限、所有者、时间戳、链接计数等。**不存储文件名**（文件名存在目录的数据块中）。
* **结构关键部分**：包含一组 **块指针**（Block Pointers），直接指向存储文件内容的数据块。
* **指针类型**：
    * **直接块指针（Direct Blocks）**：通常有 12 个，直接指向数据块。适用于小文件。
    * **一级间接指针（Indirect Block）**：指向一个存放**更多块指针的块**。该块称为"间接块"。
    * **二级间接指针（Double Indirect Block）**：指向一个存放**一级间接块指针的块**。
    * **三级间接指针（Triple Indirect Block）**：指向一个存放**二级间接块指针的块**。适用于超大文件。
* **核心**：`inode` 是查找文件数据的**起点**。

---

### 2. **i_bmap （Inode Bitmap 索引节点位图）**
* **职责**：**标记 inode 的使用状态**。文件系统初始化时创建，每个bit位对应磁盘上一个`inode`。
* **工作原理**：
    * `0`：代表对应的 `inode` 是**空闲**的。
    * `1`：代表对应的 `inode` 是**已使用**的（存储了有效元数据）。
* **位置**：通常存储在`Block Group Descriptor Table`（块组描述符表）指定的块中。
* **作用**：分配或删除文件/目录时，文件系统通过`i_bmap`快速查找空闲的`inode`并更新其状态。

---

### 3. **d_bmap （Block Bitmap 块位图 / 数据块位图）**
* **职责**：**标记数据块（Data Block）的使用状态**。每个bit位对应磁盘上一个**数据块**。
* **工作原理**：
    * `0`：代表对应的**数据块**是**空闲**的。
    * `1`：代表对应的**数据块**是**已使用**的（存储了文件内容或元数据）。
* **位置**：也存储在`Block Group Descriptor Table`指定的块中。
* **作用**：分配空间给文件或目录时，文件系统通过`d_bmap`快速查找连续或分散的**空闲数据块**并更新其状态。

---

### 🔗 三者关系 (协同工作流程)

1. **创建文件：**
   * **找空闲 inode**：查询`i_bmap`，找到一个状态为`0`（空闲）的 bit，将对应的 `inode` 分配给新文件，并将 `i_bmap` 中该 bit 置 `1`。
   * **初始化 inode**：在选中的 `inode` 结构中写入文件元数据（大小、权限、时间戳等）。
   * **分配存储空间 (数据块)：**
     * 计算所需数据块数量。
     * **找空闲数据块**：查询`d_bmap`，找到所需数量的空闲数据块（状态为 `0`）。
     * **更新 d_bmap**：将这些块的 bit 置 `1`。
     * **更新 inode**：将分配到的数据块的地址填入 `inode` 的**块指针数组**中（直接指针、间接指针等）。
   * **更新目录**：在父目录的数据块中添加新条目（文件名 + 新分配的 `inode` 号）。

2. **读取文件：**
   * **定位 inode**：通过目录项找到文件的 `inode` 号。
   * **读取 inode**：从磁盘读取该 `inode` 结构。
   * **找到数据块**：根据 `inode` 中的**块指针数组**（直接块、间接块）信息，找到文件内容存储在哪些数据块上。
   * **读取数据**：从这些数据块中读取内容。

3. **删除文件：**
   * **删除目录项**：从父目录的数据块中移除该文件的条目。
   * **释放 inode**：将该文件的 `inode` 在 `i_bmap` 中的对应 bit **置 `0`**，标记为空闲。
   * **释放数据块**：将该文件占用的**所有数据块**在 `d_bmap` 中的对应 bits **置 `0`**，标记为空闲。
   * **清除 inode**：通常不会立即擦除磁盘上的 inode 结构数据，但会将其标记为无效（链接数减为0，时间戳更新）。

4. **扩展文件 (追加写入)：**
   * **找新空闲数据块**：查询`d_bmap`，找到新需要的空闲块。
   * **更新 d_bmap**：将这些新块的 bit 置 `1`。
   * **更新 inode**：
     * 将新的块地址添加到 `inode` 的指针数组中（可能需要分配新的间接块）。
     * 更新文件大小等信息。

---

### 📌 关键点总结

| 结构       | 作用                                     | 管理对象     | 查找依据                     | 状态表示           |
|------------|------------------------------------------|--------------|------------------------------|--------------------|
| **`inode`**   | 存储文件元数据和指向文件数据的**指针** | 一个特定文件 | `inode` 号 (通过目录查找)   | N/A                |
| **`i_bmap`**  | 记录 **inode 是否空闲/已用**            | 所有 inode   | Bit 位顺序 (按 inode 号索引) | `0`=空闲, `1`=已用 |
| **`d_bmap`**  | 记录 **数据块是否空闲/已用**          | 所有数据块   | Bit 位顺序 (按数据块号索引) | `0`=空闲, `1`=已用 |

* **`inode` 是文件和数据的联系枢纽**。它通过内部指针指向实际存储文件内容的数据块。
* **`i_bmap` 管理 `inode` 资源池**。它确保文件系统能快速找到可用的 `inode` 来创建新文件/目录。
* **`d_bmap` 管理磁盘空间资源池**。它确保文件系统能快速找到可用的数据块来存储文件内容和元数据（如目录项）。
* 文件系统通过 **块组描述符（Block Group Descriptor）** 来定位一个块组内的 `inode table`, `i_bmap`, `d_bmap` 以及数据块区的位置。

### 📊 关联流程图

```mermaid
graph LR
    A[用户操作<br>(创建/读/写/删文件)] --> B((文件系统代码))

    subgraph 元数据管理
        B --> C[查找目录项<br>获取 inode 号]
        C --> D[读取 inode 结构]
        D --> E{需要空间操作?<br>(创建/扩展/删除)}
        E -- Yes --> F[查 d_bmap<br>找空闲数据块]
        F --> G[更新 d_bmap]
        G --> H[更新 inode 指针]
        E -- 创建/删除 --> I[查 i_bmap<br>分配/释放 inode]
        I --> J[更新 i_bmap]
    end

    subgraph 磁盘结构
        ITable[Inode Table] --> |存储| D
        dBMAP[Data Block Bitmap<br>(d_bmap)] --> |管理| DataBlocks[数据块]
        iBMAP[Inode Bitmap<br>(i_bmap)] --> |管理| ITable
    end

    H --> K[写入文件数据]
    K --> DataBlocks
```
理解 `inode`, `i_bmap`, `d_bmap` 之间的关系，是深入理解 Linux/Unix 文件系统磁盘空间管理和性能调优的基础。


## 五、崩溃一致性（Crash Consistency）

### 1. 核心概念与问题定义

**崩溃一致性**是指当计算机系统在文件操作过程中发生意外崩溃（如断电、内核错误等）后，文件系统能够保持数据结构完整性和逻辑正确性的能力。

```mermaid
graph TD
    A[正常文件操作] --> B[系统崩溃]
    B --> C{崩溃后状态}
    C -->|一致性保持| D[数据完整]
    C -->|一致性破坏| E[数据损坏]
```

#### 崩溃一致性问题本质
文件操作通常涉及**多个相关但独立的磁盘写操作**，而崩溃可能发生在这些操作之间，导致文件系统处于不一致状态：

| **操作类型** | **涉及磁盘写** | **崩溃风险** |
|-------------|---------------|-------------|
| 文件创建 | 1. 目录项更新<br>2. inode分配<br>3. 位图更新 | 可能只完成部分操作 |
| 文件写入 | 1. 数据块写入<br>2. inode更新<br>3. 位图更新 | 数据与元数据不一致 |
| 文件删除 | 1. 目录项标记<br>2. inode释放<br>3. 位图更新 | 空间泄漏或数据残留 |

### 2. 崩溃场景案例分析

#### 2.1. 文件创建不一致
```mermaid
sequenceDiagram
    participant FS as 文件系统
    participant Disk as 磁盘
    
    FS->>Disk: 1. 更新目录项
    Note right of Disk: 目录已记录新文件
    FS->>Disk: 2. 分配inode
    Note right of Disk: 系统崩溃！
    FS->>Disk: 3. 更新位图(未执行)
    
    结果： 目录指向不存在的inode
```

#### 2.2. 数据写入不一致
```mermaid
graph LR
    A[写入数据块] --> B[更新inode大小]
    
    崩溃点1 --> C[数据写入但未记录]
    崩溃点2 --> D[大小更新但数据丢失]
```

#### 2.3. 元数据更新不一致
```c
// 典型文件追加操作步骤：
1. 写新数据块
2. 更新文件大小(inode)
3. 更新修改时间(inode)
4. 更新位图(新块分配)
5. 更新目录项(文件大小变化)
```
**崩溃风险**：任何步骤中断都会导致元数据与实际数据不匹配

### 3. 解决方案架构

#### 3.1. 文件系统检查工具(fsck)
```mermaid
graph TB
    A[系统启动] --> B[运行fsck]
    B --> C[扫描整个文件系统]
    C --> D[检查元数据一致性]
    D -->|发现问题| E[尝试修复]
    D -->|无问题| F[挂载文件系统]
    
    subgraph 检查项
        D1[inode链接计数]
        D2[块分配状态]
        D3[目录项有效性]
        D4[超级块完整性]
    end
```

**缺点**：全盘扫描耗时长（TB级磁盘需数小时）

#### 3.2. 日志机制(Journaling)
```mermaid
graph LR
    A[操作开始] --> B[写日志记录]
    B --> C[执行实际写入]
    C --> D[删除日志记录]
    
    style B fill:#e6f7ff,stroke:#1890ff
    style D fill:#f6ffed,stroke:#52c41a
```

**日志工作流程**：
```mermaid
sequenceDiagram
    participant App
    participant FS
    participant Journal
    participant Disk
    
    App->>FS: 文件操作请求
    FS->>Journal: 1. 写入事务开始标记
    FS->>Journal: 2. 记录元数据变更
    FS->>Journal: 3. 写入事务提交标记
    FS->>Disk: 4. 执行实际写入
    FS->>Journal: 5. 清除日志记录
```

#### 3.3. 写时复制(Copy-on-Write)
```mermaid
graph LR
    A[修改请求] --> B[写入新位置]
    B --> C[更新指针]
    C --> D[释放旧空间]
    
    崩溃点 --> E[旧数据保持完整]
```
