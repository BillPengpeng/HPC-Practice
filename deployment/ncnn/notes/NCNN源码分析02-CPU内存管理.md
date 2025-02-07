NCNN CPU内存管理主要通过allocator.h/allocator.cpp，包括：ncnn::fastMalloc、ncnn::fastFree、Allocator、类UnlockedPoolAllocator、类PoolAllocator

## ncnn::fastMalloc

ncnn::fastMalloc对于不同类型平台采用不同形式的内存对齐、分配，额外地分配三部分内存：sizeof(void*)、NCNN_MALLOC_ALIGN、NCNN_MALLOC_OVERREAD，其中，sizeof(void*)用于存放指针，NCNN_MALLOC_ALIGN用于内存对齐，NCNN_MALLOC_OVERREAD用于防止optimized kernels大批量加载数据越界。

```
// src/allocator.h

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp>
static NCNN_FORCEINLINE _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

static NCNN_FORCEINLINE void* fastMalloc(size_t size)
{
#if _MSC_VER
    return _aligned_malloc(size, NCNN_MALLOC_ALIGN);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void* ptr = 0;
    if (posix_memalign(&ptr, NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD);
#else
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + NCNN_MALLOC_ALIGN + NCNN_MALLOC_OVERREAD);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, NCNN_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
#endif
}
```

这里的alignPtr用于指针强制类型转换、内存对齐，adata[-1]用于保存malloc实际分配内存的首地址。

## 类Allocator

类Allocator是CPU内存分类的基类，包括接口：fastMalloc、fastFree。

```
// src/allocator.h

class NCNN_EXPORT Allocator
{
public:
    virtual ~Allocator();
    virtual void* fastMalloc(size_t size) = 0;
    virtual void fastFree(void* ptr) = 0;
};
```

## 类PoolAllocatorPrivate和UnlockedPoolAllocatorPrivate

类PoolAllocatorPrivate、UnlockedPoolAllocatorPrivate分别应用于PoolAllocator、UnlockedPoolAllocator，区别是PoolAllocatorPrivate额外含有成员budgets_lock、payouts_lock，在内存申请、释放时，用于对budgets、payouts加锁、释放锁。

```
// src/allocator.cpp

class PoolAllocatorPrivate
{
public:
    Mutex budgets_lock;
    Mutex payouts_lock;
    unsigned int size_compare_ratio; // 0~256
    size_t size_drop_threshold;
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};

class UnlockedPoolAllocatorPrivate
{
public:
    unsigned int size_compare_ratio; // 0~256
    size_t size_drop_threshold;
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};
```

两个list里的元素std::pair<size_t, void*>记录内存的大小和地址。budgets里记录的是空闲的内存，payouts里记录的是已经分配的内存。

## 类PoolAllocator和UnlockedPoolAllocator 

类PoolAllocator和UnlockedPoolAllocator内存申请、释放的原理基本相同，均使用内存池管理内存，均继承自Allocator，区别是带不带锁，ncnn默认使用的是带锁的PoolAllocator。

类UnlockedPoolAllocator的fastMalloc，首先遍历budgets中所有没有被占用的内存块，若大小满足bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size，即大小符合size约束，且不会过分冗余，直接复用该内存块并返回；若无合适内存块，释放budgets中空闲内存块，最大内存块大小it_max->first低于期望size，则释放最小内存块，最小内存块大小it_min->first高于期望size，则释放最大内存块；最后利用ncnn::fastMalloc创建新内存块，置入payouts并返回。

```
// src/allocator.cpp

void* UnlockedPoolAllocator::fastMalloc(size_t size)
{
    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin(), it_max = d->budgets.begin(), it_min = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;
            d->budgets.erase(it);
            d->payouts.push_back(std::make_pair(bs, ptr));
            return ptr;
        }
        if (bs > it_max->first)
        {
            it_max = it;
        }
        if (bs < it_min->first)
        {
            it_min = it;
        }
    }
    if (d->budgets.size() >= d->size_drop_threshold)
    {
        if (it_max->first < size)
        {
            ncnn::fastFree(it_min->second);
            d->budgets.erase(it_min);
        }
        else if (it_min->first > size)
        {
            ncnn::fastFree(it_max->second);
            d->budgets.erase(it_max);
        }
    }

    // new
    void* ptr = ncnn::fastMalloc(size);
    d->payouts.push_back(std::make_pair(size, ptr));
    return ptr;
}

void* PoolAllocator::fastMalloc(size_t size)
{
    d->budgets_lock.lock();

    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin(), it_max = d->budgets.begin(), it_min = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        size_t bs = it->first;
        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;
            d->budgets.erase(it);
            d->budgets_lock.unlock();
            d->payouts_lock.lock();
            d->payouts.push_back(std::make_pair(bs, ptr));
            d->payouts_lock.unlock();
            return ptr;
        }
        if (bs < it_min->first)
        {
            it_min = it;
        }
        if (bs > it_max->first)
        {
            it_max = it;
        }
    }
    if (d->budgets.size() >= d->size_drop_threshold)
    {
        // All chunks in pool are not chosen. Then try to drop some outdated
        // chunks and return them to OS.
        if (it_max->first < size)
        {
            // Current query is asking for a chunk larger than any cached chunks.
            // Then remove the smallest one.
            ncnn::fastFree(it_min->second);
            d->budgets.erase(it_min);
        }
        else if (it_min->first > size)
        {
            // Current query is asking for a chunk smaller than any cached chunks.
            // Then remove the largest one.
            ncnn::fastFree(it_max->second);
            d->budgets.erase(it_max);
        }
    }
    d->budgets_lock.unlock();

    // new
    void* ptr = ncnn::fastMalloc(size);
    d->payouts_lock.lock();
    d->payouts.push_back(std::make_pair(size, ptr));
    d->payouts_lock.unlock();
    return ptr;
}
```

类UnlockedPoolAllocator的fastFree，遍历payouts，找到输入指针ptr对应的内存块，将其转移至budgets中变成了空闲内存块；若匹配失败，直接使用ncnn::fastFree释放内存。

```
// src/allocator.cpp

void UnlockedPoolAllocator::fastFree(void* ptr)
{
    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
    for (; it != d->payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;
            d->payouts.erase(it);
            d->budgets.push_back(std::make_pair(size, ptr));
            return;
        }
    }
    NCNN_LOGE("FATAL ERROR! unlocked pool allocator get wild %p", ptr);
    ncnn::fastFree(ptr);
}

void PoolAllocator::fastFree(void* ptr)
{
    d->payouts_lock.lock();

    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
    for (; it != d->payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;
            d->payouts.erase(it);
            d->payouts_lock.unlock();
            d->budgets_lock.lock();
            d->budgets.push_back(std::make_pair(size, ptr));
            d->budgets_lock.unlock();
            return;
        }
    }
    d->payouts_lock.unlock();
    NCNN_LOGE("FATAL ERROR! pool allocator get wild %p", ptr);
    ncnn::fastFree(ptr);
}
```

类UnlockedPoolAllocator的clear，清除掉budgets中所有的空闲内存块。

```
// src/allocator.cpp

void UnlockedPoolAllocator::clear()
{
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        void* ptr = it->second;
        ncnn::fastFree(ptr);
    }
    d->budgets.clear();
}

void PoolAllocator::clear()
{
    d->budgets_lock.lock();
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        void* ptr = it->second;
        ncnn::fastFree(ptr);
    }
    d->budgets.clear();
    d->budgets_lock.unlock();
}
```