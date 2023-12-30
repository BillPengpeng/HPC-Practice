类Mat是NCNN数据表示基础数据类型。

## Mat成员变量

类Mat的成员变量如下，包括：
1. data: 分配内存首地址
2. refcount: 引用计数
3. allocator: 内存分配类
4. dims: 数据维度，最多允许4维
5. c/d/h/w：具体每个维度的值，其中d是指的depth，通常在3d卷积中会用到
6. elempack: 有多少个数据打包在一起，通常用于SIMD的场景，可简单认为一个SIMD的寄存器里存了几个元素
7. elemsize: 打包在一起的数据占的字节数
8. cstep: channel step, 即每个channel对应的字节数

```
// src/mat.h

// the three dimension matrix
class NCNN_EXPORT Mat
{
public:
// ...

// pointer to the data
void* data;

// pointer to the reference counter
// when points to user-allocated data, the pointer is NULL
int* refcount;

// element size in bytes
// 4 = float32/int32
// 2 = float16
// 1 = int8/uint8
// 0 = empty
size_t elemsize;

// packed count inside element
// c/1-d-h-w-1  c/1-h-w-1  h/1-w-1  w/1-1  scalar
// c/4-d-h-w-4  c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
// c/8-d-h-w-8  c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
int elempack;

// the allocator
Allocator* allocator;

// the dimension rank
int dims;

int w;
int h;
int d;
int c;

size_t cstep;
};
```

## Mat构造函数

类Mat的构造函数如下，针对不同成员变量传参，包括：
1. empty情况
2. 不指定elempack，采用默认值1
3. 指定elempack
4. 拷贝构造函数
5. 不指定elempack，采用默认值1，并传递外部数据指针data，不做内存分配
6. 指定elempack，并传递外部数据指针data，不做内存分配

```
// src/mat.h

// empty
Mat();
// vec
Mat(int w, size_t elemsize = 4u, Allocator* allocator = 0);
// image
Mat(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
// dim
Mat(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
// cube
Mat(int w, int h, int d, int c, size_t elemsize = 4u, Allocator* allocator = 0);
// packed vec
Mat(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
// packed image
Mat(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
// packed dim
Mat(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
// packed cube
Mat(int w, int h, int d, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
// copy
Mat(const Mat& m);
// external vec
Mat(int w, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
// external image
Mat(int w, int h, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
// external dim
Mat(int w, int h, int c, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
// external cube
Mat(int w, int h, int d, int c, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
// external packed vec
Mat(int w, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
// external packed image
Mat(int w, int h, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
// external packed dim
Mat(int w, int h, int c, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
// external packed cube
Mat(int w, int h, int d, int c, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
```

典型的Mat(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0)实现如下，首先，完成成员变量elemsize、elempack、allocator、dims、w、h、d、c赋值；然后，元素长度按16倍对齐计算cstep，进一步由cstep计算totalsize；最后，由totalsize执行内存分配，给指针refcount额外申请sizeof(*refcount)。

```
// src/mat.h
NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

// src/mat.cpp
void Mat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();
    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;
    dims = 3;
    w = _w;
    h = _h;
    d = 1;
    c = _c;

    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }
    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

// src/mat.h
NCNN_FORCEINLINE size_t Mat::total() const
{
    return cstep * c;
}
```

## Mat引用计数

类似智能指针，Mat使用引用计数来管理内存。refcount初始值为1，拷贝构造函数、赋值重载函数均会对refcount加1。在释放时，对refcount减1，当refcount减至0时，采用allocator->fastFree或ncnn::fastFree释放内存。

```
// src/mat.h
NCNN_FORCEINLINE Mat& Mat::operator=(const Mat& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        NCNN_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    d = m.d;
    c = m.c;

    cstep = m.cstep;
    return *this;
}

NCNN_FORCEINLINE void Mat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

NCNN_FORCEINLINE void Mat::release()
{
    if (refcount && NCNN_XADD(refcount, -1) == 1)
    {
        if (allocator)
            allocator->fastFree(data);
        else
            fastFree(data);
    }

    data = 0;

    elemsize = 0;
    elempack = 0;

    dims = 0;
    w = 0;
    h = 0;
    d = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
}
```

## fill函数

fill函数，可以对Mat的数据内容用指定数值来进行填充。

```
// src/mat.h
NCNN_FORCEINLINE void Mat::fill(float _v)
{
    int size = (int)total();
    float* ptr = (float*)data;

    int i = 0;
#if __ARM_NEON
    float32x4_t _c = vdupq_n_f32(_v);
    for (; i + 3 < size; i += 4)
    {
        vst1q_f32(ptr, _c);
        ptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *ptr++ = _v;
    }
}
```
