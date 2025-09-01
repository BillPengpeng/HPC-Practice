本文主要整理PMPP Chapter 3 Exercise。

## Q1

1. In this chapter we implemented a matrix multiplication kernel that has each
 thread produce one output matrix element. In this question, you will
 implement different matrix-matrix multiplication kernels and compare them.
 - a. Write a kernel that has each thread produce one output matrix row. Fill in
 the execution configuration parameters for the design.
 - b. Write a kernel that has each thread produce one output matrix column. Fill
 in the execution configuration parameters for the design.
 - c. Analyze the pros and cons of each of the two kernel designs.
 - Answer: 

```c
__global__ void matrixMulGPUPerRow( int * a, int * b, int * c, int width)
{
  /*
   * Build out this kernel.
   */
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < width)
  {
    for (int col = 0; col < width; col++)
    {
      int val = 0;
      for ( int k = 0; k < width; ++k )
        val += a[row * width + k] * b[k * width + col];
      c[row * width + col] = val;
    }
  }
}

__global__ void matrixMulGPUPerCol( int * a, int * b, int * c, int width)
{
  /*
   * Build out this kernel.
   */
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < width)
  {
    for (int row = 0; row < width; row++)
    {
      int val = 0;
      for ( int k = 0; k < width; ++k )
        val += a[row * width + k] * b[k * width + col];
      c[row * width + col] = val;
    }
  }
}

dim3 threads_per_block (16, 1, 1); // A 16 x 16 block threads
dim3 number_of_blocks ((N / threads_per_block.x) + 1, 1, 1);
matrixMulGPUPerRow <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu, N );

dim3 threads_per_block (1, 16, 1); // A 16 x 16 block threads
dim3 number_of_blocks (1, (N / threads_per_block.x) + 1, 1);
matrixMulGPUPerCol <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu, N );
```

**每个线程计算一行的优缺点：**

*优点：*
- **对矩阵M的连续访问**：每个线程访问M的一整行，由于M按行优先存储，这些访问是连续的，能够实现高效的内存合并访问
- **线程映射简单**：线程索引直接对应行索引，代码逻辑清晰

*缺点：*
- **对矩阵N的非连续访问**：需要访问N的不同行中的元素，导致内存访问模式不连续，严重影响性能
- **寄存器使用量高**：每个线程需要计算整行结果，占用更多寄存器资源
- **并行度较低**：线程数量受行数I限制，可能无法充分利用GPU的并行能力

**每个线程计算一列的优缺点：**

*优点：*
- **线程映射简单**：线程索引直接对应列索引
- **在某些情况下可能优化N的访问**：但受限于行优先存储，效果有限

*缺点：*
- **对矩阵N的非连续访问**：访问N的列元素在内存中不连续，无法实现内存合并访问
- **对矩阵M的跨行访问**：虽然每行内访问连续，但需要跨行访问，效率不高
- **寄存器使用量高**：每个线程计算整列结果，需要较多寄存器
- **并行度较低**：线程数量受列数K限制

## Q2

2. A matrix-vector multiplication takes an input matrix B and a vector C and
 produces one output vector A. Each element of the output vector A is the dot
 product of one row of the input matrix B and C, that is, A[i]5Pj B[i][j]1C[j].
 For simplicity we will handle only square matrices whose elements are single
precision floating-point numbers. Write a matrix-vector multiplication kernel and
 the host stub function that can be called with four parameters: pointer to the output
 matrix, pointer to the input matrix, pointer to the input vector, and the number of
 elements in each dimension. Use one thread to calculate an output vector element.

 - Answer: 

```c
__global__ void matrixMulGPU( int * A, int * B, int *C)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // 计算行索引
    
    if (i < N) {  // 边界检查
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            // 访问矩阵元素 B[i][j] (行优先存储)
            sum += B[i * N + j] * C[j];
        }
        A[i] = sum;  // 写入结果
    }
}
void matrixMulCPU( int * A, int * B, int * C)
{
  int sum = 0;

  for( int i = 0; i < N; ++i )
  {
    sum = 0;
    for( int j = 0; j < N; ++j )
    {
      sum += B[i * N + j] * C[j];
    }
    A[i] = sum;  // 写入结果
  }
}

int blockSize = 16;  // 典型块大小
int gridSize = (N + blockSize - 1) / blockSize;  // 向上取整
matrixMulGPU <<< gridSize, blockSize >>> ( a_gpu, b, c );
```

## Q3

3. Consider the following CUDA kernel and the corresponding host function that
 calls it:

```c
__global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(row < M && col < N) {
        b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
    }
}

void foo(float* a_d, float* b_d) {
    unsigned int M = 150;
    unsigned int N = 300;
    dim3 bd(16, 32);
    dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1);
    foo_kernel <<< gd, bd >>> (a_d, b_d, M, N);
}
```

- a. What is the number of threads per block?
- b. What is the number of threads in the grid?
- c. What is the number of blocks in the grid?
- d. What is the number of threads that execute the code on line 05?
- Answer: 
  - a. 16 * 32 = 512
  - b. 19 * 5 * 512 = 48640
  - c. 19 * 5 = 95
  - d. 150 * 300 = 45000

## Q4

4. Consider a 2D matrix with a width of 400 and a height of 500. The matrix is
stored as a one-dimensional array. Specify the array index of the matrix
element at row 20 and column 10:
- a. If the matrix is stored in row-major order.
- b. If the matrix is stored in column-major order.
- Answer: 
  - a. 20 * 400 + 10 = 8010
  - b. 10 * 500 + 20 = 5020

## Q5

5. Consider a 3D tensor with a width of 400, a height of 500, and a depth of
300. The tensor is stored as a one-dimensional array in row-major order.
Specify the array index of the tensor element at x=10, y=20, and z=5.
- Answer: 5 * (400*500) + 20 * 400 + 10 = 1008010