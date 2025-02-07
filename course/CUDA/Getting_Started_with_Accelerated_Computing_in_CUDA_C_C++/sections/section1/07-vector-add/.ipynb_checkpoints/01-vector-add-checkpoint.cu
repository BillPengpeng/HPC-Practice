#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

/*void addVectorsInto(float *result, float *a, float *b, int N)
{
  for(int i = 0; i < N; ++i)
  {
    result[i] = a[i] + b[i];
  }
}*/

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;

  for (int i = indexWithinTheGrid; i < N; i += gridStride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

/*
  a = (float *)malloc(size);
  b = (float *)malloc(size);
  c = (float *)malloc(size);
*/
cudaMallocManaged(&a, size);
cudaMallocManaged(&b, size);
cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  // addVectorsInto(c, a, b, N);
size_t threads_per_block = 1024;
  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;;
  addVectorsInto<<<number_of_blocks, threads_per_block>>>(c, a, b, N);
cudaDeviceSynchronize();

  checkElementsAre(7, c, N);


  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
