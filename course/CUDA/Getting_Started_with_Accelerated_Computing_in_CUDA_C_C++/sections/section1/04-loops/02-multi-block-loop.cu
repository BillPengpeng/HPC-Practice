#include <stdio.h>

/*
 * Refactor `loop` to be a CUDA Kernel. The new kernel should
 * only do the work of 1 iteration of the original loop.
 */

/*void loop(int N)
{
  for (int i = 0; i < N; ++i)
  {
    printf("This is iteration number %d\n", i);
  }
}
*/

__global__ void loop()
{
  printf("This is iteration number %d\n", threadIdx.x + blockIdx.x * blockDim.x);
}


int main()
{
  /*
   * When refactoring `loop` to launch as a kernel, be sure
   * to use the execution configuration to control how many
   * "iterations" to perform.
   *
   * For this exercise, be sure to use more than 1 block in
   * the execution configuration.
   */

  int N = 10;
  // loop(N);
  loop<<<2, N / 2>>>();
  cudaDeviceSynchronize();
}
