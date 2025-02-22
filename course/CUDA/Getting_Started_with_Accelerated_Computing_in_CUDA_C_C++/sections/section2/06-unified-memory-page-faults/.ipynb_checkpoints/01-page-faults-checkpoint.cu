__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

int main()
{

  int N = 2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);

  /*
   * Conduct experiments to learn more about the behavior of
   * `cudaMallocManaged`.
   *
   * What happens when unified memory is accessed only by the GPU?
   * What happens when unified memory is accessed only by the CPU?
   * What happens when unified memory is accessed first by the GPU then the CPU?
   * What happens when unified memory is accessed first by the CPU then the GPU?
   *
   * Hypothesize about UM behavior, page faulting specificially, before each
   * experiment, and then verify by running `nsys`.
   */
  // What happens when unified memory is accessed only by the GPU?
  deviceKernel<<<256, 256>>>(a, N);
  cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

  // What happens when unified memory is accessed only by the CPU?
  // hostFunction(a, N);

  // What happens when unified memory is accessed first by the GPU then the CPU?
  // deviceKernel<<<256, 256>>>(a, N);
  // hostFunction(a, N);

  // What happens when unified memory is accessed first by the CPU then the GPU?
  // hostFunction(a, N);
  // deviceKernel<<<256, 256>>>(a, N);

  // cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

  cudaFree(a);
}
