#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define N_THR 512

void fill_ints(int* a, int size){
  for(int i =0; i<size; i++)
    a[i]=i;
}

__global__ void dotVecs(int *x, int *y, int *r){
  __shared__ int s_tmp[N_THR];
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int temp = x[index] * y[index];
  s_tmp[threadIdx.x] = temp;  // store the multiplication to the shared memory

  __syncthreads();
  // Thread 0 performs the reduction 
  if(threadIdx.x == 0){
    int sum = 0;
    for(int i = 0 ; i < N_THR ; i++) sum += s_tmp[i];
    *r += sum;
  }
}

int main(void){
  int *a, *b, *c;                    // host pointers
  int *d_a, *d_b, *d_c;        // device pointers
  int size = N * sizeof(int);

  // Alloc space for device copies of a, b, c  
  cudaMalloc((void **)&d_a, size); 
  cudaMalloc((void **)&d_b, size); 
  cudaMalloc((void **)&d_c, sizeof(int)); 

  a = (int *)malloc(size);
  fill_ints(a, N);   // Alloc space host, random initialization
  b = (int *)malloc(size);
  fill_ints(b, N);
  c = (int *)malloc(sizeof(int));
  
  // Copy data from host to device memory
  // cudaMemcpyHostToDevice is a flag determining copying from host to dev.
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  cudaMemset(d_c,0,sizeof(int));
  
  // Launch kernel to add two vector with N threads and 1 block
  // Kernel calls are asynchronous
  dotVecs<<<2,N_THR>>>(d_a, d_b, d_c);

  // Copy results from device to host
  // cudaMemcpy blocks CPU until Kernels finish execution
  cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

  printf("%d\n",*c);

  
  // needs cudaFree to deallocate device pointers
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  free(a); free(b); free(c);
  return 0;
}
