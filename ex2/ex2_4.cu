#include <stdio.h>
#include <stdlib.h>

#define N 1000
#define Nth 32

void random_ints(int* a, int size){
  for(int i =0; i<size; i++)
    a[i]=rand()%1000;
}

__global__ void addVecs(int *c, int *a, int *b, int L){
  int index = threadIdx.x + blockIdx.x *  blockDim.x;
  if(index < L)
    c[index] = a[index]+b[index];
}

int main(void){
  int *a, *b, *c;                    // host pointers
  int *d_a, *d_b, *d_c;        // device pointers
  int size = N * sizeof(int);

  // Alloc space for device copies of a, b, c  
  cudaMalloc((void **)&d_a, size); 
  cudaMalloc((void **)&d_b, size); 
  cudaMalloc((void **)&d_c, size); 

  a = (int *)malloc(size);
  random_ints(a, N);   // Alloc space host, random initialization
  b = (int *)malloc(size);
  random_ints(b, N);
  c = (int *)malloc(size); 

  // Copy data from host to device memory
  // cudaMemcpyHostToDevice is a flag determining copying from host to dev.
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Launch kernel to add two vector with 1 thread and N blocks
  // Kernel calls are asynchronous
  addVecs<<<(N+Nth-1)/Nth,Nth>>>(d_c, d_a, d_b, N);

  // Copy results from device to host
  // cudaMemcpy blocks CPU until Kernels finish execution
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  for(int i =0; i<N; i++)
    printf("%d + %d = %d\n",a[i],b[i],c[i]);

  
  // needs cudaFree to deallocate device pointers
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  free(a); free(b); free(c);
  return 0;
}
