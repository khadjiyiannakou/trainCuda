#include <stdio.h>
#include <stdlib.h>

#define N (100*1024*1024)
#define Nth 1024

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
inline void __cudaSafeCall( cudaError_t err, const char *file, const int line )
{
   if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
   return;
}

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
  int *a=nullptr, *b=nullptr, *c=nullptr;                    // host pointers
  int *d_a, *d_b, *d_c;        // device pointers
  size_t size = N * sizeof(int);

  CudaSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost));
  
  CudaSafeCall(cudaHostAlloc((void**) &a, size,cudaHostAllocMapped));// cudaHostAllocMapped));
  CudaSafeCall(cudaHostAlloc((void**) &b, size, cudaHostAllocMapped));
  CudaSafeCall(cudaHostAlloc((void**) &c, size, cudaHostAllocMapped));

  random_ints(a, N);
  random_ints(b, N);

  
  cudaHostGetDevicePointer(&d_a,a,0);
  cudaHostGetDevicePointer(&d_b,b,0);
  cudaHostGetDevicePointer(&d_c,c,0);
  

  cudaEvent_t start, stop;
  cudaEventCreate(&start);cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  
  // Launch kernel to add two vector with 1 thread and N blocks
  // Kernel calls are asynchronous
  addVecs<<<(N+Nth-1)/Nth,Nth>>>(d_c, d_a, d_b, N);

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime,start,stop);
  printf("Elapsed time is %f ms\n",elapsedTime);
  printf("Last element is %d\n",c[N-1]);
  cudaEventDestroy(start);cudaEventDestroy(stop);
  
  // needs cudaFree to deallocate host pointers which are allocated with cudaHostAlloc
  cudaFree(a); cudaFree(b); cudaFree(c);
  return 0;
}
