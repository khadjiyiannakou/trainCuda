#include <stdio.h>
#define N (100*1024*1024)
#define CHUNK_SIZE (1024*1024)

void random_ints(int* a, int size){
  for(int i =0; i<size; i++)
    a[i]=rand()%1000;
}

__global__ void addVecs(int *c, int *a, int *b){
  int index = threadIdx.x + blockIdx.x *  blockDim.x;
  c[index] = a[index]+b[index];
}

int main(){
  int *h_x, *h_y, *h_z;
  int *d_x0, *d_y0, *d_z0; // for stream 0
  int *d_x1, *d_y1, *d_z1; // for stream 1

  // Allocate page-locked host memory
  cudaHostAlloc((void**)&h_x, N*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_y, N*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_z, N*sizeof(int), cudaHostAllocDefault);

  // initialize vectors with random numbers
  random_ints(h_x,N);
  random_ints(h_y,N);

  
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  cudaStream_t stream0, stream1;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);

  // Allocate device memory
  cudaMalloc((void**)&d_x0, CHUNK_SIZE*sizeof(int));
  cudaMalloc((void**)&d_y0, CHUNK_SIZE*sizeof(int));
  cudaMalloc((void**)&d_z0, CHUNK_SIZE*sizeof(int));

  cudaMalloc((void**)&d_x1, CHUNK_SIZE*sizeof(int));
  cudaMalloc((void**)&d_y1, CHUNK_SIZE*sizeof(int));
  cudaMalloc((void**)&d_z1, CHUNK_SIZE*sizeof(int));



  for(int i = 0; i < N ; i += 2* CHUNK_SIZE){
    // operations on stream0
    cudaMemcpyAsync(d_x0, h_x+i, CHUNK_SIZE*sizeof(int), cudaMemcpyHostToDevice,stream0);
    cudaMemcpyAsync(d_y0, h_y+i, CHUNK_SIZE*sizeof(int), cudaMemcpyHostToDevice,stream0);
    addVecs<<<CHUNK_SIZE/1024,CHUNK_SIZE/1024, 0, stream0>>>(d_z0, d_x0, d_y0);
    cudaMemcpyAsync(h_z+i, d_z0, CHUNK_SIZE*sizeof(int), cudaMemcpyDeviceToHost,stream0);

    // operations on stream1
    cudaMemcpyAsync(d_x1, h_x+i+CHUNK_SIZE, CHUNK_SIZE*sizeof(int), cudaMemcpyHostToDevice,stream1);
    cudaMemcpyAsync(d_y1, h_y+i+CHUNK_SIZE, CHUNK_SIZE*sizeof(int), cudaMemcpyHostToDevice,stream1);
    addVecs<<<CHUNK_SIZE/1024,CHUNK_SIZE/1024, 0, stream1>>>(d_z1, d_x1, d_y1);
    cudaMemcpyAsync(h_z+i+CHUNK_SIZE, d_z1, CHUNK_SIZE*sizeof(int), cudaMemcpyDeviceToHost,stream1);
  }
  // we need to sync both streams
  cudaStreamSynchronize(stream0);
  cudaStreamSynchronize(stream1);

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Elapsed Time is %f ms \n",elapsedTime);
  printf("Last element is %d\n",h_z[N-1]);
  cudaFreeHost(h_x); cudaFreeHost(h_y); cudaFreeHost(h_z);
  cudaFree(d_x0);  cudaFree(d_y0); cudaFree(d_z0); cudaFree(d_x1);  cudaFree(d_y1); cudaFree(d_z1);
  cudaStreamDestroy(stream0); cudaStreamDestroy(stream1); cudaEventDestroy(start); cudaEventDestroy(stop);
  return 0;
}
