#include <stdio.h>

__global__ void helloWorld(){
  printf("Hello World from (block=%d,thread=%d)\n",blockIdx.x,threadIdx.x);
}

int main(){
  helloWorld<<<3,2>>>();
  cudaDeviceSynchronize();
  return 0;
}
