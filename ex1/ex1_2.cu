#include <stdio.h>

__global__ void helloWorld(){
  printf("Hello World!\n");
}

int main(){
  helloWorld<<<1,1>>>();
  return 0;
}
