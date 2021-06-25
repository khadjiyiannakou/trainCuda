#include <stdio.h>
#include <iostream>

using namespace std;

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("Device name: %s\n", prop.name);
    cout << prop.totalGlobalMem/(1024*1024*1024) << "\n";
  }
  return 0;
}


    // struct cudaDeviceProp {
    //     char name[256];
    //     size_t totalGlobalMem;
    //     size_t sharedMemPerBlock;
    //     int regsPerBlock;
    //     int warpSize;
    //     size_t memPitch;
    //     int maxThreadsPerBlock;
    //     int maxThreadsDim[3];
    //     int maxGridSize[3];
    //     size_t totalConstMem;
    //     int major;
    //     int minor;
    //     int clockRate;
    //     size_t textureAlignment;
    //     int deviceOverlap;
    //     int multiProcessorCount;
    //     int kernelExecTimeoutEnabled;
    //     int integrated;
    //     int canMapHostMemory;
    //     int computeMode;
    //     int concurrentKernels;
    //     int ECCEnabled;
    //     int pciBusID;
    //     int pciDeviceID;
    //     int tccDriver;
    // }
