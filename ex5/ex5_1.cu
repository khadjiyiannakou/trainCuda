#include <stdio.h>

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

int main(){
  double *d_a;
  double *a;
  size_t btoM=1024*1024;
  size_t size = 31*1024*btoM; 
  CudaSafeCall( cudaMalloc((void**)&d_a,size) ); // increase 31 -> to 32 to see the error reported
  CudaSafeCall( cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice) );
  return 0;
}
