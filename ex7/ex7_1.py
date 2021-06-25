import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

N=512

a = np.random.rand(N)
b = np.random.rand(N)
c = np.empty(a.shape,dtype=a.dtype)

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

b_gpu = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(b_gpu, b)

c_gpu = cuda.mem_alloc(b.nbytes)

mod = SourceModule("""
__global__ void addVecs(double* a, double* b, double* c){
c[threadIdx.x]=a[threadIdx.x]+b[threadIdx.x];
}
""")

addVecs=mod.get_function("addVecs")
addVecs(a_gpu,b_gpu,c_gpu,block=(N,1,1))
cuda.memcpy_dtoh(c,c_gpu)
print("Total difference is = %f" % np.sum(a+b-c) )


