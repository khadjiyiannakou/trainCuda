import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from mpi4py import MPI
from timeit import default_timer as timer
import sys

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()
import pycuda.autoinit
NGPUs=cuda.Device(0).count() # count the number of GPUs

if sizeComm != NGPUs:
    print("We want to map one MPI task to each GPU")
    sys.exit(-1)

dev=cuda.Device(rank) # each MPI task will handle different GPU based on the rank
ctx=dev.make_context()
########## instead of reading a long vector and break in different process we randomly create it
NN=100
N=NN*1024*1024
np.random.seed((1+rank)*12345) # set a different seed for each process to have a long vector different
a = np.random.rand(N)
b = np.random.rand(N)
c = np.empty(a.shape,dtype=a.dtype)

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

b_gpu = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(b_gpu, b)

c_gpu = cuda.mem_alloc(b.nbytes)
freeM,totalM=cuda.mem_get_info()

if rank ==0: print("Each GPU holds %d elements of the vector" % (len(a)))
print("Rank=%d, Used memory is %f GB" % (rank,(totalM-freeM)/(1024*1024*1024)) )

mod = SourceModule("""
__global__ void addVecs(double* a, double* b, double* c){
long int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y*gridDim.x*blockDim.x;
c[index]=a[index]+b[index];
}
""")

addVecs=mod.get_function("addVecs")
start = timer()
addVecs(a_gpu,b_gpu,c_gpu,grid=(1024,NN,1),block=(1024,1,1))
cuda.memcpy_dtoh(c,c_gpu)
dt=timer()-start
print("Rank %d: Time to add vectors is %f secs" % (rank, dt))
ctx.pop()
print("Rank %d: Total difference is = %f" % (rank,np.sum(a+b-c)) )


#fullVec=np.array(comm.gather(c,root=0)).flatten() # collect from each MPI process their chuck of the vector
#if rank == 0: print("Now we have the full vector with %d elements" % len(fullVec))

