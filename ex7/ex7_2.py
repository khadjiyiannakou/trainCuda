import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.cublas as cublas
from timeit import default_timer as timer

A = np.array(np.random.rand(10000,20000), order='F')
B = np.array(np.random.rand(20000,10000), order='F')

A_gpu = gpuarray.to_gpu(A) # it implicitely copies the data from host to device memory
B_gpu = gpuarray.to_gpu(B)

m, k = A_gpu.shape
k, n = B_gpu.shape

C_gpu = gpuarray.empty((m, n), np.float64) # allocate an array to store results

alpha = np.float64(1.0)
beta  = np.float64(0.0)

cublas_handle = cublas.cublasCreate()
start = timer()
cublas.cublasDgemm(cublas_handle, 'n', 'n', m, n, k, alpha, A_gpu.gpudata, m, B_gpu.gpudata, k, beta, C_gpu.gpudata, m)
dt=timer()-start
print("GPU multiplication using cuBLAS in %f secs" % dt)
cublas.cublasDestroy(cublas_handle)
C_gpu = C_gpu.reshape(C_gpu.shape, order = 'F') # interepret the data as column major like fortran since BLAS functionality uses this convention

start = timer()
C=np.linalg.linalg.dot(A, B)
dt2=timer()-start
print("Total difference is = %f" % (np.sum(C_gpu.get() - C) ))
print("CPU multiplication using BLAS behind numpy in %f secs" % dt2)
print("GPU is %f times faster than CPU" % (dt2/dt))
