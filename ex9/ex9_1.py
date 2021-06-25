import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from mpi4py import MPI
from timeit import default_timer as timer

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

print(sizeComm, rank) # different MPI processes run asych. unless we put barriers
