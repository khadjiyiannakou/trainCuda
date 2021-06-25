Example 0 (C)
-----------
1) Explains how to find the properties of the GPU devices available.
   In order to compile it execute the script compile.sh
####################################################################
Example 1 (C)
---------
1) Standard CPU code is given from the nvcc compiler to the host compiler
2) Include "Hello world" kernel needs files to have extensions .cu
3) The meaning of asynchronous kernels
4) Choose different number of blocks & threads, asycnh. blocks
###################################################################
Example 2 (C)
---------
1) Add two vectors with 1 block and N threads
2) Add two vectors with N blocks and 1 thread
3) Add two vectors with N blocks and m threads
4) Add two vector with non divizible number of elements with blocks-threads
##################################################################
Example 3 (C)
---------
1) Dot product two vectors no thread synch.
2) Dot product two vectors with thread synch.
3) Dot product two vectors now using two blocks and simple addition
4) Dot product two vectors with two blocks and the atomicAdd
5) Dot product two vectors now with extern shared memory
#################################################################
Example 4 (C)
---------
1) Example to add vectors but explaining how to do timing with events
2) How to overlap communication and computation on GPU
################################################################
Example 5 (C)
---------
1) Example of how to call CUDA API functions in a safe manner
2) Example how to get errors from the kernel calls
###############################################################
Example 6 (C)
---------
1) Example to add vectors but using mapped memory between host and device
##############################################################
Example 7 (P)
---------
1) How to use pycuda to add two vectors with kernel example
2) Matrix-Matrix multiplication example using cublas from skcuda
#############################################################
Example 8 (P)
---------
1) The Mandelbrot fractal, how to jitify both CPU and GPU implementations
############################################################
Example 9 (P)
---------
1) Example of initializing MPI for python and meaning of communicators, rank
2) Vector addition on multiple GPUs using python
