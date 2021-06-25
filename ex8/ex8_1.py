import numpy as np
from pylab import imsave 
from timeit import default_timer as timer


# The `mandel` function performs the Mandelbrot set calculation for a given (x,y) position on the imaginary plane. It returns the number of iterations before the computation "escapes".

########## Pseudocode from https://en.wikipedia.org/wiki/Mandelbrot_set ####################
# for each pixel (Px, Py) on the screen do
#     x0 := scaled x coordinate of pixel (scaled to lie in the Mandelbrot X scale (-2, 1))
#     y0 := scaled y coordinate of pixel (scaled to lie in the Mandelbrot Y scale (-1, 1))
#     x := 0.0
#     y := 0.0
#     iteration := 0
#     max_iteration := 1000
#     while (x*x + y*y < 2*2 AND iteration < max_iteration) do
#         xtemp := x*x - y*y + x0
#         y := 2*x*y + y0
#         x := xtemp
#         iteration := iteration + 1
    
#     color := palette[iteration]
#     plot(Px, Py, color)


######################################## Standard Python ##############################

#mandel function that given the coordinates we create the while loop 
def mandel(x, y, max_iters):
  """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
  """
  c = complex(x, y)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4:
      return i

  return max_iters


# iterates over all the pixels in the image, computing the complex coordinates from the pixel coordinates,
# and calls the `mandel` function at each pixel. The return value of `mandel` is used to color the pixel.


def create_fractal(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height
  
  for x in range(width):
    real = min_x + x * pixel_size_x
    for y in range(height):
      imag = min_y + y * pixel_size_y
      color = mandel(real, imag, iters)
      image[y, x] = color


# Create a 1536x1024 pixel image as a numpy array of bytes. We then call `create_fractal` with appropriate coordinates to fit the whole mandelbrot set.
image = np.zeros((1024, 1536), dtype = np.uint8)
start = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
dt = timer() - start

print("Mandelbrot created in %f s" % (dt*100)) # we multiply by 100 since later we will have x100 images
imsave("./img1.png",image,format='png')


################################################### CPU code but compiled
#https://numba.pydata.org/numba-doc/latest/user/jit.html
from numba import jit,njit,prange

@jit
def mandel(x, y, max_iters):
  """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
  """
  c = complex(x, y)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4:
      return i

  return max_iters

@njit(parallel=True)
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height
    
  for x in prange(width):
    real = min_x + x * pixel_size_x
    for y in range(height):
      imag = min_y + y * pixel_size_y
      color = mandel(real, imag, iters)
      image[y, x] = color



image = np.zeros((10240, 15360), dtype = np.uint8)
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)  # first time we call it includes compile time so we need to call it again to see the speedup
start = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20) # this time we do the timing
dt = timer() - start

print("Mandelbrot created in %f s" % dt)
imsave("./img2.png",image,format='png')

###################################### CUDA compilation ##############################
from numba import cuda
from numba import *

mandel_gpu = cuda.jit(device=True)(mandel) # the mandel function now it compiles also as a device function


@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height

  startX, startY = cuda.grid(2)               # location on the cuda grid for each thread
  gridX = cuda.gridDim.x * cuda.blockDim.x;   # offset we need to jump
  gridY = cuda.gridDim.y * cuda.blockDim.y;

  for x in range(startX, width, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, height, gridY):
      imag = min_y + y * pixel_size_y 
      image[y, x] = mandel_gpu(real, imag, iters)


gimage = np.zeros((10240, 15360), dtype = np.uint8)
blockdim = (32, 8)
griddim = (32,16)

# First time we do not time since it will compile it and run and will include overheads
d_image = cuda.to_device(gimage)
mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20) 
h_image=d_image.copy_to_host()


start = timer()
d_image = cuda.to_device(gimage)
mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20) 
h_image=d_image.copy_to_host()
dt = timer() - start

print("Mandelbrot created on GPU in %f s" % dt)

imsave("./img3.png",h_image,format='png')
