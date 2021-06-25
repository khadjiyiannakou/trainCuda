nvcc -arch=sm_70 -O3 ex1_1.c -o ex1_1.x &
nvcc -arch=sm_70 -O3 ex1_2.cu -o ex1_2.x &
nvcc -arch=sm_70 -O3 ex1_3.cu -o ex1_3.x &
nvcc -arch=sm_70 -O3 ex1_4.cu -o ex1_4.x &

wait
