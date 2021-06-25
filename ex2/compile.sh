nvcc -arch=sm_70 -O3 ex2_1.cu -o ex2_1.x &
nvcc -arch=sm_70 -O3 ex2_2.cu -o ex2_2.x &
nvcc -arch=sm_70 -O3 ex2_3.cu -o ex2_3.x &
nvcc -arch=sm_70 -O3 ex2_4.cu -o ex2_4.x &

wait

