nvcc -arch=sm_70 -O3 ex3_1.cu -o ex3_1.x &
nvcc -arch=sm_70 -O3 ex3_2.cu -o ex3_2.x &
nvcc -arch=sm_70 -O3 ex3_3.cu -o ex3_3.x &
nvcc -arch=sm_70 -O3 ex3_4.cu -o ex3_4.x &
nvcc -arch=sm_70 -O3 ex3_5.cu -o ex3_5.x &
wait

