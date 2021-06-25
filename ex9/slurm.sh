#!/bin/bash -l
#SBATCH --job-name="test"    # this is the name of slurm script we will submit
#SBATCH --nodes=1            # We will allocate one node for this job
#SBATCH --ntasks=4           # We will have 4 parallel processes per node
#SBATCH --partition=gpu      # This specify that we will use the GPU partition from the machine
#SBATCH --gres=gpu:4         # We allocate 4 GPUs for that
#SBATCH --time=00:10:00      # The time limit of the job
#SBATCH --output=./test.out  # Where the output will be provided
#SBATCH --error=./test.err   # The error if any
 
mpirun -n 4 python ./ex9_2.py # this is command we will execute provided with 4 parallel processes and a python script to run

