export PATH=/nvme/h/khadjiyiannakou/scratch/frensh_install/anaconda/bin:${PATH}
module purge
module load Autotools/20180311-GCCcore-8.3.0
module load CMake/3.15.3-GCCcore-8.3.0
module load gcccuda/2019b
module load OpenMPI/3.1.4-gcccuda-2019b
for i in 0 1 2 3 4 5 6
do
    cd ex$i
    ./compile.sh
    echo "ex$i compiled"
    cd ../
done
