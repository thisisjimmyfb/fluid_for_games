rm navier_stoke.o
rm particle.o
/usr/local/cuda/bin/nvcc -c -g -D_DEBUG -D_CONSOLE --maxrregcount=16 -Xptxas=-v navier_stoke.cu
/usr/local/cuda/bin/nvcc -c -g -D_DEBUG -D_CONSOLE --maxrregcount=16 -Xptxas=-v particle.cu
