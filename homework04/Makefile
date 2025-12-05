CC =  gcc
CPP = g++
NVCC = nvcc
CCFLAGS = -O3 -fopenmp -lm -std=c++11
CFLAGS += -O3 -fopenmp -lm -std=c11

NVCCFLAGS = -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90

SRC  = $(notdir $(wildcard *.cc)) $(notdir $(wildcard *.cu)) $(notdir $(wildcard *.c))

OBJS = $(addsuffix .o, $(basename $(SRC)))
EXEC = spmv

all: ${EXEC}

${EXEC}: ${OBJS}
	$(NVCC) ${NVCCFLAGS} -Xcompiler -fopenmp $^ -o $@

%.o : %.c
	$(CC) ${CFLAGS} ${INC} -c $< -o $@ ${LIBS}

%.o : %.cc
	$(CPP) ${CCFLAGS} ${INC} -c $< -o $@ ${LIBS}

%.o : %.cu
	$(NVCC) ${NVCCFLAGS} ${INC} -c $< -o $@

clean:
	rm -f *.o *.d *~ *.a *.so *.s ${EXEC}

