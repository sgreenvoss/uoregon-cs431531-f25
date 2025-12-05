cc = mpic++

spmv_sources = main.cc mmio.cc common.cc
spmv_objects = $(addsuffix .o, $(basename $(spmv_sources)))

flags = -g -W -Wall -std=c++14 -fopenmp -DDEBUG=0

spmv_target = spmv

all: $(spmv_target) 

$(spmv_target) : $(spmv_objects)
	$(cc) $(flags) -o $(spmv_target) $(spmv_objects)

%.o : %.cc
	$(cc) -c $(flags) $< -o $@

clean:
	rm -rf $(spmv_target) $(spmv_objects) 
