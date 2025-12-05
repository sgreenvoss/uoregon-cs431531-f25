cc = gcc

sources = main.c mmio.c common.c
objects = $(addsuffix .o, $(basename $(sources)))
libs = -lm
flags = -g -std=c11 -fopenmp

target = spmv

all: $(target) 

$(target) : $(objects)
	$(cc) $(flags) -o $(target) $(objects) $(libs)

%.o : %.c
	$(cc) -c $(flags) $< -o $@ $(libs)

clean:
	rm -rf $(target) $(objects) 
