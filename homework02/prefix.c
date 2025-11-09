#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>
#include "common.h"


void usage(int argc, char** argv);
void verify(int* sol, int* ans, int n);
void prefix_sum(int* src, int* prefix, int n);
void prefix_sum_p1(int* src, int* prefix, int n);
void prefix_sum_p2(int* src, int* prefix, int n);


int main(int argc, char** argv)
{
    // get inputs
    uint32_t n = 1048576;
    unsigned int seed = time(NULL);
    if(argc > 2) {
        n = atoi(argv[1]); 
        seed = atoi(argv[2]);
    } else {
        usage(argc, argv);
        printf("using %"PRIu32" elements and time as seed\n", n);
    }


    // set up data 
    int* prefix_array = (int*) AlignedMalloc(sizeof(int) * n);  
    int* input_array = (int*) AlignedMalloc(sizeof(int) * n);
    srand(seed);
    for(int i = 0; i < n; i++) {
        input_array[i] = rand() % 100;
    }


    // set up timers
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();


    // execute serial prefix sum and use it as ground truth
    start_t = ReadTSC();
    prefix_sum(input_array, prefix_array, n);
    end_t = ReadTSC();
    printf("Time to do O(N-1) prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));


    // execute parallel prefix sum which uses a NlogN algorithm
    int* input_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    int* prefix_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    memcpy(input_array1, input_array, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p1(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do O(NlogN) //prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);

    
    // execute parallel prefix sum which uses a 2(N-1) algorithm
    memcpy(input_array1, input_array, sizeof(int) * n);
    memset(prefix_array1, 0, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p2(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do 2(N-1) //prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);


    // free memory
    AlignedFree(prefix_array);
    AlignedFree(input_array);
    AlignedFree(input_array1);
    AlignedFree(prefix_array1);


    return 0;
}

void usage(int argc, char** argv)
{
    fprintf(stderr, "usage: %s <# elements> <rand seed>\n", argv[0]);
}


void verify(int* sol, int* ans, int n)
{
    int err = 0;
    for(int i = 0; i < n; i++) {
        if(sol[i] != ans[i]) {
            err++;
        }
    }
    if(err != 0) {
        fprintf(stderr, "There was an error: %d\n", err);
    } else {
        fprintf(stdout, "Pass\n");
    }
}

void prefix_sum(int* src, int* prefix, int n)
{
    prefix[0] = src[0];
    for(int i = 1; i < n; i++) {
        prefix[i] = src[i] + prefix[i - 1];
    }
}

void prefix_sum_p1(int* src, int* prefix, int n)
{
	int logn =  ceil(log2((double)n));
	int * tmp;
	int power;
	for (int i = 0; i <= logn; i++) {
		power = 1 << i;
		#pragma omp parallel for
		for (int j = 0; j < n; j++) {
			if (j >= power) {
				prefix[j] = src[j] + src[j - power];
			
			} else {
				prefix[j] = src[j];
			}

		}
		tmp = src;
		src = prefix;
		prefix = tmp;
	}
	if (n % 2 == 0) {
		tmp = src;
		src = prefix;
		prefix = tmp;
	}
	
}

void prefix_sum_p2(int* src, int* prefix, int n)
{
	if (n & (n-1)) {
		printf("Currently, this code only works on inputs where n (the array length) is a power of two.\n");
		exit(1);
	}
	int logn = log2((double)n);
	int step;
	int leap;
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		prefix[i] = src[i];
	}
	for (int i = 1; i < logn + 1; i++) {
		leap = 1 << i;	
		step = 1 << (i-1);
		#pragma omp parallel for
		for (int j = leap - 1; j < n; j += leap) {
			prefix[j] += prefix[j - step];
		}
	}
	
	int root = n - 1;
	int final = prefix[root];

	prefix[root] = 0;
	int tmp;

	for (int i = logn; i > 0; i--) {
		step = 1 << (i-1);
		leap = 1 << i;
		#pragma omp parallel for
		for (int j = root; j > 0; j -= leap) {
			tmp = prefix[j];
			prefix[j] += prefix[j-step];		
			prefix[j-step] = tmp;
		}
	}
	for (int i = 0; i < n - 1; i++) {
		prefix[i] = prefix[i+1];
	} prefix[root] = final;
		
}

