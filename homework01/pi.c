#include <stdlib.h> 
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "common.h"
#include <inttypes.h>
#include <time.h>
#include <string.h>

#define PI 3.1415926535

void usage(int argc, char** argv);
double calcPi_Serial(int num_steps);
double calcPi_P1(int num_steps);
double calcPi_P2(int num_steps);


int main(int argc, char** argv)
{
    fprintf(stdout, "The first 10 digits of Pi are %0.10f\n", PI);
    char buffer[30];
    sprintf(buffer, "run_16_thr_%i.csv", atoi(argv[1]));
    FILE* out = fopen(buffer, "w");
    char s[] = "n, base, pi_base, p1, pi_1, p2, pi_2\n";
	fwrite(s, sizeof(char), strlen(s), out);

    // set up timer
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();
	
	for (int i = 1; i < 10; i++) {
    
  		uint32_t num_steps = (int)pow(10, (double)i);
		fprintf(out, "%i, ", num_steps);
		// calculate in serial 
    		start_t = ReadTSC();
		double Pi0 = calcPi_Serial(num_steps);
    		end_t = ReadTSC();
		printf("Time to calculate Pi serially with %"PRIu32" steps is: %g\n",
		num_steps, ElapsedTime(end_t - start_t));
    		fprintf(out, "%0.10f, ", ElapsedTime(end_t - start_t));
		fprintf(out, "%0.10f, ", Pi0);
		printf("Pi is %0.10f\n", Pi0);
    
   		 // calculate in parallel with integration
		start_t = ReadTSC();
    		double Pi1 = calcPi_P1(num_steps);
    		end_t = ReadTSC();
    		
		fprintf(out, "%0.10f, ", ElapsedTime(end_t - start_t));		
		fprintf(out, "%0.10f, ", Pi1);
    		printf("Time to calculate Pi in // with %"PRIu32" steps is: %g\n",
				num_steps, ElapsedTime(end_t - start_t));
		printf("Pi is %0.10f\n", Pi1);


		// calculate in parallel with Monte Carlo
		start_t = ReadTSC();
		double Pi2 = calcPi_P2(num_steps);
		end_t = ReadTSC();

    		fprintf(out, "%0.10f, ", ElapsedTime(end_t - start_t));
		fprintf(out, "%0.10f\n", Pi2);
		printf("Time to calculate Pi in // with %"PRIu32" guesses is: %g\n",
           	num_steps, ElapsedTime(end_t - start_t));
    		printf("Pi is %0.10f\n", Pi2);
	}
	
	fclose(out);
	return 0;
}


void usage(int argc, char** argv)
{
    fprintf(stdout, "usage: %s <# steps>\n", argv[0]);
}

double calcPi_Serial(int num_steps)
{
    double pi = 0.0;
	double line_seg = 1.0 / num_steps;
    for (int i = 0; i < num_steps; i++) {
	double left_end = (double)i / num_steps;
	double y = sqrt(1.0 - (left_end * left_end));
	pi += (y * line_seg) * 2;	
    }
    return pi * 2;
}

double calcPi_P1(int num_steps)
{
	double pi = 0.0;
	double line_seg = 1.0 / num_steps;
	#pragma omp parallel for reduction(+:pi)
	for (int i = 0; i < num_steps; i++) {
		double left_end = (double)i / num_steps;
		double y = sqrt(1.0 - (left_end * left_end));
		pi += (y * line_seg) * 2;	
    	}

    return pi * 2;
}

double calcPi_P2(int num_steps)
{
	double x, y;
	int sum;
	unsigned int seed;

	#pragma omp parallel private(x, y, seed)
	{
		// seed the thread
		seed = omp_get_thread_num();
	
		#pragma omp for reduction(+:sum)
		for (int i = 0; i < num_steps; i++) {
			x = (double) rand_r(&seed) / RAND_MAX;
			y = (double) rand_r(&seed) / RAND_MAX;
			if ((x*x + y*y) <= 1) {
				sum += 1;
			}
		}
	} 
	return 4 * ((double)sum / (double)num_steps);	
}
