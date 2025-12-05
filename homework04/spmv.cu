#include <iostream>
#include <stdio.h>
#include <assert.h>

#include "helpers/helper_cuda.h"
#include <cooperative_groups.h>

#include "spmv.h"

// using n to pass the max row
template <class T>
__global__ void
spmv_kernel_ell(unsigned int* col_ind, T* vals, int m, int n, int nnz, 
                double* x, double* b)
{
	// this array will be shared between all threads in the block.
	extern __shared__ T data[];

	unsigned int row = blockIdx.x;
	unsigned int thread = threadIdx.x;

	T part_sum = 0;

	unsigned int row_start = row * n;
	unsigned int row_end = row_start + n;

	for (int k = row_start + thread; k < row_end; k+=blockDim.x) {
		int col = col_ind[k]; // not doing any checking here because
					// i have the oobs as 0.
		part_sum += vals[k] * x[col];
	}
	// this array holds the partial sums computed by each thread in the block
	data[thread] = part_sum;
	__syncthreads();

	// in parallel, reduce the partial sums into a single sum. 
	for (int i = blockDim.x / 2; i > 0; i>>=1) {
		if (thread < i) {
			data[thread] += data[thread + i];
		}
		__syncthreads();
	} 
	// assign the full sum computed for this row to the proper index.
	if (thread == 0) {
		b[row] = data[0];
	}

}





void spmv_gpu_ell(unsigned int* col_ind, double* vals, int m, int n, int nnz, 
                  double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters
    unsigned int blocks = m; 
    unsigned int threads = 32; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel_ell<double><<<dimGrid, dimBlock, shared>>>(col_ind, vals, 
                                                               m, n, nnz, x, b);
    }
    //checkCudaErrors(cudaFree(col_ind));
    //checkCudaErrors(cudaFree(vals));
    //checkCudaErrors(cudaFree(x));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));
    cudaDeviceSynchronize();
}




void allocate_ell_gpu(unsigned int* col_ind, double* vals, int m, int n, 
                      int nnz, double* x, unsigned int** dev_col_ind, 
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // copy ELL data to GPU and allocate memory for output
	CopyData(col_ind, m * n, sizeof(unsigned int), dev_col_ind);
	CopyData(vals, m * n, sizeof(double), dev_vals);
}

void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
                      double* vals, int m, int n, int nnz, double* x, 
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b)
{
	CopyData(row_ptr, m + 1, sizeof(int), dev_row_ptr);

	CopyData(col_ind, nnz, sizeof(int), dev_col_ind);

	CopyData(vals, nnz, sizeof(double), dev_vals);
	
	CopyData(x, n, sizeof(double), dev_x);
	
	checkCudaErrors(cudaMalloc((void**)dev_b, m * sizeof(double)));
}

void get_result_gpu(double* dev_b, double* b, int m)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;


    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(b, dev_b, sizeof(double) * m, 
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Pinned Host to Device bandwidth (GB/s): %f\n",
         (m * sizeof(double)) * 1e-6 / elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in)
{
  // timers
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  // Allocate pinned memory on host (for faster HtoD copy)
  T* h_in_pinned = NULL;
  checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
  assert(h_in_pinned);
  memcpy(h_in_pinned, input, N * dsize);

  // copy data
  checkCudaErrors(cudaMalloc((void**) d_in, N * dsize));
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(*d_in, h_in_pinned,
                             N * dsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("  Pinned Device to Host bandwidth (GB/s): %f\n",
         (N * dsize) * 1e-6 / elapsedTime);
	checkCudaErrors(cudaFreeHost(h_in_pinned));	
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


template <class T>
__global__ void
spmv_kernel(unsigned int* row_ptr, unsigned int* col_ind, T* vals, 
        int m, int n, int nnz, double* x, double* b)
{
	extern __shared__ T data[];

        unsigned int row = blockIdx.x;
        unsigned int thread = threadIdx.x;
	
	if (row >= m) return;

        T part_sum = 0;

        unsigned int row_start = row_ptr[row];
        unsigned int row_end = row_ptr[row + 1];
	
        for (int i = row_start + thread; i < row_end; i+=blockDim.x) {
                part_sum += vals[i] * x[col_ind[i]];
        }
        // this array holds the partial sums computed by each thread in the block
        data[thread] = part_sum;
        __syncthreads();

        // in parallel, reduce the partial sums into a single sum.
        for (int i = blockDim.x / 2; i > 0; i>>=1) {
                if (thread < i) {
                        data[thread] += data[thread + i];
                }
                __syncthreads();
        }
        // assign the full sum computed for this row to the proper index.
        if (thread == 0) {
                b[row] = data[0];
        }

/*	int row = threadIdx.x + blockDim.x * blockIdx.x;
	if (row < m) {
		int start = row_ptr[row];
		int end = row_ptr[row+1];
	
		T sum = 0;
		for (int i = start; i < end; i++) {
			sum += vals[i] * x[col_ind[i]];
		}
		b[row] = sum;
	}
*/
}


void spmv_gpu(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
              int m, int n, int nnz, double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters
    // 1 thread block per row
    // 64 threads working on the non-zeros on the same row
    unsigned int blocks = m; 
    unsigned int threads = 64; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel<double><<<dimGrid, dimBlock, shared>>>(row_ptr, col_ind, 
                                                           vals, m, n, nnz, 
                                                           x, b);
    }
    //checkCudaErrors(cudaFree(col_ind));
    //checkCudaErrors(cudaFree(vals));
    //checkCudaErrors(cudaFree(x));
    //checkCudaErrors(cudaFree(row_ptr));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));
    cudaEventDestroy(start);
	cudaEventDestroy(stop);
    cudaDeviceSynchronize();
}

void spmv_all_free(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
              double* x, double* b, unsigned int *row, double*v) {
	cudaFree(row_ptr);
	cudaFree(col_ind);
	cudaFree(vals);
	cudaFree(x);
	cudaFree(b);
	cudaFree(row);
	cudaFree(v);
}
