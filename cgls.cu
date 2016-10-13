#include "cgls.h"
#include "cgls.cuh"

unsigned cudacgls(const value *val, const unsigned *ptr, const unsigned *ind, const unsigned m,
		  const unsigned n, const unsigned nnz, const value *b, value *x) {

	value *d_val, *d_b, *d_x;
	int *d_ptr, *d_ind;

	cudaMallocManaged(&d_val, sizeof(value) * nnz);
	cudaMallocManaged(&d_ptr, sizeof(int) * n + 1);
	cudaMallocManaged(&d_ind, sizeof(int) * nnz);
	cudaMallocManaged(&d_b, sizeof(value) * m);
	cudaMallocManaged(&d_x, sizeof(value) * n);

	memcpy(d_val, val, sizeof(value) * nnz);
	memcpy(d_ptr, ptr, sizeof(int) * n + 1);
	memcpy(d_ind, ind, sizeof(int) * nnz);
	memcpy(d_b, b, sizeof(value) * m);
	memset(d_x, 0, sizeof(value) * n);

	unsigned rc = cgls::Solve<value, cgls::CSC>(d_val, d_ptr, d_ind, (int)m, (int)n, (int)nnz,
						    d_b, d_x, 0, TOLERANCE, MAXITERATIONS, !CGLSDEBUG);

	cudaDeviceSynchronize();
	memcpy(x, d_x, sizeof(value) * n);

	cudaFree(d_val);
	cudaFree(d_ptr);
	cudaFree(d_ind);
	cudaFree(d_b);
	cudaFree(d_x);

	return rc;
}
