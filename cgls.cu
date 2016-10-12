#include "cgls.h"
#include "cgls.cuh"

unsigned cudacgls(const value *val, const unsigned *ptr, const unsigned *ind, const unsigned m, const unsigned n, const unsigned nnz, const value *b, value *x) {

	unsigned rc = cgls::Solve<value, cgls::CSC>(val, (int *)ptr, (int *)ind, (int)m, (int)n, (int)nnz, b, x, 0, TOLERANCE, MAXITERATIONS, false);

	return rc;
}
