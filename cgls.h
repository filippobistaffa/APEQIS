#ifndef CGLS_H_
#define CGLS_H_

// Headers

#include "types.h"

#define TOLERANCE 1e-4
#define MAXITERATIONS 500

#define CUDACHECKERROR() { \
cudaError_t e=cudaGetLastError(); \
if (e!=cudaSuccess) { printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(0); } }

unsigned cudacgls(const value *val, const unsigned *ptr, const unsigned *ind, unsigned m,
		  unsigned n, unsigned nnz, value *b, value *x, float *rt, bool quiet);

#endif /* CGLS_H_ */
