#ifndef CONSTRAINTS_H_
#define CONSTRAINTS_H_

// Headers

#include <ilcplex/ilocplex.h>
#include <float.h>

#ifdef APE_SUBDIR
#include "../instance.h"
#include "../types.h"
#else
#include "instance.h"
#include "types.h"
#endif
#include "params.h"
#include "macros.h"
#include "iqsort.h"

using namespace std;

__attribute__((always_inline)) inline
void printc(const agent *c, value v) {

	agent n = *c;
	printf("[ ");
	while (n--) printf("%u ", *(++c));
	printf("] = %.2f\n", v);
}

value constraints(const edge *g, const agent *adj, const chunk *l, value (*cf)(agent *, const chunk *, void *), void *data,
		  IloEnv &env, IloModel &model, IloFloatVarArray &ea, IloFloatVarArray &da, agent maxc, agent maxl);

#endif /* CONSTRAINTS_H_ */
