#ifndef APELIB_H_
#define APELIB_H_

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
#include "constraints.h"

using namespace std;

#define EPSILON 0.01
#define C CEILBPC(N)

#ifdef POSITIVEEDGES
#define MINEDGEVALUE 0
#else
#define MINEDGEVALUE (-FLT_MAX)
#endif

#ifdef UNFEASIBLE
#define UNFEASIBLEVALUE FLT_MAX
#else
#define UNFEASIBLEVALUE 0
#endif

double *apeqis(const edge *g, value (*cf)(agent *, const chunk *, void *), void *data = NULL,
	       const chunk *l = NULL, agent maxc = N, agent maxl = N);

#endif /* APELIB_H_ */
