#ifndef APELIB_H_
#define APELIB_H_

// Headers

#include <stdlib.h>
#include <float.h>

// Armadillo library

#include <iostream>
#include <armadillo>

#ifdef APE_SUBDIR
#include "../instance.h"
#include "../params.h"
#include "../types.h"
#else
#include "instance.h"
#include "types.h"
#endif
#include "params.h"
#include "macros.h"
#include "iqsort.h"
#include "coal.h"

using namespace std;

#define _C CEILBPC(_N)

#ifdef APE_UNFEASIBLE
#define UNFEASIBLEVALUE FLT_MAX
#else
#define UNFEASIBLEVALUE 0
#endif

#ifdef APE_CSV
#define APE_SILENT
#endif

double *apeqis(const edge *g, value (*cf)(agent *, agent, void *), void *data = NULL,
	       const chunk *l = NULL, agent maxc = _N, agent maxl = _N);

#endif /* APELIB_H_ */
