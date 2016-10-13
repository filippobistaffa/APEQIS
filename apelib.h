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
#include "cgls.h"

using namespace std;
using namespace arma;

#define _C CEILBPC(_N)

#ifdef APE_UNFEASIBLE
#define UNFEASIBLEVALUE FLT_MAX
#else
#define UNFEASIBLEVALUE 0
#endif

#ifdef APE_CSV
#define APE_SILENT
#endif

typedef struct {

	value (*cf)(agent *, agent, void *);
	void *cfdata;
	value *b, tv;

	size_t rowidx, locidx;
	umat *locs;

} funcdata;

value *apeqis(const edge *g, value (*cf)(agent *, agent, void *), void *cfdata = NULL,
	       const chunk *l = NULL, agent maxc = _N, agent maxl = _N);

#endif /* APELIB_H_ */
