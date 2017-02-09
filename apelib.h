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
#include "colours.h"
#include "macros.h"
#include "iqsort.h"
#include "coal.h"
#include "cgls.h"
#include "ip.h"

using namespace std;
using namespace arma;

#define _C CEILBPC(_N)

#ifdef APE_UNFEASIBLE
#define UNFEASIBLEVALUE FLT_MAX
#else
#define UNFEASIBLEVALUE 0
#endif

#ifdef PARALLEL
#define _T CORES
#else
#define _T 1
#endif

#ifdef APE_CSV
#define APE_SILENT
#endif

#if defined(SINGLETONS) && defined(WEIGHT)
#error "Both hard (SINGLETONS) and soft (WEIGHT) constraints specified!"
#endif

typedef struct {

	value (*cf)(agent *, agent, void *);
	void *cfdata;
	value *b, tv;
	#ifdef SINGLETONS
	value *s;
	#endif

	size_t rowidx, locidx;
	size_t rowoff, valoff;
	umat *locs;
	id *size;

} funcdata;

value *apeqis(const edge *g, value (*cf)(agent *, agent, void *), void *cfdata = NULL,
	      const chunk *l = NULL, agent maxc = _N, agent maxl = _N);

#endif /* APELIB_H_ */
