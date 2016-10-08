#ifndef APEQIS_H_
#define APEQIS_H_

#include <omp.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <immintrin.h>

// Cplex headers
#include <ilcplex/ilocplex.h>

#ifdef APE_SUBDIR
#include "../instance.h"
#else
#include "instance.h"
#endif
#include "params.h"
#include "macros.h"
#include "types.h"

#define D (N * DRIVERSPERC / 100)
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

using namespace std;

#include "apelib.h"
#include "constraints.h"
#include "random.h"
#include "iqsort.h"
#include "value.h"
#include "sp.h"

#endif /* APEQIS_H_ */
