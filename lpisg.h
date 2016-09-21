#ifndef LPISG_H_
#define LPISG_H_

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

#include "instance.h"
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

using namespace std;

#include "random.h"
#include "iqsort.h"
#include "slyce.h"
#include "value.h"
#include "sp.h"

#endif /* LPISG_H_ */
