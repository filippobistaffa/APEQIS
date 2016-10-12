#ifndef COAL_H_
#define COAL_H_

// Headers

#include <stdlib.h>
#include "instance.h"
#include "macros.h"
#include "types.h"

#include "iqsort.h"
#include "sorted.h"

#define _C CEILBPC(_N)

template <agent n>
__attribute__((always_inline)) inline
agent *createadj(const edge *g, edge ne, const chunk *l) {

	if (!ne) return NULL;
	agent *adj = (agent *)calloc(n * n, sizeof(agent));
	agent ab[2 * ne];
	edge ei = 0;

	for (agent v1 = 0; v1 < n; v1++)
		for (agent v2 = v1 + 1; v2 < n; v2++) {
			const edge e = g[v1 * n + v2];
			if (e) {
				X(ab, ei) = v1;
				Y(ab, ei) = v2;
				ei++;
			}
		}

	agent *a = ab;

	do {
		adj[a[0] * n + (adj[a[0] * n]++) + 1] = a[1];
		adj[a[1] * n + (adj[a[1] * n]++) + 1] = a[0];
		a += 2;
	} while (--ne);

	for (agent i = 0; i < n; i++)
		QSORT(agent, adj + i * n + 1, adj[i * n], LTL);

	return adj;
}

// Count the number of elements in the buffer set to 1 in the mask

template <typename type>
__attribute__((always_inline)) inline
unsigned maskcount(const type *buf, unsigned n, const chunk *mask) {

	unsigned ret = 0;
	do {
		ret += GET(mask, *buf);
		buf++;
	} while (--n);
	return ret;
}

void coalitions(const edge *g, void (*cf)(agent *, agent, const edge *, const agent *, const chunk *, void *),
		void *data = NULL, agent maxc = _N, const chunk *l = NULL, agent maxl = _N);

#endif /* COAL_H_ */
