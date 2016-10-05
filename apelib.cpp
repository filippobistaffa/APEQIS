#include "apeqis.h"

agent *creteadj(const edge *g, edge ne, const chunk *l) {

	agent *adj = (agent *)calloc(N * N, sizeof(agent));
	agent ab[2 * ne];
	edge e = 0;

	for (agent i = 0; i < N; i++)
		for (agent j = i + 1; j < N; j++)
			if (g[i * N + j]) {
				X(ab, e) = i;
				Y(ab, e) = j;
				e++;
			}

	agent *a = ab;

	do {
		adj[a[0] * N + (adj[a[0] * N]++) + 1] = a[1];
		adj[a[1] * N + (adj[a[1] * N]++) + 1] = a[0];
		a += 2;
	} while (--e);

	for (agent i = 0; i < N; i++)
		QSORT(agent, adj + i * N + 1, adj[i * N], LTL);

	return adj;
}
