#include "lpisg.h"

// Print content of buffer

#include <iostream>
template <typename type>
__attribute__((always_inline)) inline
void printbuf(const type *buf, unsigned n, const char *name = NULL) {

	if (name) printf("%s = [ ", name);
	else printf("[ ");
	while (n--) std::cout << *(buf++) << " ";
	printf("]\n");
}

#ifndef TWITTER

__attribute__((always_inline)) inline
void createedge(agent *adj, agent v1, agent v2) {

	printf("%u -- %u\n", v1, v2);
	adj[v1 * N + (adj[v1 * N]++) + 1] = v2;
	adj[v2 * N + (adj[v2 * N]++) + 1] = v1;
}

__attribute__((always_inline)) inline
edge scalefree(agent *adj, const chunk *dr) {

	edge ne = 0;
	agent deg[N] = {0};

	for (agent i = 1; i <= M; i++) {
		for (agent j = 0; j < i; j++) {
			createedge(adj, i, j);
			deg[i]++;
			deg[j]++;
			ne++;
		}
	}

	agent t = 0;

	for (agent i = M + 1; i < N; i++) {
		t &= ~((1UL << i) - 1);
		for (agent j = 0; j < M; j++) {
			agent d = 0;
			for (agent h = 0; h < i; h++)
				if (!((t >> h) & 1)) d += deg[h];
			if (d > 0) {
				int p = nextInt(d);
				agent q = 0;
				while (p >= 0) {
					if (!((t >> q) & 1)) p = p - deg[q];
					q++;
				}
				q--;
				t |= 1UL << q;
				createedge(adj, i, q);
				deg[i]++;
				deg[q]++;
				ne++;
			}
		}
	}

	for (agent i = 0; i < N; i++)
		QSORT(agent, adj + i * N + 1, adj[i * N], LTDR);

	return ne;
}

#endif

int main(int argc, char *argv[]) {

	unsigned seed = atoi(argv[1]);
	meter *sp = createsp(seed);

	agent dra[N] = {0};
	chunk dr[C] = {0};

	for (agent i = 0; i < D; i++)
		dra[i] = 1;

	shuffle(dra, N, sizeof(agent));

	for (agent i = 0; i < N; i++)
		if (dra[i]) SET(dr, i);

	init(seed);
	agent *adj = (agent *)calloc(N * N, sizeof(agent));
	edge ne = scalefree(adj, dr);

	#ifdef DEBUG
	printf("%u edges + %u autoedges\n", ne, N);
	puts("Adjacency lists");
	for (agent i = 0; i < N; i++)
		printbuf(adj + i * N + 1, adj[i * N]);
	#endif

	// CPLEX environment and model

	IloEnv env;
	IloModel model(env);

	// Variables representing edge values

	IloFloatVarArray e(env, ne + N);
	ostringstream ostr;

	for (agent i = 0; i < ne + N; i++) {
		ostr << "e[" << i << "]";
		e[i] = IloFloatVar(env, -FLT_MAX, FLT_MAX, ostr.str().c_str());
		//cout << ostr.str() << endl;
		ostr.str("");
	}

	env.end();
	free(adj);

	return 0;
}
