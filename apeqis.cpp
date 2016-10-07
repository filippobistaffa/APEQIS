#include "apeqis.h"

__attribute__((always_inline)) inline
void createedge(edge *g, agent v1, agent v2, edge e) {

	#ifdef DOT
	printf("\t%u -- %u [label = \"e_%u,%u\"];\n", v1, v2, v1, v2);
	#endif
	g[v1 * N + v2] = g[v2 * N + v1] = e;
}

#ifdef TWITTER

__attribute__((always_inline)) inline
edge twitter(const char *filename, edge *g) {

	#define MAXLINE 1000
	static char line[MAXLINE];
	FILE *f = fopen(filename, "rt");
	fgets(line, MAXLINE, f);
	edge ne = atoi(line);

	for (edge i = 0; i < ne; i++) {
		fgets(line, MAXLINE, f);
		const agent v1 = atoi(line);
		fgets(line, MAXLINE, f);
		const agent v2 = atoi(line);
		createedge(g, v1, v2, N + i);
	}

	fclose(f);

	return ne;
}

#else

__attribute__((always_inline)) inline
edge scalefree(edge *g) {

	edge ne = 0;
	agent deg[N] = {0};

	for (agent i = 1; i <= M; i++) {
		for (agent j = 0; j < i; j++) {
			createedge(g, i, j, N + ne);
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
				createedge(g, i, q, N + ne);
				deg[i]++;
				deg[q]++;
				ne++;
			}
		}
	}

	return ne;
}

#endif

int main(int argc, char *argv[]) {

	unsigned seed = atoi(argv[1]);
	meter *sp = createsp(seed);

	// Create leaders array

	agent la[N] = {0};
	chunk l[C] = {0};

	for (agent i = 0; i < D; i++)
		la[i] = 1;

	shuffle(la, N, sizeof(agent));

	for (agent i = 0; i < N; i++)
		if (la[i]) SET(l, i);

	// Create/read graph

	init(seed);
	edge *g = (edge *)calloc(N * N, sizeof(edge));

	#ifdef DOT
	printf("graph G {\n");
	#endif
	#ifdef TWITTER
	twitter(argv[2], g);
	#else
	scalefree(g);
	#endif
	#ifdef DOT
	printf("}\n\n");
	#endif

	double *w = apeqis(g, srvalue, sp, l, K, MAXDRIVERS);

	free(sp);
	free(g);
	free(w);

	return 0;
}
