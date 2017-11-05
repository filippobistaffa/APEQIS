#include "apeqis.h"

__attribute__((always_inline)) inline
void createedge(edge *g, agent v1, agent v2, edge e) {

	#ifdef DOT
	printf("\t%u -- %u [label = \"e_%u,%u\"];\n", v1, v2, v1, v2);
	#endif
	g[v1 * _N + v2] = g[v2 * _N + v1] = e;
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
		createedge(g, v1, v2, _N + i);
	}

	fclose(f);

	return ne;
}

#else

__attribute__((always_inline)) inline
edge scalefree(edge *g) {

	edge ne = 0;
	agent deg[_N] = {0};

	for (agent i = 1; i <= _M; i++) {
		for (agent j = 0; j < i; j++) {
			createedge(g, i, j, _N + ne);
			deg[i]++;
			deg[j]++;
			ne++;
		}
	}

	chunk t[_C] = { 0 };
	chunk t1[_C] = { 0 };

	for (agent i = _M + 1; i < _N; i++) {
		ONES(t1, i, _C);
		MASKANDNOT(t, t1, t, _C);
		for (agent j = 0; j < _M; j++) {
			agent d = 0;
			for (agent h = 0; h < i; h++)
				if (!GET(t, h)) d += deg[h];
			if (d > 0) {
				int p = nextInt(d);
				agent q = 0;
				while (p >= 0) {
					if (!GET(t, q)) p = p - deg[q];
					q++;
				}
				q--;
				SET(t, q);
				createedge(g, i, q, _N + ne);
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

	agent la[_N] = {0};
	chunk l[_C] = {0};

	for (agent i = 0; i < _D; i++)
		la[i] = 1;

	shuffle(la, _N, sizeof(agent));

	for (agent i = 0; i < _N; i++)
		if (la[i]) SET(l, i);

	// Create/read graph

	init(seed);
	edge *g = (edge *)calloc(_N * _N, sizeof(edge));

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

	value *w = apeqis(g, srvalue, sp, l, K, MAXDRIVERS, argv[3], argv[4]);

	free(sp);
	free(g);
	if (w) free(w);

	return 0;
}
