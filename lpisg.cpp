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
void createedge(edge *g, agent *adj, agent v1, agent v2, edge e, IloEnv &env, IloFloatVarArray &ea) {

	//printf("%u -- %u\n", v1, v2);
	g[v1 * N + v2] = g[v2 * N + v1] = e;
	adj[v1 * N + (adj[v1 * N]++) + 1] = v2;
	adj[v2 * N + (adj[v2 * N]++) + 1] = v1;

	ostringstream ostr;
	ostr << "e_" << v1 << "," << v2;
	ea.add(IloFloatVar(env, -FLT_MAX, FLT_MAX, ostr.str().c_str()));
}

__attribute__((always_inline)) inline
edge scalefree(edge *g, agent *adj, const chunk *dr, IloEnv &env, IloFloatVarArray &ea) {

	edge ne = 0;
	agent deg[N] = {0};

	for (agent i = 1; i <= M; i++) {
		for (agent j = 0; j < i; j++) {
			createedge(g, adj, i, j, N + ne, env, ea);
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
				createedge(g, adj, i, q, N + ne, env, ea);
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

	// CPLEX environment and model

	IloEnv env;
	IloModel model(env);

	// Variables representing edge values

	IloFloatVarArray ea(env, N);
	IloFloatVarArray da(env);
	ostringstream ostr;

	for (agent i = 0; i < N; i++) {
		ostr << "e_" << i;
		ea[i] = IloFloatVar(env, -FLT_MAX, FLT_MAX, ostr.str().c_str());
		ostr.str("");
	}

	init(seed);
	edge *g = (edge *)calloc(N * N, sizeof(edge));
	agent *adj = (agent *)calloc(N * N, sizeof(agent));
	edge ne = scalefree(g, adj, dr, env, ea);

	#ifdef DEBUG
	printf("%u edges + %u autoedges\n", ne, N);
	puts("\nAdjacency lists");
	for (agent i = 0; i < N; i++)
		printbuf(adj + i * N + 1, adj[i * N]);
	puts("\nAdjacency matrix");
	for (agent i = 0; i < N; i++)
		printbuf(g + i * N, N);
	puts("");
	#endif

	// Create constraints

	constraints(g, adj, dr, sp, env, model, ea, da);

	// Create objective expression

	IloExpr expr(env);
	for (agent i = 0; i < da.getSize(); i++)
		expr += da[i];

	#ifdef DEBUG
	cout << expr << endl;
	#endif

	model.add(IloMinimize(env, expr));
	expr.end();

	IloCplex cplex(model);
	IloTimer timer(env);
	timer.start();

	#ifndef DEBUG
	cplex.setOut(env.getNullStream());
	#endif

	if (!cplex.solve()) {
		env.out() << "Unable to find a solution" << endl;
		exit(1);
	}

	timer.stop();
	IloNumArray eval(env);
	#ifdef DEBUG
	env.out() << "Solution status = " << cplex.getStatus() << endl;
	env.out() << "Elapsed time = " << timer.getTime() << endl;
	#endif
	env.out() << "Overall difference = " << cplex.getObjValue() << endl ;
	cplex.getValues(eval, ea);
	env.out() << "Edge values = " << eval << endl;
	env.end();
	free(adj);
	free(g);

	return 0;
}
