#include "apeqis.h"

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

__attribute__((always_inline)) inline
void createedge(agent *adj, agent v1, agent v2, IloEnv &env, IloFloatVarArray &ea) {

	#ifdef DOT
	printf("\t%u -- %u [label = \"e_%u,%u\"];\n", v1, v2, v1, v2);
	#endif
	adj[v1 * N + (adj[v1 * N]++) + 1] = v2;
	adj[v2 * N + (adj[v2 * N]++) + 1] = v1;

	ostringstream ostr;
	ostr << "e_" << v1 << "," << v2;
	ea.add(IloFloatVar(env, MINEDGEVALUE, FLT_MAX, ostr.str().c_str()));
}

#ifdef TWITTER

__attribute__((always_inline)) inline
edge twitter(const char *filename, agent *adj, const chunk *dr, IloEnv &env, IloFloatVarArray &ea) {

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
		createedge(adj, v1, v2, env, ea);
	}

	fclose(f);

	for (agent i = 0; i < N; i++)
		QSORT(agent, adj + i * N + 1, adj[i * N], LTDR);

	return ne;
}

#else

__attribute__((always_inline)) inline
edge scalefree(agent *adj, const chunk *dr, IloEnv &env, IloFloatVarArray &ea) {

	edge ne = 0;
	agent deg[N] = {0};

	for (agent i = 1; i <= M; i++) {
		for (agent j = 0; j < i; j++) {
			createedge(adj, i, j, env, ea);
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
				createedge(adj, i, q, env, ea);
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

edge *createg(const agent *adj) {

	edge e = N, *g = (edge *)calloc(N * N, sizeof(edge));

	for (agent v1 = 0; v1 < N; v1++)
		for (agent i = 0; i < adj[v1 * N]; i++) {
			const agent v2 = adj[v1 * N + i + 1];
			if (v1 > v2) g[v1 * N + v2] = e++;
		}

	return g;
}

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
		ea[i] = IloFloatVar(env, MINEDGEVALUE, FLT_MAX, ostr.str().c_str());
		ostr.str("");
	}

	init(seed);
	agent *adj = (agent *)calloc(N * N, sizeof(agent));

	#ifdef DOT
	printf("graph G {\n");
	#endif
	#ifdef TWITTER
	twitter(argv[2], adj, dr, env, ea);
	#else
	scalefree(adj, dr, env, ea);
	#endif
	#ifdef DOT
	printf("}\n\n");
	#endif

	edge *g = createg(adj);

	#ifndef CSV
	puts("Creating model...");
	#endif

	#ifdef DEBUG
	puts("\nAdjacency lists");
	for (agent i = 0; i < N; i++)
		printbuf(adj + i * N + 1, adj[i * N]);
	puts("\nAdjacency matrix");
	for (agent i = 0; i < N; i++)
		printbuf(g + i * N, N);
	puts("");
	#endif

	// Create constraints

	const penny tv = constraints(g, adj, dr, sp, env, model, ea, da);

	// Create individual rationality constraints

	#ifdef INDIVIDUALLYRATIONAL
	for (agent i = 0; i < N; i++) {
		IloExpr expr(env);
		for (agent j = 0; j < N; j++) {
			const edge e = g[i * N + j];
			if (e) expr += 0.5 * ea[e];
		}
		#ifdef DEBUG
		cout << expr << endl;
		#endif
		model.add(expr <= 0);
		expr.end();
	}
	#endif

	// Create objective expression

	IloExpr expr(env);
	for (agent i = 0; i < da.getSize(); i++)
		#ifdef LSE
		expr += da[i] * da[i];
		#else
		expr += da[i];
		#endif

	#ifdef DEBUG
	cout << expr << endl << endl;
	#endif

	model.add(IloMinimize(env, expr));
	expr.end();

	#ifndef CSV
	puts("Starting CPLEX...\n");
	#endif

	IloCplex cplex(model);
	IloTimer timer(env);
	timer.start();

	#ifdef CSV
	cplex.setOut(env.getNullStream());
	#endif

	if (!cplex.solve()) {
		env.out() << "Unable to find a solution" << endl;
		exit(1);
	}

	timer.stop();
	double difbuf[da.getSize()];
	double dif = 0;

	#ifdef DIFFERENCES
	puts("\nDifferences:");
	#endif
	for (agent i = 0; i < da.getSize(); i++) {
		difbuf[i] = cplex.getValue(da[i]);
		dif += difbuf[i];
		#ifdef DIFFERENCES
		cout << da[i].getName() << " = " << difbuf[i] << endl;
		#endif
	}

	QSORT(double, difbuf, da.getSize(), GT);
	double topdif = 0;

	#ifdef SINGLETONS
	for (agent i = 0; i < N / 2; i++)
	#else
	for (agent i = 0; i < N; i++)
	#endif
		topdif += difbuf[i];

	// Print output

	#ifdef CSV
	printf("%u,%u,%u,%.2f,%.2f,%.2f,%.2f\n",
	       N, DRIVERSPERC, seed, dif, (dif * 1E4) / tv, dif / da.getSize(), timer.getTime() * 1000);
	#else
	puts("\nEdge values:");
	for (edge i = 0; i < ea.getSize(); i++) {
		try {
			const double val = cplex.getValue(ea[i]);
			cout << ea[i].getName() << " = " << val << endl;
		}
		catch (IloException& e) {
			cout << ea[i].getName() << " never occurs in a feasible coalition" << endl;
			e.end();
		}
	}

	env.out() << "\nSolution elapsed time = " << timer.getTime() * 1000 << "ms" << endl;
	printf("Overall difference = %.2f\n", dif);
	printf("Percentage difference = %.2f%%\n", (dif * 1E4) / tv);
	#ifdef SINGLETONS
	printf("Average difference (excluding singletons) = %.2f\n", dif / (da.getSize() - N));
	printf("Sum of the %u highest differences = %.2f\n", N / 2, topdif);
	#else
	printf("Average difference = %.2f\n", dif / da.getSize());
	printf("Sum of the %u highest differences = %.2f\n", N, topdif);
	#endif
	#endif

	// Write output file

	#ifdef CFSS
	FILE *cfss = fopen(CFSS, "w+");
	for (agent i = 0; i < N; i++)
		fprintf(cfss, "%s%f\n", GET(dr, i) ? "*" : "", -cplex.getValue(ea[i]));
	for (agent i = 0; i < N; i++) {
		for (agent j = 0; j < adj[i * N]; j++) {
			const agent k = adj[i * N + j + 1];
			if (k > i) {
				double val = 0;
				try { val = -cplex.getValue(ea[g[i * N + k]]); }
				catch (IloException& e) {}
				fprintf(cfss, "%u %u %f\n", i, k, val);
			}
		}
	}
	fclose(cfss);
	#endif

	// Compute Shapley values

	#ifdef SHAPLEY
	double sv[N];
	double sing[N];
	bool ir = true;

	for (agent i = 0; i < N; i++) {
		sing[i] = sv[i] = cplex.getValue(ea[i]);
		for (agent j = 0; j < N; j++) {
			const edge e = g[i * N + j];
			try {
				if (e) sv[i] += cplex.getValue(ea[e]) / 2;
			}
			catch (IloException& e) {
				e.end();
			}
		}
		ir &= sing[i] + EPSILON >= sv[i];
	}

	printbuf(sv, N, "Shapley values");
	printbuf(sing, N, "Singleton values");
	if (ir) puts("Payments are individually rational");
	#endif

	env.end();
	free(adj);
	free(g);

	return 0;
}
