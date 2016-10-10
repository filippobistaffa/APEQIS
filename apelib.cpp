#include "apelib.h"

__attribute__((always_inline)) inline
agent *creteadj(const edge *g, edge ne, const chunk *l, IloEnv &env, IloFloatVarArray &ea) {

	agent *adj = (agent *)calloc(N * N, sizeof(agent));
	agent ab[2 * ne];

	for (agent v1 = 0; v1 < N; v1++)
		for (agent v2 = v1 + 1; v2 < N; v2++) {
			const edge e = g[v1 * N + v2];
			if (e) {
				X(ab, e - N) = v1;
				Y(ab, e - N) = v2;
			}
		}

	agent *a = ab;

	do {
		adj[a[0] * N + (adj[a[0] * N]++) + 1] = a[1];
		adj[a[1] * N + (adj[a[1] * N]++) + 1] = a[0];
		ostringstream ostr;
		ostr << "e_" << a[0] << "," << a[1];
		ea.add(IloFloatVar(env, MINEDGEVALUE, FLT_MAX, ostr.str().c_str()));
		a += 2;
	} while (--ne);

	for (agent i = 0; i < N; i++)
		QSORT(agent, adj + i * N + 1, adj[i * N], LTL);

	return adj;
}

double *apeqis(const edge *g, value (*cf)(agent *, agent, void *),
	       void *data, const chunk *l, agent maxc, agent maxl) {

	chunk *tl;

	if (!l) {
		tl = (chunk *)malloc(sizeof(chunk) * C);
		ONES(tl, N, C);
	}

	IloEnv env;
	IloFloatVarArray ea(env, N);
	IloFloatVarArray da(env);

	// Cplex model
	IloModel model(env);

	// Variables representing edge values
	ostringstream ostr;

	for (agent i = 0; i < N; i++) {
		ostr << "e_" << i;
		ea[i] = IloFloatVar(env, MINEDGEVALUE, FLT_MAX, ostr.str().c_str());
		ostr.str("");
	}

	edge ne = 0;

	for (agent i = 0; i < N; i++)
		for (agent j = i + 1; j < N; j++)
			if (g[i * N + j]) ne++;

	agent *adj = creteadj(g, ne, l ? l : tl, env, ea);

	#ifndef APE_SILENT
	puts("Creating model...");
	#endif

	#ifdef APE_DEBUG
	puts("\nAdjacency lists");
	for (agent i = 0; i < N; i++)
		printbuf(adj + i * N + 1, adj[i * N]);
	puts("\nAdjacency matrix");
	for (agent i = 0; i < N; i++)
		printbuf(g + i * N, N);
	puts("");
	#endif

	// Create constraints

	#ifndef APE_SILENT
	const value tv =
	#endif
	constraints(g, adj, l ? l : tl, cf, data, env, model, ea, da, maxc, maxl);

	// Create objective expression

	IloExpr expr(env);
	for (agent i = 0; i < da.getSize(); i++)
		#ifdef LSE
		expr += da[i] * da[i];
		#else
		expr += da[i];
		#endif

	#ifdef APE_DEBUG
	cout << expr << endl << endl;
	#endif

	model.add(IloMinimize(env, expr));
	expr.end();

	#ifndef APE_SILENT
	puts("Starting CPLEX...\n");
	#endif

	IloCplex cplex(model);
	IloTimer timer(env);
	timer.start();

	#ifdef APE_SILENT
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

	// Generate weights array

	double *w = (double *)malloc(sizeof(double) * ea.getSize());

	for (edge i = 0; i < ea.getSize(); i++) {
		try { w[i] = cplex.getValue(ea[i]); }
		catch (IloException& e) {
			w[i] = UNFEASIBLEVALUE;
			e.end();
		}
	}

	// Print output

	#ifdef CSV
	printf("%u,%.2f,%.2f,%.2f,%.2f\n", N, dif, (dif * 1E4) / tv, dif / da.getSize(), timer.getTime() * 1000);
	#endif

	#ifndef APE_SILENT
	puts("\nEdge values:");
	for (edge i = 0; i < ea.getSize(); i++)
		cout << ea[i].getName() << " = " << w[i] << endl;
	env.out() << "\nSolution elapsed time = " << timer.getTime() * 1000 << "ms" << endl;
	printf("Overall difference = %.2f\n", dif);
	printf("Percentage difference = %.2f%%\n", dif < EPSILON ? 0 : (dif * 100) / tv);
	#ifdef SINGLETONS
	printf("Average difference (excluding singletons) = %.2f\n", dif < EPSILON ? 0 : dif / (da.getSize() - N));
	printf("Sum of the %u highest differences = %.2f\n", N / 2, topdif);
	#else
	printf("Average difference = %.2f\n", dif / da.getSize());
	printf("Sum of the %u highest differences = %.2f\n", N, topdif);
	#endif
	#endif

	env.end();
	if (!l) free(tl);
	free(adj);

	// Write output file

	#ifdef CFSS
	FILE *cfss = fopen(CFSS, "w+");
	for (agent i = 0; i < N; i++)
		fprintf(cfss, "%s%f\n", GET(l, i) ? "*" : "", -w[i]);
	for (agent i = 0; i < N; i++) {
		for (agent j = 0; j < adj[i * N]; j++) {
			const agent k = adj[i * N + j + 1];
			if (k > i) fprintf(cfss, "%u %u %f\n", i, k, -w[g[i * N + k]]);
		}
	}
	fclose(cfss);
	#endif

	#ifdef APE_NOERROR
	assert(dif < EPSILON);
	#endif

	return w;
}
