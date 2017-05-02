#include "apelib.h"

__attribute__((always_inline)) inline
agent *createadj(const edge *g, edge ne, const chunk *l, IloEnv &env, IloFloatVarArray &ea) {

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

value resvalue(agent *c, agent nl, void *data) {

	map<vector<agent>, value> *resmap = (map<vector<agent>, value> *)data;
	vector<agent> key(c + 1, c + *c + 1);
	return resmap->at(key);
}

double *apeqis(const edge *g, value (*cf)(agent *, agent, void *),
	       void *data, const chunk *l, agent maxc, agent maxl, agent maxit) {

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

	agent *adj = createadj(g, ne, l ? l : tl, env, ea);

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

	map<vector<agent>, value> resmap;

	// Create constraints

	const value tv = constraints(g, adj, l ? l : tl, cf, data, env, model, ea, da, maxc, maxl, resmap);

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
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	IloTimer timer(env);
	timer.start();

	#ifdef APE_SILENT
	cplex.setOut(env.getNullStream());
	#endif

	#ifndef PARALLEL
	cplex.setParam(IloCplex::Threads, 1);
	#endif

	#ifdef TIME_LIMIT
	cplex.setParam(IloCplex::TiLim, TIME_LIMIT);
	#endif

	try {
		if (!cplex.solve()) {
			#ifndef APE_SILENT
			env.out() << "Unable to find a solution" << endl;
			#endif
			exit(EXIT_FAILURE);
		}
	}
	catch (IloCplex::Exception e) {
		#ifndef APE_SILENT
		env.out() << "An exception occurred" << endl;
		#endif
		exit(EXIT_FAILURE);
	}

	gettimeofday(&t2, NULL);
	timer.stop();
	double difbuf[da.getSize()];
	double dif = 0;

	#ifdef DIFFERENCES
	puts("\nDifferences:");
	#endif
	for (agent i = 0; i < da.getSize(); i++) {
		difbuf[i] = cplex.getValue(da[i]);
		dif += fabs(difbuf[i]);
		#ifdef DIFFERENCES
		cout << da[i].getName() << " = " << difbuf[i] << endl;
		#endif
	}

	// Update residuals map

	for(map<vector<agent>, value>::iterator it = resmap.begin(); it != resmap.end(); ++it)
		it->second = cplex.getValue(da[it->second]);

	#ifdef APE_DEBUG
	for(map<vector<agent>, value>::const_iterator it = resmap.begin(); it != resmap.end(); ++it) {
		printf("v(");
		printvec(it->first, NULL, NULL, ") = ");
		printf("%f\n", it->second);
	}
	#endif

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

	#ifdef APE_CSV
	printf("%.2f,%.2f,%.2f,%.2f,%.2f\n", dif, (dif * 100) / tv, dif / da.getSize(), timer.getTime(),
					     (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec);
	#endif

	#ifndef APE_SILENT
	puts("\nEdge values:");
	for (edge i = 0; i < ea.getSize(); i++)
		cout << ea[i].getName() << " = " << w[i] << endl;
	env.out() << "\nCPLEX elapsed time = " << timer.getTime() << endl;
	printf("Clock elapsed time = %f\n", (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec);
	printf("Overall difference = %.2f\n", dif);
	printf("Percentage difference = %.2f%%\n", dif < EPSILON ? 0 : (dif * 100) / tv);
	#ifdef SINGLETONS
	printf("Average difference (excluding singletons) = %.2f\n", dif < EPSILON ? 0 : dif / (da.getSize() - N));
	#else
	printf("Average difference = %.2f\n", dif / da.getSize());
	#endif
	#endif

	env.end();
	if (!l) free(tl);
	free(adj);

	#ifdef APE_NOERROR
	assert(dif < EPSILON);
	#endif

	// Recursion (if possible / necessary)

	if (maxit && fabs(tv - dif) > EPSILON && dif > THRESHOLD)
		apeqis(g, resvalue, &resmap, l, K, MAXDRIVERS, maxit - 1);

	return w;
}
