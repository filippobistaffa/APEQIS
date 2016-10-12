#include "apelib.h"

double *apeqis(const edge *g, value (*cf)(agent *, agent, void *),
	       void *data, const chunk *l, agent maxc, agent maxl) {

	chunk *tl;

	if (!l) {
		tl = (chunk *)malloc(sizeof(chunk) * _C);
		ONES(tl, _N, _C);
	}

	edge ne = 0;

	for (agent i = 0; i < _N; i++)
		for (agent j = i + 1; j < _N; j++)
			if (g[i * _N + j]) ne++;

	agent *adj = createadj<_N>(g, ne, l ? l : tl);

	#ifndef APE_SILENT
	puts("Creating model...");
	#endif

	#ifdef APE_DEBUG
	puts("\nAdjacency lists");
	for (agent i = 0; i < _N; i++)
		printbuf(adj + i * _N + 1, adj[i * _N]);
	puts("\nAdjacency matrix");
	for (agent i = 0; i < _N; i++)
		printbuf(g + i * _N, _N);
	puts("");
	#endif

	// Create constraints

	/*#ifndef APE_SILENT
	const value tv =
	#endif
	constraints(g, adj, l ? l : tl, cf, data, env, model, ea, da, maxc, maxl);*/

	// Create objective expression

	#ifndef APE_SILENT
	puts("Starting CUDA solver...\n");
	#endif

	/*double dif = 0;
	double difbuf[da.getSize()];

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
	for (agent i = 0; i < _N / 2; i++)
	#else
	for (agent i = 0; i < _N; i++)
	#endif
		topdif += difbuf[i];*/

	// Generate weights array

	double *w = (double *)malloc(sizeof(double) * (ne + _N));

	/*for (edge i = 0; i < ea.getSize(); i++) {
		w[i] = UNFEASIBLEVALUE;
	}*/

	// Print output

	#ifdef APE_CSV
	//printf("%u,%.2f,%.2f,%.2f,%.2f\n", _N, dif, (dif * 1E4) / tv, dif / da.getSize(), timer.getTime() * 1000);
	#endif

	#ifndef APE_SILENT
	/*puts("\nEdge values:");
	for (edge i = 0; i < ea.getSize(); i++)
		cout << ea[i].getName() << " = " << w[i] << endl;
	env.out() << "\nSolution elapsed time = " << timer.getTime() * 1000 << "ms" << endl;
	printf("Overall difference = %.2f\n", dif);
	printf("Percentage difference = %.2f%%\n", dif < EPSILON ? 0 : (dif * 100) / tv);
	#ifdef SINGLETONS
	printf("Average difference (excluding singletons) = %.2f\n", dif < EPSILON ? 0 : dif / (da.getSize() - _N));
	printf("Sum of the %u highest differences = %.2f\n", _N / 2, topdif);
	#else
	printf("Average difference = %.2f\n", dif / da.getSize());
	printf("Sum of the %u highest differences = %.2f\n", _N, topdif);
	#endif*/
	#endif

	if (!l) free(tl);
	free(adj);

	// Write output file

	#ifdef CFSS
	FILE *cfss = fopen(CFSS, "w+");
	for (agent i = 0; i < _N; i++)
		fprintf(cfss, "%s%f\n", GET(l, i) ? "*" : "", -w[i]);
	for (agent i = 0; i < _N; i++) {
		for (agent j = 0; j < adj[i * _N]; j++) {
			const agent k = adj[i * _N + j + 1];
			if (k > i) fprintf(cfss, "%u %u %f\n", i, k, -w[g[i * _N + k]]);
		}
	}
	fclose(cfss);
	#endif

	#ifdef APE_NOERROR
	assert(dif < EPSILON);
	#endif

	return w;
}
