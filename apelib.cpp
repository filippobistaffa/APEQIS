#include "apelib.h"

void count(agent *c, agent nl, const edge *g, const agent *adj, const chunk *l, void *data) {

	size_t *cnt = (size_t *)data;

	// Increase row counter
	cnt[0]++;

	// Increase non-zero element counter
	cnt[1] += *c;
	for (agent i = 0; i < *c; i++)
		for (agent j = i + 1; j < *c; j++)
			if (g[i * _N + j]) cnt[1]++;
}

void locations(agent *c, agent nl, const edge *g, const agent *adj, const chunk *l, void *data) {

	funcdata *fd = (funcdata *)data;
	printbuf(c + 1, *c, NULL, NULL, " = ");
	value cv = fd->cf(c, nl, fd->cfdata);
	printf("%.2f\n", cv);
	fd->tv += cv;
}

double *apeqis(const edge *g, value (*cf)(agent *, agent, void *),
	       void *cfdata, const chunk *l, agent maxc, agent maxl) {

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

	// Count rows and non-zero elements

	size_t cnt[2] = { 0, 0 };
	coalitions(g, count, cnt, K, l ? l : tl, MAXDRIVERS);
	printbuf(cnt, 2, "cnt");

	// #rows = #coalitions, #columns = #variables = #edges + #autoloops + dif

	sp_fmat *mat = new sp_fmat(cnt[0], ne + _N + 1);
	uvec *vals = new uvec(cnt[0] + cnt[1]);

	// Create sparse matrix

	funcdata *fd = (funcdata *)malloc(sizeof(funcdata));
	fd->locs = new umat(2, cnt[0] + cnt[1]);
	fd->cfdata = cfdata;
	fd->cf = cf;

	coalitions(g, locations, fd, K, l ? l : tl, MAXDRIVERS);

	delete fd->locs;
	delete vals;
	free(fd);

	#ifndef APE_SILENT
	puts("Starting CUDA solver...\n");
	#endif

	delete mat;

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
