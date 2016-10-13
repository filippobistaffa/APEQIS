#include "apelib.h"

void count(agent *c, agent nl, const edge *g, const agent *adj, const chunk *l, void *data) {

	size_t *cnt = (size_t *)data;

	// Increase row counter
	cnt[0]++;

	// Increase non-zero element counter
	cnt[1] += *c;
	for (agent i = 0; i < *c; i++) {
		const agent v1 = c[i + 1];
		for (agent j = i + 1; j < *c; j++) {
			const agent v2 = c[j + 1];
			if (g[v1 * _N + v2]) cnt[1]++;
		}
	}
}

__attribute__((always_inline)) inline
void setlocation(agent i, agent j, size_t *idx, umat *locs) {

	(*locs)(0, *idx) = i;
	(*locs)(1, *idx) = j;
	(*idx)++;
}

void locations(agent *c, agent nl, const edge *g, const agent *adj, const chunk *l, void *data) {

	funcdata *fd = (funcdata *)data;
	value cv = fd->cf(c, nl, fd->cfdata);
	fd->b[fd->rowidx] = cv;
	fd->tv += cv;
	//printbuf(c + 1, *c, NULL, NULL, " = ");
	//printf("%.2f\n", cv);

	for (agent i = 0; i < *c; i++) {
		const agent v1 = c[i + 1];
		setlocation(fd->rowidx, v1, &(fd->locidx), fd->locs);
		for (agent j = i + 1; j < *c; j++) {
			const agent v2 = c[j + 1];
			if (g[v1 * _N + v2])
				setlocation(fd->rowidx, g[v1 * _N + v2], &(fd->locidx), fd->locs);
		}
	}

	fd->rowidx++;
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

	#ifndef APE_SILENT
	puts("Creating adjacendy lists...");
	#endif

	agent *adj = createadj<_N>(g, ne, l ? l : tl);

	#ifdef APE_DEBUG
	puts("\nAdjacency lists");
	for (agent i = 0; i < _N; i++)
		printbuf(adj + i * _N + 1, adj[i * _N]);
	puts("\nAdjacency matrix");
	for (agent i = 0; i < _N; i++)
		printbuf(g + i * _N, _N, NULL, "% 3u");
	puts("");
	#endif

	#ifndef APE_SILENT
	puts("Counting matrix non-zero elements...");
	#endif

	// Count rows and non-zero elements

	size_t cnt[2] = { 0, 0 };
	coalitions(g, count, cnt, K, l ? l : tl, MAXDRIVERS);

	// #rows = #coalitions, #columns = #variables = #edges + #autoloops + (#dif = #coalitions)
	const size_t nvals = cnt[1];
	const size_t nrows = cnt[0];
	const size_t ncols = ne + _N;

	#ifndef APE_SILENT
	printf("\nA\n%zu rows\n%zu columns\n%zu ones\n%zu bytes\n\n", nrows, ncols, nvals, sizeof(float) * nvals + sizeof(uword) * (nvals + ncols + 1));
	#endif

	fvec *vals = new fvec(nvals);
	vals->ones();

	// Create sparse matrix

	#ifndef APE_SILENT
	puts("Computing elements' locations..");
	#endif

	funcdata *fd = (funcdata *)malloc(sizeof(funcdata));
	value *b = (value *)malloc(sizeof(value) * nrows);
	fd->tv = 0;
	fd->b = b;

	fd->locs = new umat(2, nvals);
	fd->rowidx = 0;
	fd->locidx = 0;

	fd->cfdata = cfdata;
	fd->cf = cf;

	coalitions(g, locations, fd, K, l ? l : tl, MAXDRIVERS);

	#ifndef APE_SILENT
	puts("Creating sparse matrix...");
	#endif

	sp_fmat A(*(fd->locs), *vals);
	delete fd->locs;
	delete vals;
	free(fd);

	#if defined PRINTDENSE || defined PRINTCCS
	puts("");
	#endif

	#ifdef PRINTDENSE
	puts("A as dense matrix");
	fmat *dmat = new fmat(A);
	dmat->raw_print();
	delete dmat;
	puts("");
	#endif

	#ifdef PRINTCCS
	puts("A as CCS arrays");
	printbuf(A.values, nvals, "val");
	printbuf(A.row_indices, nvals, "row_idx");
	printbuf(A.col_ptrs, ncols + 1, "col_ptr");
	puts("");
	#endif

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

	free(b);

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
