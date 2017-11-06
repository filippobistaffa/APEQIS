#include "apelib.h"

template <typename type>
__attribute__((always_inline)) inline
void exclprefixsum(const type *hi, type *ho, unsigned hn) {

	if (hn) {
		ho[0] = 0;
		for (unsigned i = 1; i < hn; i++)
			ho[i] = hi[i - 1] + ho[i - 1];
	}
}

template <typename type>
__attribute__((always_inline)) inline
void inplaceinclpfxsum(vector<type>& vec) {

	for (id i = 1; i < vec.size(); ++i)
		vec[i] += vec[i - 1];
}

template <typename type>
void count(agent *c, agent nl, const edge *g, const agent *adj, const chunk *l, type *data) {

	#ifdef SINGLETONS
	if (*c == 1) return;
	#endif

	size_t *cnt = (size_t *)data;

	// Increase row counter
	cnt[0]++;

	// Increase non-zero element counter
	#ifndef SINGLETONS
	cnt[1] += *c;
	#endif
	for (agent i = 0; i < *c; i++) {
		const agent v1 = c[i + 1];
		for (agent j = i + 1; j < *c; j++) {
			const agent v2 = c[j + 1];
			if (g[v1 * _N + v2]) cnt[1]++;
		}
	}
}

__attribute__((always_inline)) inline
void setlocation(agent i, agent j, size_t *idx, size_t rowoff, size_t valoff, umat *locs) {

	(*locs)(0, valoff + *idx) = rowoff + i;
	(*locs)(1, valoff + *idx) = j;
	(*idx)++;
}

template <typename type>
void locations(agent *c, agent nl, const edge *g, const agent *adj, const chunk *l, type *data) {

	funcdata *fd = (funcdata *)data;
	value cv = fd->cf(c, nl, fd->cfdata);
	fd->tv += cv;
	/*#pragma omp critical
	{
		printf("%u: ", omp_get_thread_num());
		printbuf(c + 1, *c, NULL, NULL, " = ");
		printf("%.2f\n", cv);
	}*/

	#ifdef SINGLETONS
	if (*c == 1) return;
	#endif

	#ifdef WEIGHT
	if (*c == 1) fd->sl[c[1]] = fd->valoff + fd->locidx;
	#endif

	fd->b[fd->rowidx] = cv;
	fd->size[fd->rowidx] = *c;

	for (agent i = 0; i < *c; i++) {
		const agent v1 = c[i + 1];
		#ifdef SINGLETONS
		fd->b[fd->rowidx] -= fd->s[v1];
		#else
		setlocation(fd->rowidx, v1, &(fd->locidx), fd->rowoff, fd->valoff, fd->locs);
		#endif
		for (agent j = i + 1; j < *c; j++) {
			const agent v2 = c[j + 1];
			if (g[v1 * _N + v2])
				#ifndef SINGLETONS
				setlocation(fd->rowidx, g[v1 * _N + v2], &(fd->locidx), fd->rowoff, fd->valoff, fd->locs);
				#else
				setlocation(fd->rowidx, g[v1 * _N + v2] - _N, &(fd->locidx), fd->rowoff, fd->valoff, fd->locs);
				#endif
		}
	}

	fd->rowidx++;
}

value *apeqis(const edge *g, value (*cf)(agent *, agent, void *),
	      void *cfdata, const chunk *l, agent maxc, agent maxl,
	      char *cfssfilename, char *resfilename) {

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
	agent a[2 * ne];
	for (agent i = 0; i < _N; i++)
		for (agent j = i + 1; j < _N; j++)
			if (g[i * _N + j]) {
				X(a, g[i * _N + j] - _N) = i;
				Y(a, g[i * _N + j] - _N) = j;
			}

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

	// Count rows and non-zero elements

	#ifndef APE_SILENT
	puts("Counting matrix non-zero elements...");
	#endif

	size_t *cnt[_T];

	for (agent t = 0; t < _T; ++t)
		cnt[t] = (size_t *)calloc(_T, sizeof(size_t));

	#ifdef PARALLEL
	#ifndef APE_SILENT
	printf("Using %u threads...\n", _T);
	#endif
	parcoalitions(g, count, cnt, K, l ? l : tl, MAXDRIVERS);
	#else
	coalitions(g, count, cnt[0], K, l ? l : tl, MAXDRIVERS);
	#endif

	size_t valcnt[_T], rowcnt[_T];
	size_t nvals = 0, nrows = 0;

	for (agent t = 0; t < _T; ++t) {
		nrows += rowcnt[t] = cnt[t][0];
		nvals += valcnt[t] = cnt[t][1];
	}

	for (agent t = 0; t < _T; ++t)
		free(cnt[t]);

	size_t valpfx[_T], rowpfx[_T];
	exclprefixsum(rowcnt, rowpfx, _T);
	exclprefixsum(valcnt, valpfx, _T);

	// #rows = #coalitions, #columns = #variables = #edges + #autoloops (only if SINGLETONS is not defined)

	#ifdef SINGLETONS
	const size_t ncols = ne;
	#else
	const size_t ncols = ne + _N;
	#endif

	#ifndef APE_SILENT
	printf("\nA\n%zu rows\n%zu columns\n%zu ones\n%zu bytes\n\n", nrows, ncols, nvals,
	       sizeof(float) * nvals + sizeof(uword) * (nvals + ncols + 1));
	#endif

	#ifdef SINGLETONS
	value *w = (value *)malloc(sizeof(value) * (ncols + _N));
	#else
	value *w = (value *)malloc(sizeof(value) * ncols);
	#endif

	#ifdef SINGLETONS
	for (agent i = 0; i < _N; i++) {
		agent c[] = { 1, i };
		w[i] = cf(c, GET(l ? l : tl, i), cfdata);
	}
	#endif

	// Create sparse matrix

	#ifndef APE_SILENT
	puts("Computing elements' locations..");
	#endif

	funcdata *fd[_T];
	value *b = (value *)malloc(sizeof(value) * nrows);
	id *size = (id *)malloc(sizeof(id) * nrows);
	umat *locs = new umat(2, nvals);
	#ifdef WEIGHT
	size_t sl[_N];
	#endif

	for (agent t = 0; t < _T; ++t) {
		fd[t] = (funcdata *)malloc(sizeof(funcdata));
		fd[t]->size = size + rowpfx[t];
		fd[t]->b = b + rowpfx[t];
		fd[t]->tv = 0;
		#ifdef SINGLETONS
		fd[t]->s = w;
		#endif
		#ifdef WEIGHT
		fd[t]->sl = sl;
		#endif
		fd[t]->locs = locs;
		fd[t]->valoff = valpfx[t];
		fd[t]->rowoff = rowpfx[t];
		fd[t]->rowidx = 0;
		fd[t]->locidx = 0;
		fd[t]->cfdata = cfdata;
		fd[t]->cf = cf;
	}

	#ifdef PARALLEL
	#ifndef APE_SILENT
	printf("Using %u threads...\n", _T);
	#endif
	parcoalitions(g, locations, fd, K, l ? l : tl, MAXDRIVERS);
	#else
	coalitions(g, locations, fd[0], K, l ? l : tl, MAXDRIVERS);
	#endif

	value tv = 0;

	for (agent t = 0; t < _T; ++t) {
		tv += fd[t]->tv;
		free(fd[t]);
	}

	#ifndef APE_SILENT
	puts("Creating sparse matrix...");
	#endif

	fvec *vals = new fvec(nvals);
	vals->ones();

	#ifdef WEIGHT
	for (size_t i = 0; i < nrows; i++)
		if (size[i] == 1)
			b[i] *= sqrt(WEIGHT);
	for (size_t i = 0; i < _N; i++)
		vals->at(sl[i]) *= sqrt(WEIGHT);
	#endif

	sp_fmat A(*locs, *vals);

	// Manually set A size if necessary
	if (ncols != A.n_cols)
		A.resize(nrows, ncols);

	delete locs;
	delete vals;

	#if defined PRINTDENSE || defined PRINTCCS || defined PRINTB
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

	#ifdef PRINTB
	printbuf(b, nrows, "b");
	puts("");
	#endif

	#ifndef APE_SILENT
	puts("Starting CUDA solver...\n");
	const bool quiet = false;
	#else
	const bool quiet = true;
	#endif

	const unsigned *ptr, *idx;
	#if (__cplusplus >= 201103L)
	unsigned *uptr = (unsigned *)malloc(sizeof(unsigned) * (ncols + 1));
	unsigned *uidx = (unsigned *)malloc(sizeof(unsigned) * nvals);
	for (size_t i = 0; i < ncols + 1; ++i) uptr[i] = A.col_ptrs[i];
	for (size_t i = 0; i < nvals; ++i) uidx[i] = A.row_indices[i];
	ptr = uptr;
	idx = uidx;
	#else
	ptr = A.col_ptrs;
	idx = A.row_indices;
	#endif

	float rt;
	#ifdef SINGLETONS
	unsigned rc = cudacgls(A.values, ptr, idx, nrows, ncols, nvals, b, w + _N, &rt, quiet);
	#else
	unsigned rc = cudacgls(A.values, ptr, idx, nrows, ncols, nvals, b, w, &rt, quiet);
	#endif

	#if (__cplusplus >= 201103L)
	free(uptr);
	free(uidx);
	#endif

	#ifdef APE_UNFEASIBLE
	for (id i = 0; i < ncols; ++i)
		if (A.col(i).max() == 0)
			w[i + _N] = UNFEASIBLEVALUE;
	#endif

	value dif = 0, difsq = 0;
	value *difbuf = (value *)malloc(sizeof(value) * nrows);
	vector<value> difs[K + 1];

	if (!rc) {

		#ifdef RESIDUAL
		FILE *res = fopen(resfilename, "w+");
		#ifdef SINGLETONS
		for (id i = 0; i < _N; i++) {
			fprintf(res, "%u %f\n", i, 0.0);
		}
		#endif
		#endif

		#ifdef DIFFERENCES
		puts("Differences:");
		#endif
		for (id i = 0; i < nrows; i++) {

			#ifdef WEIGHT
			difbuf[i] = b[i] / (size[i] == 1 ? WEIGHT : 1);
			#else
			difbuf[i] = b[i];
			#endif

			#ifdef RESIDUAL
			#ifdef SINGLETONS
			std::set<agent> coal;
			for (id j = 0; j < ncols; ++j) {
				if (A(i, j) == 1) {
					//printf("%u %u %u\n", j, X(a, j), Y(a, j));
					coal.insert(X(a, j));
					coal.insert(Y(a, j));
				}
			}
			//print_it(coal.begin(), coal.end());
			for (auto it = coal.begin(); it != coal.end(); ++it) {
				fprintf(res, "%u ", *it);
			}
			#else
			for (id j = 0; j < _N; ++j) {
				if (A(i, j) == 1) {
					fprintf(res, "%u ", j);
				}
			}
			#endif
			fprintf(res, "%f\n", difbuf[i]);
			#endif

			difs[size[i]].push_back(difbuf[i]);
			dif += fabs(difbuf[i]);
			difsq += difbuf[i] * difbuf[i];
			#ifdef DIFFERENCES
			cout << "d_" << i << " = " << difbuf[i] << endl;
			#endif
		}
		#ifdef DIFFERENCES
		puts("");
		#endif

		#ifdef RESIDUAL
		fclose(res);
		#endif

		#ifdef SINGLETONS
		for (agent i = 0; i < _N; ++i)
			difs[1].push_back(0);
		#endif
	}

	for (id k = 1; k <= K; ++k) {
		std::sort(difs[k].begin(), difs[k].end());
		//printvec(difs[k]);
	}

	for (id k = 1; k <= K; ++k) {
		inplaceinclpfxsum(difs[k]);
		//printvec(difs[k]);
	}

	free(difbuf);
	free(size);
	free(b);

	if (!rc) {

		// Print output

		#ifdef APE_CSV
		printf("%f,%f,%f,%f,%f,%f,%f\n",
		       dif, (dif * 1e2) / tv, dif / nrows, difsq, (difsq * 1e2) / tv, difsq / nrows, rt / 1e3);
		#endif

		#ifndef APE_SILENT
		puts("Edge values:");
		for (agent i = 0; i < _N; i++)
			cout << "e_" << i << " = " << w[i] << endl;
		for (edge i = _N; i < _N + ne; i++)
			cout << "e_" << X(a, i - _N) << "," << Y(a, i - _N) << " = " << w[i] << endl;
		cout << "\nSolution elapsed time = " << rt << "ms" << endl;
		printf("Overall difference = %.2f\n", dif);
		printf("Percentage difference = %.2f%%\n", dif < EPSILON ? 0 : (dif * 100) / tv);
		#ifdef SINGLETONS
		printf("Average difference (excluding singletons) = %.2f\n", dif < EPSILON ? 0 : dif / nrows);
		#else
		printf("Average difference = %.2f\n", dif / nrows);
		#endif
		printf("Minimum error w.r.t. integer partitions = %.2f\n", minpartition(difs));
		#endif
	}

	// Write output file

	if (!rc) {

		#ifdef CFSS
		FILE *cfss = fopen(cfssfilename, "w+");
		for (agent i = 0; i < _N; i++)
			fprintf(cfss, "%s%f\n", GET(l ? l : tl, i) ? "*" : "", -w[i]);
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
	}

	if (!l) free(tl);
	free(adj);

	if (rc) {
		fprintf(stderr, RED("ERROR: exit code of CGLS = %u\n"), rc);
		return NULL;
	}

	return w;
}
