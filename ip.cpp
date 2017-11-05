#include "ip.h"

typedef struct {
	const std::vector<value> *difpfx;
	value minval;
} mindata;

__attribute__((always_inline)) inline
void initialise(int *a, unsigned n, unsigned m) {

	a[0] = n - m + 1;

	for (unsigned i = 1; i < m; ++i)
		a[i] = 1;

	a[m] = -1;
}

__attribute__((always_inline)) inline
void printpart(int *a, unsigned n, void *data) {

	printbuf(a, n);
}

__attribute__((always_inline)) inline
void computehist(const int *a, unsigned n, unsigned *hist) {

	for (unsigned i = 0; i < n; ++i)
		hist[a[i]]++;
}

template <typename type>
__attribute__((always_inline)) inline
type reducebuf(const type *buf, unsigned n) {

	type ret = 0;

	for (unsigned i = 0; i < n; ++i)
		ret += buf[i];

	return ret;
}

//#include <assert.h>

__attribute__((always_inline)) inline
void minvaluepart(int *a, unsigned n, void *data) {

	mindata *md = (mindata *)data;
	unsigned hist[K + 1] = { 0 };
	computehist(a, n, hist);
	value val = 0;

	// If the number of cars exceeds the number of drivers, the integer partition is unfeasible
	if (reducebuf(hist + 2, K - 1) > _D)
		return;

	for (unsigned k = 1; k <= K; ++k) {
		//printf("k = %u, hist[%u] = %u, md->difpfx[%u].size() = %lu\n", k, k, hist[k], k, md->difpfx[k].size());
		if (hist[k] > md->difpfx[k].size()) return;
		if (hist[k] == 0) continue;
		val += md->difpfx[k][hist[k] - 1];
	}

	//printbuf(hist, K + 1, "hist");
	//printbuf(a, n, NULL, NULL, " = ");
	//printf("%f\n", val);
	if (val < md->minval) md->minval = val;
}

__attribute__((always_inline)) inline
void conjugate(const int *a, unsigned m, void (*cf)(int *, unsigned, void *), void *data) {

	int c[a[0]];
	unsigned i = 0;

	while (1) {
		c[i++] = m;
		while (i >= a[m - 1]) {
			m--;
			if (m == 0) goto exit;
		}
	}

exit:	cf(c, i, data);
}

__attribute__((always_inline)) inline
size_t enumerate(int *a, unsigned m, void (*pf)(int *, unsigned, void *), void (*cf)(int *, unsigned, void *), void *data) {

	// The algorithm follows Knuth v4 fasc3 p38 in rough outline;
	// Knuth credits it to Hindenburg, 1779.

	size_t count = 0;

	while (1) {

		count++;
		if (pf) pf(a, m, data);
		if (cf) conjugate(a, m, cf, data);
		if (m == 1) return count;
		if (a[0] - 1 > a[1]) {
			a[0]--;
			a[1]++;
			continue;
		}

		int j = 2;
		int s = a[0] + a[1] - 1;

		while (j < m && a[j] >= a[0] - 1) {
		    s += a[j];
		    j++;
		}

		if (j >= m) return count;
		int x = a[j] + 1;
		a[j] = x;
		j--;

		while (j > 0) {
		    a[j] = x;
		    s -= x;
		    j--;
		}

		a[0] = s;
	}
}

value minpartition(const std::vector<value> *difpfx) {

	int a[K + 1];
	size_t count = 0;
	mindata md = { .difpfx = difpfx, .minval = FLT_MAX };

	for (unsigned m = 1; m <= K; ++m) {
		initialise(a, _N, m);
		size_t c = enumerate(a, m, NULL, minvaluepart, &md);
		//printf("%zu partition(s) with min value = %u\n", c, m);
		count += c;
	}

	printf("%zu total integer partition(s)\n", count);
	return md.minval;
}
