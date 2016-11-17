#include "ip.h"

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

int main(int argc, char *argv[]) {

	int a[K + 1];
	size_t count = 0;

	for (unsigned m = 1; m <= K; ++m) {
		initialise(a, _N, m);
		size_t c = enumerate(a, m, NULL, printpart, NULL);
		printf("%zu partition(s) for m = %u\n", c, m);
		count += c;
	}

	printf("%zu total partition(s)\n", count);

	return 0;
}
