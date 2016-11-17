#include "ip.h"

__attribute__((always_inline)) inline
void initialise(int *a) {

	a[0] = _N - K + 1;

	for (id i = 1; i < K; ++i)
		a[i] = 1;

	a[K] = -1;
}

void printpart(int *a, void *data) {

	printbuf(a, K, NULL, "%2d");
}

__attribute__((always_inline)) inline
size_t enumerate(int *a, void (*pf)(int *, void *), void *data) {

	// The algorithm follows Knuth v4 fasc3 p38 in rough outline;
	// Knuth credits it to Hindenburg, 1779.

	size_t count = 0;

	while (1) {
		count ++;
		pf(a, data);
		if (a[0] - 1 > a[1]) {
			a[0]--;
			a[1]++;
			continue;
		}
		id j = 2;
		id s = a[0] + a[1] - 1;
		while (j < K && a[j] >= a[0] - 1) {
		    s += a[j];
		    j++;
		}
		if (j >= K) return count;
		id x = a[j] + 1;
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

	initialise(a);
	printf("%zu partitions\n", enumerate(a, printpart, NULL));

	return 0;
}
