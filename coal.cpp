#include "coal.h"

typedef int16_t sign;
//static size_t bcm[(_N + 1) * (_N + 1)], pm[_N * _N];

/*#define P(_s, _i) (pm[(_s) * _N + (_i)])
#define BC(_n, _m) (bcm[(_n) * (_N + 1) + (_m)])

void filltables() {

	for (agent i = 0; i <= _N; i++) BC(i, 0) = BC(i, i) = 1ULL;
	for (agent i = 1; i <= _N; i++) for (agent j = 1; j < i; j++) BC(i, j) = BC(i - 1, j - 1) + BC(i - 1, j);
	for (agent i = 1; i < _N; i++) P(i, 1) = 1ULL;
	for (agent i = 2; i < _N; i++) P(1, i) = i;
	for (agent i = 2; i < _N; i++) for (agent j = 2; j < _N; j++) P(i, j) = P(i - 1, j) + P(i, j - 1);
}*/

__attribute__((always_inline)) inline
sign twiddle(sign *x, sign *y, sign *z, sign *p) {

	register sign i, j = 1, k;
	while (p[j] <= 0) j++;

	if (p[j - 1] == 0) {

		for (i = j - 1; i != 1; i--) p[i] = -1;

		p[j] = 0;
		*x = *z = 0;
		p[1] = 1;
		*y = j - 1;
	}
	else {
		if (j > 1) p[j - 1] = 0;

		do j++;
		while (p[j] > 0);

		k = j - 1;
		i = j;

		while (p[i] == 0) p[i++] = -1;

		if (p[i] == -1) {

			p[i] = p[k];
			*z = p[k] - 1;
			*x = i - 1;
			*y = k - 1;
			p[k] = -1;
		}
		else {
			if (i == p[0]) return 1;
			else {
				p[j] = p[i];
				*z = p[i] - 1;
				p[i] = 0;
				*x = j - 1;
				*y = i - 1;
			}
		}
	}

	return 0;
}

__attribute__((always_inline)) inline
void inittwiddle(sign m, sign n, sign *p) {

	sign i;
	p[0] = n + 1;

	for (i = 1; i != n - m + 1; i++) p[i] = 0;

	while (i != n + 1) {
		p[i] = i + m - n;
		i++;
	}

	p[n + 1] = -2;
	if (m == 0) p[1] = 1;
}

__attribute__((always_inline)) inline
void neighbours(const agent *f, agent m, const agent *adj, agent *n, const chunk *l) {

	if (m) {
		agent t[_N + 1];
		memcpy(n, adj + *f * _N, sizeof(agent) * (adj[*f * _N] + 1));
		f++;

		while (--m) {
			unionsorted(n + 1, *n, adj + *f * _N + 1, adj[*f * _N], t + 1, t, l);
			memcpy(n, t, sizeof(agent) * (*t + 1));
			f++;
		}
	}
	else *n = 0;
}

__attribute__((always_inline)) inline
void nbar(const agent *f, agent n, const agent *r, const agent *ruf, const agent *adj, agent *nb, const chunk *l) {

	agent a[_N + 1], b[_N + 1];
	neighbours(f, n, adj, a, l);
	agent i = 0;
	while (i < *a && LEL(a + i + 1, ruf + 1)) i++;
	memmove(a + 1, a + i + 1, sizeof(agent) * (*a - i));
	*a -= i;
	neighbours(r + 1, *r, adj, nb, l);
	unionsorted(nb + 1, *nb, ruf + 1, *ruf, b + 1, b, l);
	differencesorted(a + 1, *a, b + 1, *b, nb + 1, nb, l);
}

void recursive(agent *r, agent *f, agent m, agent maxc, const edge *g, const agent *adj, agent d, const chunk *l, agent maxl,
	       void (*cf)(agent *, agent, const edge *, const agent *, const chunk *, void *), void *data) {

	if (*r && (d || *r == 1))
		cf(r, d, g, adj, l, data);

	if (*f && m) {

		agent k, *nr = r + maxc + 1, *nf = f + _N + 1, *nfs = nr + *r + 1, fs[_N], rt[_N];
		memcpy(rt, r + 1, sizeof(agent) * *r);
		sign w, y, z, p[_N + 2];

		for (k = 1; k <= MIN(*f, m); k++) {
			*nr = *r + k;
			memcpy(nr + 1, r + 1, sizeof(agent) * *r);
			memcpy(fs, f + *f - k + 1, sizeof(agent) * k);
			agent nd = maskcount(fs, k, l);
			if (d + nd <= maxl) {
				memcpy(nfs, fs, sizeof(agent) * k);
				QSORT(agent, nr + 1, *nr, LTL);
				nbar(fs, k, r, nr, adj, nf, l);
				recursive(nr, nf, m - k, maxc, g, adj, d + nd, l, maxl, cf, data);
			}
			inittwiddle(k, *f, p);
			while (!twiddle(&w, &y, &z, p)) {
				nd = nd - GET(l, fs[z]) + GET(l, f[w + 1]);
				fs[z] = f[w + 1];
				if (d + nd <= maxl) {
					memcpy(nr + 1, rt, sizeof(agent) * *r);
					memcpy(nfs, fs, sizeof(agent) * k);
					QSORT(agent, nr + 1, *nr, LTL);
					nbar(fs, k, r, nr, adj, nf, l);
					recursive(nr, nf, m - k, maxc, g, adj, d + nd, l, maxl, cf, data);
				}
			}
		}
	}
}

void coalitions(const edge *g, void (*cf)(agent *, agent, const edge *, const agent *, const chunk *, void *),
		void *data, agent maxc, const chunk *l, agent maxl) {

	chunk *tl;

	if (!l) {
		tl = (chunk *)malloc(sizeof(chunk) * _C);
		ONES(tl, _N, _C);
	}

	agent *r = (agent *)malloc(sizeof(agent) * (maxc + 1) * _N);
	agent *f = (agent *)malloc(sizeof(agent) * (_N + 1) * _N);
	edge ne = 0;

	for (agent i = 0; i < _N; i++)
		for (agent j = i + 1; j < _N; j++)
			if (g[i * _N + j]) ne++;

	agent *adj = createadj<_N>(g, ne, l ? l : tl);
	edge zero[_N] = {0};

	for (agent i = 0; i < _N; i++) {
		if (memcmp(g + i * _N, zero, sizeof(agent) * _N)) {
			r[0] = 0; f[0] = 1; f[1] = i;
			recursive(r, f, maxc, maxc, g, adj, 0, l ? l : tl, maxl, cf, data);
		}
	}

	if (!l) free(tl);
	free(adj);
	free(f);
	free(r);
}
