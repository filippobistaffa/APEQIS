#ifndef APELIB_H_
#define APELIB_H_

#define LTL(X, Y) (GET(l, *(X)) == GET(l, *(Y)) ? (*(X)) < (*(Y)) : GET(l, *(X)) > GET(l, *(Y)))
#define LEL(X, Y) (GET(l, *(X)) == GET(l, *(Y)) ? (*(X)) <= (*(Y)) : GET(l, *(X)) > GET(l, *(Y)))

#define EPSILON 0.01

double *apeqis(const edge *g, value (*cf)(agent *, const chunk *, void *), void *data = NULL,
	       const chunk *l = NULL, agent maxc = N, agent maxl = N);

#endif /* APELIB_H_ */
