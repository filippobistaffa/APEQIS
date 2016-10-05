#ifndef APELIB_H_
#define APELIB_H_

#define LTL(X, Y) (GET(l, *(X)) == GET(l, *(Y)) ? (*(X)) < (*(Y)) : GET(l, *(X)) > GET(l, *(Y)))
#define LEL(X, Y) (GET(l, *(X)) == GET(l, *(Y)) ? (*(X)) <= (*(Y)) : GET(l, *(X)) > GET(l, *(Y)))

double *apeqis(const edge *g, edge ne, const chunk *l, value (*cf)(agent *, const chunk *, const void *), const void *data);

#endif /* APELIB_H_ */
