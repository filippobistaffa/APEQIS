#ifndef APELIB_H_
#define APELIB_H_

#define LTL(X, Y) (GET(l, *(X)) == GET(l, *(Y)) ? (*(X)) < (*(Y)) : GET(l, *(X)) > GET(l, *(Y)))
#define LEL(X, Y) (GET(l, *(X)) == GET(l, *(Y)) ? (*(X)) <= (*(Y)) : GET(l, *(X)) > GET(l, *(Y)))

agent *creteadj(const edge *g, edge ne, const chunk *l);

#endif /* APELIB_H_ */
