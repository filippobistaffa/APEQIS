#ifndef APELIB_H_
#define APELIB_H_

#define LTL(X, Y) (GET(l, *(X)) == GET(l, *(Y)) ? (*(X)) < (*(Y)) : GET(l, *(X)) > GET(l, *(Y)))
#define LEL(X, Y) (GET(l, *(X)) == GET(l, *(Y)) ? (*(X)) <= (*(Y)) : GET(l, *(X)) > GET(l, *(Y)))

#ifndef PRINTBUF
#define PRINTBUF

#include <iostream>
template <typename type>
__attribute__((always_inline)) inline
void printbuf(const type *buf, unsigned n, const char *name = NULL) {

	if (name) printf("%s = [ ", name);
	else printf("[ ");
	while (n--) std::cout << *(buf++) << " ";
	printf("]\n");
}

#endif

double *apeqis(const edge *g, value (*cf)(agent *, const chunk *, void *), void *data = NULL,
	       const chunk *l = NULL, agent maxc = N, agent maxl = N);

#endif /* APELIB_H_ */
