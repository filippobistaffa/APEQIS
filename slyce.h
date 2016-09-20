
#ifndef SLYCE_H_
#define SLYCE_H_

#define LTDR(X, Y) (GET(dr, *(X)) == GET(dr, *(Y)) ? (*(X)) < (*(Y)) : GET(dr, *(X)) > GET(dr, *(Y)))
#define LEDR(X, Y) (GET(dr, *(X)) == GET(dr, *(Y)) ? (*(X)) <= (*(Y)) : GET(dr, *(X)) > GET(dr, *(Y)))

__attribute__((always_inline)) inline
void printc(const agent *c, penny v) {

	agent n = *c;
	printf("[ ");
	while (n--) printf("%u ", *(++c));
	printf("] = %.2f\n", 0.01 * v);
}

penny constraints(const edge *g, const agent *adj, const chunk *dr, const meter *sp, IloEnv &env, IloModel &model, IloFloatVarArray &ea, IloFloatVarArray &da);

#endif /* SLYCE_H_ */
