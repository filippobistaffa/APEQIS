#include "lpisg.h"

int main(int argc, char *argv[]) {

	unsigned seed = atoi(argv[1]);
	meter *sp = createsp(seed);

	IloEnv env;
	IloModel model(env);
	env.end();

	return 0;
}
