// CUDA Compute Capabilities Query

#include <stdio.h>

int main() {

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	printf("sm_%d%d\n", devProp.major, devProp.minor);
	return 0;
}
