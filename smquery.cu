// CUDA Compute Capabilities Query

#include <stdio.h>

int main() {

	// Number of CUDA devices
	int devCount;
	cudaGetDeviceCount(&devCount);

	// Iterate through devices
	for (int i = 0; i < devCount; ++i) {
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		printf("sm_%d%d\n", devProp.major, devProp.minor);
	}

	return 0;
}
