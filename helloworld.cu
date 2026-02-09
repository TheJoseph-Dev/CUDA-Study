#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloWorldKernel() {
    printf("Hello, World! from GPU thread %d\n", threadIdx.x);
}

int main() {
    helloWorldKernel<<<2, 2>>>();

    // Catch kernel LAUNCH errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(launchErr));
        return 1;
    }

    // Catch kernel EXECUTION errors
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("Runtime error: %s\n", cudaGetErrorString(syncErr));
        return 1;
    }

    return 0;
}
