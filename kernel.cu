#include <stdio.h>
#include <cuda_runtime.h>

// restrict applies per kernel invocation, not across threads. So it must not have overlapping ranges across threads.
template<typename T>
__global__ void vsum(const T* v0, const T* v1, T* __restrict vout, const uint32_t n) {
    const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tidx < n) vout[tidx] = v0[tidx] + v1[tidx];
}

template<typename T>
__global__ void vsum(T* __restrict v0, const T* __restrict v1, const uint32_t n) {
    const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tidx < n) v0[tidx] += v1[tidx];
}

template<typename T>
__global__ void dot_slow(const T* v0, const T* v1, T* __restrict out, const uint32_t n) {
    extern __shared__ unsigned char smem[];
    T* result = reinterpret_cast<T*>(smem);
    const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(!tidx) *result = 0;
    
    __syncthreads(); // Barrier => All threads only start when result = 0;
    
    if(tidx < n) atomicAdd(result, v0[tidx] * v1[tidx]); // Avoid Race Conditions

    if(!tidx) out[0] = *result;
}

int main() {
    constexpr uint32_t n = 2;
    int32_t h_a[n] = {2, 4}, h_b[n] = {6, 9}, h_o[n], h_r;

    int32_t *d_a, *d_b, *d_o;
    cudaMalloc(&d_a, n * sizeof(int32_t));
    cudaMalloc(&d_b, n * sizeof(int32_t));
    cudaMalloc(&d_o, n * sizeof(int32_t));
    
    cudaMemcpy(d_a, h_a, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    constexpr uint32_t threads = std::min(256U, n);
    constexpr uint32_t blocks = (n + threads - 1)/threads;
    vsum<<<blocks,threads>>>(d_a,d_b,d_o,n);
    cudaDeviceSynchronize();

    int32_t *d_dot;
    cudaMalloc(&d_dot, n * sizeof(int32_t));
    dot_slow<<<blocks, threads, sizeof(int32_t)>>>(d_o, d_o, d_dot, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_o, d_o, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_r, d_dot, sizeof(int32_t), cudaMemcpyDeviceToHost);
    
    printf("[%d", h_o[0]);
    for(uint32_t i = 1; i < n; i++)
        printf(", %d", h_o[i]);
    puts("] ");

    printf("|o| = %.2lf\n", sqrt(h_r));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    cudaFree(d_dot);
    return 0;
}