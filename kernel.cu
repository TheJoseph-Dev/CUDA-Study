#include <stdio.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void vreduce(const T* __restrict v, T* __restrict out, const uint32_t n) {
    extern __shared__ T shm[];

    const size_t tid = threadIdx.x;    
    const size_t gid = blockIdx.x * blockDim.x + tid; // Global ID

    shm[tid] = (gid < n) ? v[gid] : T(0);
    __syncthreads();

    for(int stride = blockDim.x / 2; stride; stride >>= 1) {
        if(tid < stride) shm[tid] += shm[tid + stride];
        __syncthreads();
    }

    if(tid == 0) out[blockIdx.x] = shm[0];
}

template<typename T>
__global__ void prfxsum(T* prfx, const uint32_t n) {
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
}

int main() {
    constexpr uint32_t n = 10;
    constexpr uint32_t threads = std::min(256U, n);
    constexpr uint32_t blocks = (n + threads - 1)/threads;

    uint32_t v[n+1] = {0,1,2,3,4,5,6,7,8,9,10}; // 1-indexed
    uint32_t* d_prfx, h_prfx[n+1];

    cudaMalloc(&d_prfx, (n+1)*sizeof(v[0]));
    cudaMemcpy(v, d_prfx, (n+1)*sizeof(v[0]), cudaMemcpyHostToDevice);

    prfxsum<<<blocks, threads>>>(d_prfx, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_prfx, &d_prfx, (n+1)*sizeof(v[0]), cudaMemcpyDeviceToHost);
    cudaFree(d_prfx);
    
    printf("v: [%d ", v[1]);
    for(size_t i = 2; i <= n; i++) printf(", %d", v[i]);
    puts("]");

    printf("prfx: [%d ", h_prfx[1]);
    for(size_t i = 2; i <= n; i++) printf(", %d", h_prfx[i]);
    puts("]");
    
    return 0;
}