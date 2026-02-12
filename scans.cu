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
// Works when n < number of threads per block. That is, when there's only 1 block
__global__ void naive_scan_block_slow(T* prfx, const uint32_t n) {
    extern __shared__ T shm[];
    const size_t tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + tid;
    
    if(gid < n) shm[tid] = prfx[gid];
    __syncthreads();

    for(size_t i = 0; i < n && i < gid && gid < n; i++)
        shm[tid] += prfx[i];
        //__syncthreads(); No need to sync threads
    
    __syncthreads();
    if(gid < n) prfx[gid] = shm[tid];
}

template<typename T>
__global__ void hillissteele_scan_block(T* prfx, const uint32_t n) {
    extern __shared__ T shm[];
    const size_t tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + tid;

    shm[tid] = (gid < n) ? prfx[gid] : 0;
    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride <<= 1) {
        if(tid >= stride) shm[tid] += shm[tid - stride]; 
        __syncthreads();
    }
    
    if(gid < n) prfx[gid] = shm[tid];
}

template<typename T>
__global__ void blelloch_scan_block(T* v, const uint32_t n) {
    extern __shared__ T shm[];

    const size_t tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + tid;

    // Load
    shm[tid] = (gid < n) ? v[gid] : 0;
    __syncthreads();

    // Up-sweep
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x)
            shm[idx] += shm[idx - stride];
        __syncthreads();
    }

    // Clear last element
    if (tid == blockDim.x - 1)
        shm[tid] = 0;
    __syncthreads();

    // Down-sweep
    for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x) {
            T tmp = shm[idx - stride];
            shm[idx - stride] = shm[idx];
            shm[idx] += tmp;
        }
        __syncthreads();
    }

    // Store
    if (gid < n) v[gid] = shm[tid+1];
}

int main() {
    constexpr uint32_t n = 10;
    constexpr uint32_t threads = 256;
    constexpr uint32_t blocks = (n + threads - 1)/threads;

    int32_t v[n] = {1,2,3,4,5,6,7,8,9,10}; // 1-indexed
    const size_t vsz = n*sizeof(v[0]), bsz = threads*sizeof(v[0]);
    int32_t *d_prfx, h_prfx[n];
    
    cudaMalloc(&d_prfx, vsz);
    cudaMemcpy(d_prfx, v, vsz, cudaMemcpyHostToDevice);

    blelloch_scan_block<<<blocks, threads, bsz>>>(d_prfx, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_prfx, d_prfx, vsz, cudaMemcpyDeviceToHost);
    cudaFree(d_prfx);
    
    printf("v: [%d", v[0]);
    for(size_t i = 1; i < n; i++) printf(", %d", v[i]);
    puts("]");

    printf("prfx: [%d", h_prfx[0]);
    for(size_t i = 1; i < n; i++) printf(", %d", h_prfx[i]);
    puts("]");
    
    return 0;
}