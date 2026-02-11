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
// This dot product implementation is bad cuz shared memory is per block, so it just partially retrieves the value. Also atomicAdd breaks speed
__global__ void dot_slow(const T* v0, const T* v1, T* __restrict out, const uint32_t n) {
    extern __shared__ unsigned char smem[];
    T* result = reinterpret_cast<T*>(smem);
    const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(!tidx) *result = 0;
    
    __syncthreads(); // Barrier => All threads only start when result = 0;
    
    if(tidx < n) atomicAdd(result, v0[tidx] * v1[tidx]); // Avoid Race Conditions

    if(!tidx) out[0] = *result;
}

/*
template<typename T>
__global__ void dot(const T* v0, const T* v1, T* __restrict out, const int n) {
    extern __shared__ unsigned char smem[]; => Shared memory per block
    T* result = reinterpret_cast<T*>(smem);
    const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x; => Invalid index => Out of bounds
    
    if(tidx < n) result[tidx] = v0[tidx]*v1[tidx];
    int k = log2(n); // Steps
    for(int i = 1; i <= k; i++) {
        __syncthreads();
        result[n*i + tidx] = result[n*(i-1) + tidx] + result[n*(i-1) + tidx + 1]; // Waste of memory (NlogN)
    }
}
*/

template<typename T>
__global__ void dot_block(const T* v0, const T* v1, T* __restrict block_out, const uint32_t n) {
    extern __shared__ T sdata[];

    const size_t tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + tid; // Global ID

    // Load into shared memory
    sdata[tid] = (gid < n) ? v0[gid] * v1[gid] : T(0);
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // One value per block
    if (tid == 0) block_out[blockIdx.x] = sdata[0];
}

template<typename T>
T dot(const T* d_v0, const T* d_v1, uint32_t n) {
    T *d_in, *d_out;
    const uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1)/threads;
    cudaMalloc(&d_out, blocks * sizeof(T));
    dot_block<<<blocks, threads, threads*sizeof(T)>>>(d_v0, d_v1, d_out, n);
    cudaDeviceSynchronize();

    cudaMalloc(&d_in, blocks * sizeof(T));
    while(blocks > 1) {
        T* aux = d_out;
        d_out = d_in;
        d_in = aux;

        n = blocks;
        blocks = (n + threads - 1)/threads;
        
        vreduce<<<blocks, threads, threads * sizeof(T)>>>(d_in, d_out, n);
        cudaDeviceSynchronize();
    }

    T h_r;
    cudaMemcpy(&h_r, d_out, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return h_r;
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
    //dot_slow<<<blocks, threads, sizeof(int32_t)>>>(d_o, d_o, d_dot, n);
    h_r = dot(d_o, d_o, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_o, d_o, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(&h_r, d_dot, sizeof(int32_t), cudaMemcpyDeviceToHost);
    
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