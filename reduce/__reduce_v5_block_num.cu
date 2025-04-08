#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256
const float eps = 0.005;

__device__ __forceinline__ void warpReduce(volatile float* cache, int tid){
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

template<unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce5(float* in, float* out) {
    extern __shared__ volatile float sdata[];

    const int tid = threadIdx.x;
    float* in_begin = in + blockIdx.x * NUM_PER_BLOCK;
    sdata[tid] = 0;
    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        sdata[tid] += in_begin[tid + i * THREAD_PER_BLOCK];
    }
    __syncthreads();

    for (int mask = blockDim.x >> 1; mask > 32; mask >>= 1){
        if (tid < mask){
            sdata[tid] += sdata[tid + mask];
        }
        __syncthreads();
    } 
    if (tid < 32)
        warpReduce(sdata, tid);
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

bool check(float *out,float *res,int n){
    for(int i = 0; i < n; i++){
        if(abs(out[i] - res[i]) > eps)
            return false;
    }
    return true;
}

int main() {
    const int N = 32 * 1024 * 1024;
    float *a= (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    constexpr int block_num = 1024;
    constexpr int num_per_block = N / block_num;
    constexpr int num_per_thread = num_per_block / THREAD_PER_BLOCK;
    float *out=(float *)malloc(block_num* sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,block_num * sizeof(float));
    float *res = (float *)malloc(block_num * sizeof(float));
    
    for (int i = 0; i < N; i++)
        a[i] = 2.0 * (float)drand48() - 1.0;
    
    // calculate on cpu
    for (int i = 0; i < block_num; ++i) {
        float cur = 0.0f;
        for (int j = 0; j < num_per_block; ++j){
            cur += a[i * num_per_block + j];
        }
        res[i] = cur;
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num);
    dim3 Block(THREAD_PER_BLOCK);
    int smem = THREAD_PER_BLOCK * sizeof(float);

    reduce5<num_per_block, num_per_thread><<<Grid, Block, smem>>>(d_a, d_out);

    cudaMemcpy(out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if(check(out, res, block_num)) printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i = 0;i < block_num; i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}