#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256
const float eps = 0.005;

template<unsigned int kWarpSize>
__device__ __forceinline__ float warpReduce(float val) {
    #pragma unroll
    for (unsigned int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}
template<unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce6(float* in, float* out) {
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
    const int tid = threadIdx.x;
    const int idx = tid + blockIdx.x * NUM_PER_BLOCK;

    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    float sum = 0.0f;
    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        sum += in[idx + i * THREAD_PER_BLOCK];
    }

    sum = warpReduce<WARP_SIZE>(sum);

    static __shared__ float sdata[NUM_WARPS];
    if (lane == 0) {
        sdata[warp] = sum;
    }
    __syncthreads();

    sum = (lane < NUM_WARPS) ? sdata[lane] : 0.0f;
    sum = warpReduce<NUM_WARPS>(sum);
    if (tid == 0)
        out[blockIdx.x] = sum;
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

    reduce6<num_per_block, num_per_thread><<<Grid, Block, smem>>>(d_a, d_out);

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