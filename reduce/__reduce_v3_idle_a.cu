#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256
const float eps = 1e-3;

__global__ void reduce3_a(float* in, float* out) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int idx = threadIdx.x + blockDim.x * blockIdx.x * 2;
    sdata[tid] = in[idx] + in[idx + blockDim.x];
    __syncthreads();

    for (int mask = blockDim.x >> 1; mask >= 1; mask >>= 1){
        if (tid < mask){
            sdata[tid] += sdata[tid + mask];
        }
        __syncthreads();
    } 
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

    int block_num = N / THREAD_PER_BLOCK / 2;
    float *out=(float *)malloc(block_num* sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,block_num * sizeof(float));
    float *res = (float *)malloc(block_num * sizeof(float));
    
    for (int i = 0; i < N; i++)
        a[i] = 2.0 * (float)drand48() - 1.0;
    
    // calculate on cpu
    for (int i = 0; i < block_num; ++i) {
        float cur = 0.0f;
        for (int j = 0; j < THREAD_PER_BLOCK * 2; ++j){
            cur += a[i * THREAD_PER_BLOCK * 2 + j];
        }
        res[i] = cur;
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num);
    dim3 Block(THREAD_PER_BLOCK);
    int smem = THREAD_PER_BLOCK * sizeof(float);

    reduce3_a<<<Grid, Block, smem>>>(d_a, d_out);

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