#include "cuda.h"

__global__ void first_op(float *b, int N){

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < N){
        b[i] = 0.3 * (N - i);
    }
}

__global__ void second_op(float *a, float *c, int N){

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < N){
        c[i] = a[i] * i;
    }
}

__global__ void third_op(float *c, float *d, float *norm, int N){

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i > 0 && i < N-1){
        d[i] = (1/3) * (c[i - 1] + c[i] + c[i + 1]);
        norm += c[i] * c[i];
    }
}

void main (int argc, char *argv[])
{
    int N = 4096; 
    int BLOCK_SIZE = 16; 
    float norm;

    float * a = (float*)malloc(sizeof(float)*N);
    float * b = (float*)malloc(sizeof(float)*N);
    float * c = (float*)malloc(sizeof(float)*N);
    float * d = (float*)malloc(sizeof(float)*N);

    float *d_a, *d_b, *d_c, *d_d, *d_norm;

    cudaMalloc((void**)&d_a,N*sizeof(float));
    cudaMalloc((void**)&d_b,N*sizeof(float));
    cudaMalloc((void**)&d_c,N*sizeof(float));
    cudaMalloc((void**)&d_d,N*sizeof(float));
    cudaMalloc((void**)&d_norm,sizeof(float));


    initialize(a); /* Consider this function initializes a. Do not parallelize it. */

    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm, 0, sizeof(float), cudaMemcpyHostToDevice);
    int NUM_BLOCKS = (N+BLOCK_SIZE-1)/BLOCK_SIZE;

    first_op<<< NUM_BLOCKS, BLOCK_SIZE >>>(d_b, N)
    second_op<<< NUM_BLOCKS, BLOCK_SIZE >>>(d_a, d_c, N)
    third_op<<< NUM_BLOCKS, BLOCK_SIZE >>>(d_c, d_d, d_norm, N)

    cudaMemcpy(&d_d, vald, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_norm, norm, N*sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_norm);


    printf("The value of the mid point of d is: %f\n", vald);
    printf("The norm of the interior points of c is: %f\n",norm);

}