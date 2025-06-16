#include <stdlib.h>
#include "cuda.h"

int main(int argc, char** argv) {
    int N = 1000000;
    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));
    double *a = (double *)malloc(N * sizeof(double));
    double *b = (double *)malloc(N * sizeof(double));

    read_data(a, b, N);
    initialize(x, N);

    double *d_a;
    double *d_b;
    double *d_x;
    double *d_y;

    cudaMalloc(d_a, sizeof(double));
    cudaMalloc(d_b, sizeof(double));
    cudaMalloc(d_x, sizeof(double));
    cudaMalloc(d_y, sizeof(double));

    cudaMemcpy(d_a, a, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);

    int nthreads = 512;
    int nblocks = (N + nthreads-1)/nthreads;

    shift_mutiply_add<<<nblocks, nthreads>>> (N, d_a, d_b, d_x, d_y);

    cudaMemcpy(y, d_y, N*sizef(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_y);

    save_result(y, N);

    free(x); free(a); free(b); free(y);
    return 0;
}


__global__ void shift_multiply_add(int N, double *d_a, double *d_b, double *d_x, double *d_y){
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    id(idx < N)
        if (i == 0) {
            d_y[i] = (1 - d_a[i]) * d_b[i];
        }
        else {
            d_y[i] = (1 - d_a[i]) * d_b[i] + d_a[i] * d_x[i-1];
        }

}

shift_mutiply_add<<<nblocks, nthreads>>> (N, d_a, d_b, d_x, d_y);