__global__ void blur_iteration(double *d_im, int N, int M, double *d_im_new){
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(i < M && j < N)
        im_new[ind(i, j, N)] = (im[ind(i , j , N)] + im[ind(i , j-1, N)] + im[ind(i-1, j-1, N)] + im[ind(i+1, j , N)] + im[ind(i , j+1, N)]) / 5.0;
}


#include <stdlib.h>
#include "cuda.h"


int main(int argc, char** argv) {
    int N = 512;
    int M = 512;
    double *im = (double *)malloc(M*N*sizeof(double));
    double *im_new = (double *)malloc(M*N*sizeof(double));
    initialize_image(im, M, N);

    double *d_im, *d_im_new;

    cudaMalloc(&d_im, M*N*sizeof(double));
    cudaMalloc(&d_im_new, M*N*sizeof(double));

    cudaMemcpy(d_im, im M*N*sizeof(double), cudaMemcpyHostToDevice);

    B = M / 16;
    
    int dimgx = (M+B-1)/B
    int dimgx = (N+B-1)/B

    dim3 dimGrid(dimgx, dimgy, 1); 
    dim3 dimBlock(B, B, 1);

    blur_iteration<<<dimGrid, dimBlock>>> (d_im, M, N, d_im_new);

    cudaMemcpy(im_new, d_im_new, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_im);
    cudaFree(d_im_new);

}