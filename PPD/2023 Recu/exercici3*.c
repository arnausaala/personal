#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/** functions **/
void initialize_A(int N, double *A){
    A[0] = 0;
    for (int i = 1; i < N; ++i)
        A[i] = A[i - 1] + 2 * i;
}

void initialise_b(int N, double *B){
    #pragma acc parallel loop copyin(B[0:N]) async(1) 
    for (int i = 0; i < N; ++i)
        B[i] = 2 * i * i + 0.5;
}

void initialise_c(int N, double *A, double *C){
    C[0] = 1.0;
    C[N - 1] = 0;
    #pragma acc parallel loop copyin(C[0:N], A[0:N]) async(2)
    for (int i = 1; i < N - 1; ++i)
        C[i] = 0.33 * (A[i - 1] + A[i] + A[i + 1]);
}

double eval_maxBC(int N, double *B, double *C){
    double aux;
    double max = 0.0;
    #pragma acc wait
    #pragma acc parallel loop present(B[0:N], C[0:N]) 
    for (int i = 0; i < N; ++i){
        aux = fmax(fabs(B[i]), fabs(C[i]));
        max = fmax(max, aux);
    }
    return max;
}

/** main program **/
int main(int argc, char** argv){
    double *A, *B, *C;
    double max;
    int N = 1000000;

    A = (double*)malloc(N * sizeof(double));
    B = (double*)malloc(N * sizeof(double));
    C = (double*)malloc(N * sizeof(double));

    initialize_A(N, A);
    initialise_b(N, B);
    initialise_c(N, A, C);
    max = eval_maxBC(N, B, C);

    printf("El màxim és: %lf\n", max);

    free(A);
    free(B);
    free(C);
    return 0;
}
