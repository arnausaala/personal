#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void inicializar(double *u, int N) {
    for (int i = 0; i < N; i++) {
        u[i] = 0.0;
    }
    u[N/2] = 100.0;
}

void guardar(double *u, int N, int iter) {
    printf("Iter %d: centro = %.2f\n", iter, u[N/2]);
}

int main(int argc, char** argv) {
    int N = 1000000;
    int max_iters = 1000;
    double tol = 1e-5;

    double *u     = (double*) malloc(N * sizeof(double));
    double *u_new = (double*) malloc(N * sizeof(double));

    inicializar(u, N);

    int iter = 0;
    double diff = 1.0;

    #pragma acc enter data copyin(u[0:N]) create(u_new[0:N])
 
    while (iter < max_iters && diff > tol) {
        diff = 0.0;

        #pragma acc parallel loop gang vector present(u[0:N], u_new[0:N])
        for (int i = 1; i < N - 1; i++) {
            u_new[i] = 0.25 * u[i-1] + 0.5 * u[i] + 0.25 * u[i+1];
        }

        #pragma acc parallel for gang vector present(u[0:N], u_new[0:N]) reduction(+:diff)
        for (int i = 1; i < N - 1; i++) {
            diff += fabs(u_new[i] - u[i]);
            u[i] = u_new[i];
        }

        if (iter % 100 == 0) {
            #pragma acc update host(u[0:N])
            guardar(u, N, iter);
        }

        iter++;
    }

    #pragma acc exit data copyout(u[0:N]) delete(u_new[0:N])

    guardar(u, N, iter);

    free(u);
    free(u_new);
    return 0;
}
