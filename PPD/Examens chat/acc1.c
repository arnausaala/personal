// PONER DIRECTIVAS DE OPEN ACC


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void generate_random(double* x, double* y, int N, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < N; i++) {
        x[i] = (double) rand() / RAND_MAX;
        y[i] = (double) rand() / RAND_MAX;
    }
}

void save_pi_estimate(double pi, int iter) {
    printf("Iter %d: PI ≈ %.8f\n", iter, pi);
}

int main(int argc, char** argv) {
    int N = 1000000;
    double *x = (double*) malloc(N * sizeof(double));
    double *y = (double*) malloc(N * sizeof(double));
    int *inside = (int*) malloc(N * sizeof(int));
    double pi = 0;
    int max_iter = 100;

    #pragma acc enter data create(x[0:N], y[0:N], inside[0:N])

    #pragma acc parallel loop
    for (int iter = 0; iter < max_iter; iter++) {

        generate_random(x, y, N, iter);  // Generación en CPU

        #pragma acc update device(x[0:N], y[0:N])

        #pragma acc parallel loop gang vector present(x[0:N], y[0:N], inside[i])
        for (int i = 0; i < N; i++) {
            double dist = x[i]*x[i] + y[i]*y[i];
            inside[i] = dist <= 1.0 ? 1 : 0;
        }

        int total = 0;
        
        #pragma acc paralell loop gang vector present(inside[0:N]) reduction(+:total)
        for (int i = 0; i < N; i++) {
            total += inside[i];
        }

        pi = 4.0 * total / N;

        save_pi_estimate(pi, iter);
    }

    #pragma acc exit data delete(x[0:N], y[0:N], inside[0:N])

    free(x); free(y); free(inside);
    return 0;
}