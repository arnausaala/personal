void argmax_seq(double *v, int N, double *m, int *idx_m){
    *m = -1;
    *idx_m = -1;

    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        #pragma omp critical
        if(v[i] > *m){
            *m = v[i];
            *idx_m = i;
        }
    }
}

// A) 0.8
// B) 1.4
// C) 0.8

void argmax_tasks(double *v, int N, double *m, int *idx_m){
    if(N < 512){
        argmax_seq(v, N, m, idx_m);
    }
    else{
        double m1, m2;
        int idx_m1, idx_m2;
        
        #pragma omp task shared(m1, idx_m1)
        argmax_tasks(v, N/2, &m1, &idx_m1);

        #pragma omp task shared(m2, idx_m2)
        argmax_tasks(v + N/2, N/2, &m2, &idx_m2);

        #pragma omp taskwait

        if(m1 >= m2){
            *m = m1;
            *idx_m = idx_m1;
        }
        else{
            *m = m2;
            *idx_m = idx_m2;
        }
    }
}


#include <stdlib.h>
#include "omp.h"

#define N 1024*1024

int main(char *argv[], int argc) {

    double *v = (double *)malloc(N*sizeof(double));
    inititialize(v, N);
    double max;
    int max_idx;

    #pragma omp parallel
    #pragma omp single
    argmax_tasks(v, N, &max, &max_idx);
    
    free(v);
    return 0;
}