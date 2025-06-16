double pi_calc(int num_steps){

    double step;
    double x, pi, sum = 0.0;
    
    step = 1.0 / (double) num_steps;

    for (int i = 0; i < num_steps; i++){
        x = (i + 0.5) * step;
        sum += 4.0/(1.0 + x * x);
    }

    pi = sum * step;
    
    return pi;
}

#include "mpi.h"
#define numSteps 10000

double pi_calc_mpi(int rank, int size){
    
    double numBins = rank/size;
    double step = (double)1/(numBins*size);

    double sum = 0;
    double pi = 0;
    double local_pi = 0;
    double x;

    for(int i = 0; i < numSteps; i++){
        x = (i + 0.5) * step;
        sum += 4.0/(1.0 + x * x);
    }

    local_pi = sum * step;

    MPI_Allreduce(&local_pi, 1, MPI_DOUBLE; MPI_SUM, MPI_COMM_WORLD);

    return pi;


}