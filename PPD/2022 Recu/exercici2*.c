#include "mpi.h"

double pi_calc_mpi(int num_steps, MPI_Comm comm){

    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    MPI_Status stat[size-1];
    
    double *recvbuff = (double*)malloc(size*sizeof(double));

    double x;
    double pi; 
    double sum = 0;
    double step = 1 / (double) num_steps;
    
    for (int i = 0; i < num_steps/size; i++){
        x = (i + 0.5) * step;
        sum += 4/(1 + x * x);
    }

    if(rank != 0){
        MPI_Send(&sum, 1, MPI_DOUBLE, 0, 1, comm);
    }
    else{
        recvbuff[0] = sum;
        for(int i = 1; i < size; i++){
            MPI_Recv(&recvbuff[i], 1, MPI_DOUBLE, i, 1, comm, stat[i-1]);
        }
    }


    sum = 0;
    for(int i = 0; i < size; i++){
        sum += recvbuff[i];
    }

    pi = sum * step;
    
    return pi;
}


