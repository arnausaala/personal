#include "mpi.h"

double simDiff( double * row, int N, MPI_Comm mycomm){

    double maxdiff;

    int rank, size = N;
    MPI_Comm_rank(mycomm, &rank);
    double local_max;
    MPI_Status stat;

    for (int i = 0; i < N; ++i){
        if(row[i] > local_max){
            local_max = row[i];
        }
    }

    MPI_Send(&local_max, 1, MPI_DUBLE, 0, 1, mycomm, stat);

    if(rank == 0){
        (double) *recvbuf = (double*)malloc(N*sizeof(double))
        MPI_Recv(&recvbuf[i], 1, MPI_double, i, 1, mycomm, stat);
        MPI_Reduce(&recvbuf, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, mycomm);
    }

    return maxdiff;
}