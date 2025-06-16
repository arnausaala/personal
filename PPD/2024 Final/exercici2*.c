#include "mpi.h"
int main(int argc, char **argv) {
    int N = 10000000;
    double *signal_in, *signal_out;
    double sum_signal = 0;
    
    signal_in = (double *) malloc(N * sizeof(double));
    signal_out = (double *) malloc(N * sizeof(double));
    
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request req[4*size];
    MPI_Status stat[size];

    double *anterior = (double*)malloc(sizeof(double));
    double *siguiente = (double*)malloc(sizeof(double));


    // Initialization
    for (int i = 0; i < N/size; ++i) {
        signal_in[i] = sin(i*rank);
    }
    
    // Smoothing
    if(rank == 0){
        signal_out[0] = (signal_in[1] + signal_in[0]) / 2;
        MPI_Send(&signal_in[0], 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, req[0]);
    }
    else if(rank = size - 1){
        signal_out[N - 1] = (signal_in[N - 1] + signal_in[N - 2]) / 2;
        MPI_Send(&signal_in[N-1], 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, req[rank]);
    }
    else{
        MPI_ISend(&signal_in[rank], 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, req[rank]);
        MPI_ISend(&signal_in[rank], 1, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, req[2*rank]);

        MPI_Recv(&anterior, 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, req[3*rank]);
        MPI_Recv(&siguiente, 1, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, req[4*rank]);

        signal_out[rank] = (anterior + signal_in[i] + siguiente) / 3;
    }

    MPI_Wait(req, stat);
    
    if(rank == 0){
        MPI_Reduce(signal_out, sum_signal, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    
    // Print
    std::cout << "The sum is: " << sum_signal << std::endl;
    
    // Free allocated memory
    free(signal_in);
    free(signal_out);
}