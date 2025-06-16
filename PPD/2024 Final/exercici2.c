#include "mpi.h"

int main(int argc, char **argv) {
    int N = 10000000;
    double *signal_in, *signal_out;
    double sum_signal = 0;

    signal_in = (double *) malloc(N * sizeof(double));
    signal_out = (double *) malloc(N * sizeof(double));
    
    MPI_Init(argc, &argv);
    int MPI_Comm_rank(MPI_Comm comm, int *rank);
    int MPI_Comm_size(MPI_Comm comm, int *size);
    MPI_Request req[size];
    MPI_Status stat[size];

    double *sendbuf_ant = (double*)malloc(sizeof(double));
    double *sendbuf_sig = (double*)malloc(sizeof(double));
    double *recvbuf_ant = (double*)malloc(sizeof(double));
    double *recvbuf_sig = (double*)malloc(sizeof(double));

    // Initialization
    if(rank == 0){
        for (int i = 0; i < N; ++i) {
            signal_in[i] = sin(i);
        }
    }
   
    // Smoothing
    signal_out[0] = (signal_in[1] + signal_in[0]) / 2;
    signal_out[N - 1] = (signal_in[N - 1] + signal_in[N - 2]) / 2;
    
    for (int i = 1; i < N - 1; ++i) {

        sendbuf_ant = signal_in[i-1];
        sendbuf_sig = signal_in[i+1];

        MPI_ISend(sendbuf_ant, 1, MPI_DOUBLE, rank-1, 1, comm, req[i]);
        MPI_ISend(sendbuf_sig, 1, MPI_DOUBLE, rank+1, 1, comm, req[i]);

        MPI_IRecv(recvbuf_ant, 1, MPI_DOUBLE, rank-1, 1, comm, req[i]);
        MPI_IRecv(recvbuf_sig, 1, MPI_DOUBLE, rank+1, 1, comm, req[i]);

        signal_out[i] = (recvbuf_ant + signal_in[i] + recvbuf_sig) / 3;
    }

    MPI_wait(req, stat);
    
    // Sum
    if(rank = 0){
        for (int i = 0; i < N; ++i) {
            sum_signal += signal_out[i];
        }
    }

    // Print
    if(rank == 0){
        std::cout << "The sum is: " << sum_signal << std::endl;
    }
    
    // Free allocated memory
    free(signal_in);
    free(signal_out);

    MPI_Finalize();
}