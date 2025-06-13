#include "mpi.h"

void Alltoall1_Int_p2p(const int* sendbuf, int* recvbuf, MPI_Comm comm){
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    
    MPI_Request req[2*size];
    MPI_Status stat[size];

    for(int i = 0; i < size, i++){
        MPI_Isend(&sendbuf[rank+i], 1, MPI_INT, i, 1, comm, &req[i]);
        MPI_IRecv(&recvbuf[rank+i], 1, MPI_INT, i, 1, comm, &req[size+i]);
    }
    MPI_Waitall(size, req, stat);
}