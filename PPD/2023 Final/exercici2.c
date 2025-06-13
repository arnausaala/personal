#include "mpi.h"

void ScanSum_p2p(const int* sendbuf, int* recvbuf, MPI_Comm comm){
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int auxbuf[size];
    MPI_Request req;

    if(rank == 0){
        MPI_Isend(sendbuf[rank], 1, MPI_INT, rank, 1, comm, req);
        MPI_IRecv(recvbuf[rank], 1, MPI_INT, rank, 1, comm, req);
    }
    else{
        MPI_IRecv(auxbuf[rank], 1, MPI_INT, rank-1, 1, comm, req);
        auxbuf[rank] += sendbuf[rank];
        MPI_Isend(auxbuf[rank], 1, MPI_INT, rank, 1, comm, req);
        MPI_IRecv(recvbuf[rank], 1, MPI_INT, rank, 1, comm, req);
    }
}



// SOLUCIÓN

void ScanSum_p2p(const int *valor_local, int *resultado_scan, MPI_Comm comunicador)
{
    int rank, size;

    // Obtener el número total de procesos y el ID del proceso actual
    MPI_Comm_size(comunicador, &size);
    MPI_Comm_rank(comunicador, &rank);

    // Arreglo auxiliar para recibir los valores de procesos con menor rank
    int valores_recibidos[rank + 1];

    // Reservar espacio para las solicitudes y estados de comunicación
    MPI_Request solicitudes[size - 1];
    MPI_Status estados[size - 1];

    // Recibir valores de todos los procesos con menor rank
    for (int i = 0; i < rank; ++i) {
        MPI_Irecv(&valores_recibidos[i], 1, MPI_INT, i, 0, comunicador, &solicitudes[i]);
    }

    // Enviar el valor local a todos los procesos con mayor rank
    for (int i = rank + 1; i < size; ++i) {
        MPI_Isend(valor_local, 1, MPI_INT, i, 0, comunicador, &solicitudes[i - 1]);
    }

    // Esperar a que todas las comunicaciones no bloqueantes finalicen
    MPI_Waitall(size - 1, solicitudes, estados);

    // El resultado inicial es el valor local del proceso
    *resultado_scan = *valor_local;

    // Sumar todos los valores recibidos de procesos con menor rank
    for (int i = 0; i < rank; ++i) {
        *resultado_scan += valores_recibidos[i];
    }
}
