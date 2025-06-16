__global__ void media_movil_512(const float *d_in, float *d_out) {
    extern __shared__ float temp[];  // Memoria compartida dinámica
    int tid = threadIdx.x;           // ID del hilo dentro del bloque

    // Manejo de bordes: cargar vecinos extremos o poner 0.0 si están fuera de rango
    if (tid < 2) {
        temp[tid] = 0.0f;                     // i - 2, i - 1 (para los primeros hilos)
        temp[tid + 512 + 2] = 0.0f;           // i + 1, i + 2 (para los últimos hilos)
    }
    if(tid < 512){
        temp[tid + 2] = d_in[tid];
    }

    __syncthreads();  // Esperar a que todos hayan copiado

    d_out[tid] = (temp[tid] + temp[tid+1] + temp[tid+2] + temp[tid+3] + temp[tid+4]) / 5;
}
