__global__ void stencil_512(double *d_in, double *d_out) {
    extern __shared__ double temp[];

    int id = threadIdx.x;

    temp[id + 3] = d_in[id];

    if (id < 3) {
        temp[id] = 0.0;             // Valores antes de d_in[0]
        temp[id + 512] = 0.0;       // Valores después de d_in[511]
    }

    __syncthreads();

    double result = 0.0;
    for (int i = id - 3; i < id + 3; i++) {
        result += temp[i];
    }

    d_out[id] = result;
}