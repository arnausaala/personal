__global__ void suma_parcial(float *d_in, float *d_out){
    extern __shared__ float temp[];
    int idx = threadIdx.x;

    if(idx < 512){
        temp[idx] = d_in[idx];
    }

    __syncthreads();

    if(idx < 4){
        d_out[idx] = 0;
        for(int i = idx*128; i < (idx+1)*128; i++){
            d_out[idx] += temp[i];
        }
    }
}