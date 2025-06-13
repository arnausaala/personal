int transpose(double *transposeA, double *A, int N, int stride){
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i+=stride){
        for(int k = i; k < i+stride && k < N; k++){
            for(int j = 0; j < N; j+=stride){
                for(int l = j; l < j + stride && l < N; l++){
                    transposeA[l*N+k] = A[k*N+l];
                }
            }
        }
    }
    
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i+=stride){
        for(int k = i; k < i+stride && k < N; k++){
            for(int j = 0; j < N; j+=stride){
                for(int l = j; l < j+stride && l < N; l++){
                    transposeA[l*N+k] = A[k*N+l];
                }
            }
        }
    }
}

