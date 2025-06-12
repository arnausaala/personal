double simDiff(double *A, int N){
    double maxdiff = 0;
    
    #pragma omp parallel for schedule(dynamic) reduction(max:maxdiff)
    for(int i = 0; i < N; i++){
        for(int j = i; j < N; j++){
            double localdiff = fabs(A[i*N + j]-A[j*N + i]);
            if(localdiff > maxdiff){
                maxdiff = localdiff;
            }
        }
    }

    return maxdiff;
}