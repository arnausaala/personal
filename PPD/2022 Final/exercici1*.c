double simDiff(double *A, int N){
    double maxdiff = 0;

    #pragma omp parallel for schedule(dynamic) reduction(max:maxdiff)
    for(int i = 0; i < N; i++){
        maxdiff = 0;
        for(int j = i; j < N; j++){
            int local_diff = fabs(A[i*N+j]-A[j*N+i]);
            if(local_diff > maxdiff){
                maxdiff = local_diff;
            }
        }
    }

    return maxdiff;
}