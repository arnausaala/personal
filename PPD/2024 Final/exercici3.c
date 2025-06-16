int main(int argc, char** argv) {
    
    int N = atoi(argv[1]);
    double *x = (double *)malloc(N*sizeof(double));
    double *a = (double *)malloc(N*sizeof(double));
    double *b = (double *)malloc(N*sizeof(double));
    double *ax = (double *)malloc(N*sizeof(double));
    double *ab = (double *)malloc(N*sizeof(double));
    
    read_data(a, b, N);
    
    // copy a,b and allocate GPU mem for x, ax, ab
    #pragma acc enter data copyin(a[0:N], b[0:N]) create(x[0:N],ax[0:N],ab[0:N])
    
    // init x on the GPU
    #pragma acc parallel loop gang vector present(x[0:N])
    
    for (int i = 0; i < N; i++)
        x[i] = 0;
    
    // this loop should cannot be parallelized
    for (int iter = 0; iter < 500; iter++) {
    
        // this loop CAN be parallelized - each iter is independent
        #pragma acc parallel loop gang vector present(x[0:N],a[0:N],ax[0:N]) async(1)
        for (int i = 1; i < N; i++)
        ax[i] = x[i-1] * a[i];
        
        // this loop is independent of the previous loop, launch it asynchronously
        // on queue 2
        #pragma acc parallel loop gang vector present(a[0:N],b[0:N],ab[0:N]) async(2)
        for (int i = 0; i < N; i++)
            ab[i] = (1 - a[i]) * b[i];
        
        // we need to wait because we need ax and ab
        #pragma acc wait
        #pragma acc parallel loop gang vector present(x[0:N],ax[0:N],ab[0:N]) async(1)
        for (int i = 0; i < N; i++)
            x[i] = ax[i] + ab[i];
    }
    
    // wait for last loop of last iter
    #pragma acc wait
    
    // copy only x (it's the only output used)
    #pragma acc exit data copyout(x[0:N]) delete(a[0:N], b[0:N], ax[0:N], ab[0:N])
    
    save_result(x, N);
    
    free(x); free(a); free(b); free(ax); free(ab);
    return 0;
}