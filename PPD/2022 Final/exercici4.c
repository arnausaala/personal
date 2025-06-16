void vectOperations( double *x, double *y, double *z, double *d, int N){
    double * aux = (double *) malloc(N*sizeof(double));
    
    for(int i=0; i<N; ++i){
        aux[i]=x[i]-y[N-i];
        z[i]=z[i]*i;
    }
    
    for(int i=0; i<N; ++i){
        d[i]=aux[i]+z[N-i];;
    }
}
__global__ vectOperations(double *x,double *y,double *z,double *d,int N){
    
}

vectOperations<<<nblocks, nthreads>>>(...);