void main() {

    int N = 4096;
    float * a = (float*)malloc(sizeof(float)*N);
    float * b = (float*)malloc(sizeof(float)*N);
    float * c = (float*)malloc(sizeof(float)*N);
    float * d = (float*)malloc(sizeof(float)*N);

    float norm = 0.0;

    initialize(a); /* Consider this function initializes a. Do not parallelize it. */
    
    #pragma acc enter data copyin(a[0:N]) create(b[0:N], c[0:N], d[0:N])

    #pragma acc parallel for async(1) present(b[0:N])
    for(int i=0; i<N; ++i){
        b[i]=0.3*(N-i);
    }

    #pragma acc parallel for async(2) present(a[0:N], c[0:N])
    for(int i=0; i<N; ++i)
        c[i] = a[i] * i;

    #pragma acc wait

    #pragma acc parallel for reduction(+:norm) present(d[0:N], c[0:N])
    for (int i=1; i<N-1; ++i){
        d[i] = (1.0/3.0) * (c[i-1] + c[i] + c[i+1]);
        norm += c[i]*c[i];
    }

    #pragma acc exit data copyout(d[0:N]) delete(a[0:N], b[0:N], c[0:N])

    printf("The value of the mid point of d is: %f\n", d[N/2]);
    printf("The norm of the interior points of c is: %f\n",norm);
}