int transpose(double* transposeA, double* A, int N){
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            transposeA[j*N + i] = A[i*N+j];
        }
    }

    return transposeA;
}


/*

[1, 2, 3 | 4, 5, 6 | 7, 8, 9]
 0  1  2   3  4  5   6  7  8
[1, 2, 3]
[4, 5, 6]
[7, 8, 9]


[1, 4, 7]
[2, 5, 8]
[3, 6, 9]

ordre
0, 3, 6, 1, 4, 7, 2, 5, 8


*/