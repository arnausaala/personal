/*
Tamany matriu: 1.000.000 x 1.800
volem calcular el temps per arribar a 10.000 punts

*/

#define SCR 10000

// A)
void compute_times(int N, int M, int *scores, int *times){
    int done[N];

    #pragma omp for num_threads(4)
    for(int i = 0; i < N; i++){
        int score_local = 0;
        for(int j = 0; j < M; j++){
            score_local += scores[i*N+j];
            if(score_local > SCR && done[i] == 0){
                times[i] = j;
                done[i] = 1;
            }
        }
        if(done == 0){times[i] = M-1;}
    }
}

// B)
#pragma omp scheduler(dynamic, chunk_size)

// C)
void compute_histogram(int N, int M, int *times, int *hist){
    #pragma omp for
    for(int i = 0; i < N; i++){
        #pragma omp atomic
        hist[times[i]]++;
    }
}