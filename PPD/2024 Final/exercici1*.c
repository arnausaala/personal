#define toReach 10000

void compute_times(int N, int M, int *scores, int *times){

    int local_time, local_score, reached;

    #pragma omp parallel for num_threads(4)
    {
        for(int i = 0; i < N; i++){
            local_score = 0;
            local_time = 0;
            reached = 0;
            for(int j = 0; j < M; j++){
                
                local_score += scores[i*N+j];
                if(local_score > toReach && reached != 0){
                    times[i] = j;
                    reached = 1;
                }
            }
        }
    }
}


void compute_histogram(int N, int M, int *times, int *hist){
    
    #pragma omp parallel for
    for(int i = 0; i < M; i++){
        hist[i] = 0;
    }
    
    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        #pragma omp atomic
        hist[times[i]]++;
    }

}