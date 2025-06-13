int maxloc(unsigned* v, int N){
    int pos = 0;
    int NT = omp_get_num_threads();
    int max_pos[NT];
    int max = 0;

    #pragma omp parallel firstprivate(max)
    {
        int thread = omp_get_thread_num();
        max_pos[thread] = 0;

        #pragma omp for
        for(int i = 0; i < N; i++){
            if(v[i] > max){
                max = v[i];
                max_pos[thread] = i;
            }
        }
    }

    for(int i = 0; i < NT; i++){
        if(max_pos[i] > max){
            max = v[max_pos[i]];
            pos = max_pos[i];
        }
    }

    return pos;
}