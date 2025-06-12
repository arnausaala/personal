int maxloc(unsigned* v, int N){
    int pos = 0;
    unsigned max = 0;
    int num_threads = omp_get_max_threads();
    int local_max[num_threads];

    #pragma omp parallel firstprivate(max)
    {

        int id = omp_get_thread_num();
        local_max[id] = 0;

        #pragma omp for
        for(int i = 0; i < N; i++){
            if(v[i] > max){
                max = v[i];
                local_max[id] = i;
            }
        }
    }

    for(int i = 0; i < num_threads; i++){
        if(v[local_max[i]] > max){
            max = v[local_max[i]];
            pos = local_max[i];
        }
    }

    return pos;
}