#include <pthread.h>

typedef struct{
    int missingToArrive;
    pthread_mutex_t lock;
    pthread_cond_t cond;
} Barrier;

void barrier_init(Barrier *b, int N);
void barrier_wait(Barrier *b);
void barrier_destroy(Barrier *b);