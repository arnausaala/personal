#include <pthread.h>

typedef struct{
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int n;
} Semaphore;

void sem_init(Semaphore *s, int N);
void sem_wait(Semaphore *s);
void sem_signal(Semaphore *s);
void sem_post(Semaphore *s);
void sem_destroy(Semaphore *s);