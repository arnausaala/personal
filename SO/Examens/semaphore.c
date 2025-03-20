#include "semaphore.h"

void sem_init(Semaphore *s, int N){
    pthread_mutex_init(&s->lock, NULL);
    pthread_cond_init(&s->cond, NULL);
    s->n = N;
}

void sem_wait(Semaphore *s){
    pthread_mutex_lock(&s->lock);
    while(s->n <= 0){
        pthread_cond_wait(&s->cond, &s->lock);
    }
    s->n--;
    pthread_mutex_unlock(&s->lock);
}

void sem_signal(Semaphore *s){
    pthread_mutex_lock(&s->lock);
    s->n++;
    pthread_cond_signal(&s->cond);
    pthread_mutex_unlock(&s->lock);
}

void sem_post(Semaphore *s){
    pthread_mutex_lock(&s->lock);
    s->n++;
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->lock);
}

void sem_destroy(Semaphore *s){
    pthread_mutex_destroy(&s->lock);
    pthread_cond_destroy(&s->cond);
}