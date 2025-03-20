#include "barrier.h"

void barrier_wait(Barrier *b, int N){
    pthread_mutex_init(&b->lock, NULL);
    pthread_cond_init(&b->cond, NULL);
    b->missingToArrive = N;
}

void barrier_wait(Barrier *b){
    pthread_mutex_lock(&b->lock);
    b->missingToArrive--;
    if(b->missingToArrive <= 0){
        pthread_cond_broadcast(&b->cond);
    }
    else{
        pthread_mutex_wait(&b->cond, &b->lock);
    }
    pthread_mutex_lock(&b->lock);
}

void barrier_destroy(Barrier *b){
    pthread_mutex_destroy(&b->lock);
    pthread_cond_destroy(&b->cond);
}