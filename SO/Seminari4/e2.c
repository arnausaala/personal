#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#include <signal.h>
#include <pthread.h>

#define N 100
int A[N];
pthread_t tid[N];
pthread_mutex_t lock;

void square(void* arg){
    pthread_mutex_lock(&lock);
    int* i = (int*)arg;
    A[*i] = (*i) * (*i);
    pthread_mutex_unlock(&lock);
}

int main(){
    pthread_mutex_init(&lock, NULL);
    // PRINT ARRAY BEFORE EXECUTION
    for(int i = 0; i < N; i++){
        if(i % 10 == 0 && i != 0){
            printf("\n");
        }
        printf("%4d ", A[i]);
    }
    printf("\n\n");

    int indices[N];
    for(int i = 0; i < N; i++){
        indices[i] = i;
        pthread_create(&tid[i], NULL, square, &indices[i]);
    }

    for(int i = 0; i < N; i++){
        pthread_join(&tid[i], NULL);
    }

    pthread_mutex_destroy(&lock);

    // PRINT ARRAY AFTER EXECUTION
    for(int i = 0; i < N; i++){
        if(i % 10 == 0 && i != 0){
            printf("\n");
        }
        printf("%4d ", A[i]);
    }
    printf("\n");

    printf("\nPrograma terminado con éxito\n");
    return 0;
}