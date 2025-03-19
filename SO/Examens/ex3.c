#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <fcntl.h>
#include <math.h>

#define SIZE 10
#define N 3

pthread_t tid1[SIZE];
pthread_mutex_t lock;


void NormalizeVectorSequential(double *u_norm){
    pthread_mutex_lock(&lock);
    printf("Norma del vector [%.2f, %.2f, %.2f]\n", (float)u_norm[0], (float)u_norm[1], (float)u_norm[2]);
    double u_norm_2 = 0;
    for(int i = 0; i < N; i++){
        u_norm_2 += u_norm[i]*u_norm[i];
    }

    for(int i = 0; i < N; i++){
        u_norm[i] = u_norm[i]/sqrt(u_norm_2);
        printf("u_norm[%d] = %f\n", i, (float)u_norm[i]);
    }
    printf("\n");
    pthread_mutex_unlock(&lock);
}

int main(){
    pthread_mutex_init(&lock, NULL);
    double u_norm_arg[SIZE][N];
    for(int i = 0; i < SIZE; i++){
        for(int j = 0; j < N; j++){
            u_norm_arg[i][j] = rand()%10;
            printf("%.2f   ", (float)u_norm_arg[i][j]);
        }
        printf("\n");
    }

    for(int i = 0; i < SIZE; i++){
        pthread_create(&tid1[i], NULL, NormalizeVectorSequential, (double*)u_norm_arg[i]);
    }
    for(int i = 0; i < SIZE; i++){
        pthread_join(tid1[i], NULL);
    }

    pthread_mutex_destroy(&lock);
    return 0;
}