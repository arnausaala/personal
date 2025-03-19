#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>

int counter = 0;
int end = 0;

pthread_t decr[3];
pthread_t incr;
pthread_mutex_t lock;


void* increment(void *a){
    printf("He entrado al increment\n");
    while(end == 0){
        pthread_mutex_lock(&lock);
        if(counter == 0){
            printf("BEFORE = %d     ", counter);
            counter += rand()%1000;
            printf("AFTER = %d\n", counter);
        }
        pthread_mutex_unlock(&lock);
        usleep(10000);
    }
    return NULL;
}

void* decrement(void *a){
    printf("He entrado al decrement\n");
    int id = (int)(long)a;
    int dec;
    if(id == 0){dec = 1;}
    else if(id == 1){dec = 5;}
    else{id = 10;}

    while(end == 0){
        pthread_mutex_lock(&lock);
        if(counter > 0){
            printf("before = %d     ", counter);
            counter -= dec;
            printf("after = %d\n", counter);
        }
        if(counter < 0){
            end = 1;
        }
        pthread_mutex_unlock(&lock);
        usleep(10000);
    }
    return NULL;
}




int main(){
    pthread_mutex_init(&lock, NULL);
    printf("end = %d\n", end);

    pthread_create(&incr, NULL, increment, NULL);
    printf("Se ha creado el thread increment\n");

    for(int i = 0; i < 3; i++){
        pthread_create(&decr[i], NULL, decrement, (void*)(long)i);
        printf("Se ha creado el thread decrement (%d)\n", i);
    }
    for(int i = 0; i < 2; i++){
        pthread_join(decr, NULL);
    }
    pthread_join(incr, NULL);

    pthread_mutex_destroy(&lock);
    
    printf("Cerrando el programa...\nend = %d\n", end);
    return 0;
}