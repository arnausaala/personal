#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#include <signal.h>
#include <pthread.h>

void thread_function(void *i){
    int i_int = (int)i;
    printf("soc el thread numero %d\n", i_int);
    sleep(1);
}

int main(){
    pthread_t tid[100];
    for(int i = 0; i < 100; i++){
        pthread_create(&tid[i], NULL, thread_function, (void*)i);
    }
    for(int i = 0; i < 100; i++){
        pthread_join(&tid[i], NULL);
    }
}