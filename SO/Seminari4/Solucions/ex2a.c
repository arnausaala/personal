#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

int A[100];

void * thr_func(void * arg)
{
    int i = *(int*)arg;
    A[i] = i*i;
    printf("Thread %d, with its square: %d\n", i, A[i]);
    return NULL;
}

int main(void)
{
    pthread_t tid[100];
    for (int i = 0; i < 100; i++)
    {
        // We send the address of the variable, the same address to all threads.
        pthread_create(&tid[i], NULL, thr_func, &i);
    }
    for (int i = 0; i < 100; i++)
    {
        pthread_join(tid[i], NULL);
    }

    return 0;
}