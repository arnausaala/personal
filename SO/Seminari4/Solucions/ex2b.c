#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

int A[100];

void * thr_func(void * arg)
{
    int i = *(int*)arg;
    free(arg);
    A[i] = i*i;
    printf("Thread %d, with its square: %d\n", i, A[i]);
    return NULL;
}

int main(void)
{
    pthread_t tid[100];
    for (int i = 0; i < 100; i++)
    {
        int *arg = malloc(sizeof(int));
        *arg = i;
        // We send a pointer (arg) to a copy of the value. As it is a copy for each thread, no issues will be found.
        pthread_create(&tid[i], NULL, thr_func, arg);
    }
    for (int i = 0; i < 100; i++)
    {
        pthread_join(tid[i], NULL);
    }

    return 0;
}