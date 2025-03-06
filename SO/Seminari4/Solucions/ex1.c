#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

void * thr_func(void * arg)
{
    printf("Thread %d, with PID: %d\n", *(int *) arg, getpid());
    sleep(1);
}

int main(void)
{
    pthread_t tid[100];
    for (int i = 0; i < 100; i++)
    {
        pthread_create(&tid[i], NULL, thr_func, &i);
    }
    for (int i = 0; i < 100; i++)
    {
        pthread_join(tid[i], NULL);
    }

    return 0;
}