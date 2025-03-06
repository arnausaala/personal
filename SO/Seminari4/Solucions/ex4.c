#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
#include <semaphore.h>

#define NUM_RESOURCES 5
#define NUM_THREADS 20

sem_t sem; 

int nThreadsAccessing = 0;

int getFromDatabase() {
    nThreadsAccessing++;
    printf("Number of Threads accessing the DB = %d\n", nThreadsAccessing);
    if (nThreadsAccessing > NUM_RESOURCES) _exit(1);
    usleep(rand() % 10000);
    nThreadsAccessing--;
    return rand();
}

void *thread_function(void *arg) {
    int thread_id = *(int *)arg;
    free(arg);

    printf("Thread %d: Requesting access to DB\n", thread_id);

    sem_wait(&sem); // Wait for a resource to be available (decrement semaphore)
    printf("Thread %d: Access granted.\n", thread_id);

    // Simulate using the resource
    getFromDatabase();

    printf("Thread %d: Query completed.\n", thread_id);
    sem_post(&sem); // Release the resource (increment semaphore)

    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];

    // Initialize the semaphore:
    // The second parameter is 0 (local to this process) and the initial value is NUM_RESOURCES.
    sem_init(&sem, 0, NUM_RESOURCES);

    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        int *thread_id = malloc(sizeof(int));
        if (thread_id == NULL) {
            perror("Failed to allocate memory");
            exit(EXIT_FAILURE);
        }
        *thread_id = i;
        pthread_create(&threads[i], NULL, thread_function, thread_id);
    }

    // Wait for threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Database resource management finished.\n");

    // Destroy the semaphore
    sem_destroy(&sem);

    return 0;
}
