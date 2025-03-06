#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int i; 
    int j;
} Indexes;

int array[10];

// Global mutex to protect array modifications
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void * move(void * arg) {
    Indexes *ind = (Indexes *) arg;
    int i = ind->i;
    int j = ind->j;
    free(arg);
    
    // Lock the mutex to safely update the array
    pthread_mutex_lock(&mutex);
    if (array[i] > 0) {
        array[i]--;
        array[j]++;
    }
    pthread_mutex_unlock(&mutex);
    
    return NULL;
}

int main (int argc, char *argv[]) {
    srand(time(NULL));
    
    // Initialize the array with random values between 0 and 2
    for (int i = 0; i < 10; i++) {
        array[i] = rand() % 3;
        printf("array[%d] = %d\n", i, array[i]);
    }
    
    pthread_t tid[10];
    
    // Create 10 threads, each performing a random move
    for (int t = 0; t < 10; t++) {
        Indexes *indx = malloc(sizeof(Indexes));
        int src = rand() % 10;
        int dst = rand() % 10;
        // Ensure the destination is different from the source
        while (dst == src) {
            dst = rand() % 10;
        }
        indx->i = src;
        indx->j = dst;
        
        pthread_create(&tid[t], NULL, move, (void *) indx);
    }
    
    // Wait for all move operations to finish
    for (int t = 0; t < 10; t++) {
        pthread_join(tid[t], NULL);
    }
    
    printf("After the movements.\n");
    for (int i = 0; i < 10; i++) {
        printf("array[%d] = %d\n", i, array[i]);
    }
    
    return 0;
}
