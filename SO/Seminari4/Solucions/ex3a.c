#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

typedef struct {
    int i;
    int j;
} Indexes;

int array[10];

void * move(void * arg) {
    Indexes *ind = (Indexes *) arg;
    int i = ind->i;
    int j = ind->j;
    free(arg);

    if (array[i] > 0) {
        array[i]--;
        array[j]++;
    }
    return NULL;
}

int main (int argc, char *argv[]) {
    // Initialize the array with random values between 0 and 2
    for (int i = 0; i < 10; i++) {
        array[i] = rand() % 3;
        printf("array[%d] = %d\n", i, array[i]);
    }

    pthread_t tid[5];
    int t_count = 0;

    // Create threads for even-indexed cells to move a particle to the next cell
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0 && (i + 1) < 10) {  
            Indexes *indx = malloc(sizeof(Indexes));
            indx->i = i;
            indx->j = i + 1;
            pthread_create(&tid[t_count], NULL, move, (void *) indx);
            t_count++;
        }
    }

    // Wait for all threads to finish
    for (int i = 0; i < t_count; i++) {
        pthread_join(tid[i], NULL);
    }

    printf("After the movements.\n");
    for (int i = 0; i < 10; i++) {
        printf("array[%d] = %d\n", i, array[i]);
    }

    return 0;
}
