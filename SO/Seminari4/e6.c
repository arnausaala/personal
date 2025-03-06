#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define MIDA 5

int fitxer;
pthread_mutex_t mutex;
pthread_mutex_t mutex_bar;
pthread_cond_t cond_bar;
int comptador = 0;

void *escriure_fila(void *arg) {
    int index = *(int *)arg;
    int fila[MIDA];
    
    // Generar nombres aleatoris
    srand(time(NULL) + index);
    for (int i = 0; i < MIDA; i++) {
        fila[i] = rand() % 10;
    }
    
    // Mostrar la fila i el fil responsable
    printf("Fil %d: ", index);
    for (int i = 0; i < MIDA; i++) {
        printf("%d ", fila[i]);
    }
    printf("\n");
    
    usleep(10000); // Petita pausa per evitar accessos seqüencials
    
    // Escriure al fitxer amb mutex per evitar col·lisions
    pthread_mutex_lock(&mutex);
    lseek(fitxer, index * MIDA * sizeof(int), SEEK_SET);
    write(fitxer, fila, MIDA * sizeof(int));
    pthread_mutex_unlock(&mutex);
    
    // Esperar a que tots hagin escrit
    pthread_mutex_lock(&mutex_bar);
    comptador++;
    if (comptador == MIDA) {
        pthread_cond_broadcast(&cond_bar);
    } else {
        while (comptador < MIDA) {
            pthread_cond_wait(&cond_bar, &mutex_bar);
        }
    }
    pthread_mutex_unlock(&mutex_bar);
    
    // Llegir la diagonal
    int valor;
    lseek(fitxer, (index * MIDA + index) * sizeof(int), SEEK_SET);
    read(fitxer, &valor, sizeof(int));
    
    // Mostrar el valor llegit
    printf("Fil %d ha llegit la diagonal: %d\n", index, valor);
    return NULL;
}

int main() {
    pthread_t fils[MIDA];
    int indexos[MIDA];
    
    // Obrir fitxer
    fitxer = open("matriu.bin", O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fitxer == -1) {
        perror("Error obrint el fitxer");
        return 1;
    }
    
    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_init(&mutex_bar, NULL);
    pthread_cond_init(&cond_bar, NULL);
    
    // Crear els fils
    for (int i = 0; i < MIDA; i++) {
        indexos[i] = i;
        pthread_create(&fils[i], NULL, escriure_fila, &indexos[i]);
    }
    
    // Esperar que acabin
    for (int i = 0; i < MIDA; i++) {
        pthread_join(fils[i], NULL);
    }
    
    pthread_mutex_destroy(&mutex);
    pthread_mutex_destroy(&mutex_bar);
    pthread_cond_destroy(&cond_bar);
    close(fitxer);
    
    return 0;
}
