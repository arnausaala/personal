#include <stdio.h>
#include <unistd.h>
#include <string.h>

#define N 100

int main(){
    char buffer[N];
    ssize_t nBytesRead = read(0, &buffer, N-1);
    if(nBytesRead > 0){
        buffer[nBytesRead] = '\0';
    }
    else{
        printf("Error leyendo\n");
        return 1;
    }

    printf("Cadena leída: ");
    for (int i = 0; i < nBytesRead; i++) {
        printf("%c", buffer[i]);
    }
}