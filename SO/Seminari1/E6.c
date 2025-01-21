#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int main(){
    const char *filename = "Proves/input.txt";
    char buffer;
    ssize_t readBytes;

    int in = open(filename, O_RDONLY);
    if(in == -1){
        printf("Error al abrir el fichero de entrada\n");
        return 1;
    }
    
    int out = open("Proves/output.txt", O_CREAT | O_WRONLY, 0644);
    if(out == -1){
        close(in);
        printf("Error al abrir el fichero de salida\n");
        return 1;
    }

    while((readBytes = read(in, &buffer, 1)) > 0){
        write(out, &buffer, 1);
    }

    close(in);
    close(out);
    printf("Saliendo del programa con exito...\n");

    return 0;
}