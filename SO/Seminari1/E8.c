#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
    char buffer[9];
    ssize_t bytesRead;
    int num;

    write(STDOUT_FILENO, "Enter an integer (max 8 digits): ", 32);

    bytesRead = read(STDIN_FILENO, buffer, sizeof(buffer) - 1);
    if(bytesRead == -1){
        write(STDERR_FILENO, "Error reading input\n", 20);
        return 1;
    }

    buffer[bytesRead] = '\0';

    // Si el último carácter es un salto de línea, lo eliminamos
    if (buffer[bytesRead - 1] == '\n') {
        buffer[bytesRead - 1] = '\0';
    }

    // Comprobamos que la longitud no sea 0 y no se exceda de los 8 caracteres
    if (bytesRead == 0 || bytesRead > 8) {
        write(STDERR_FILENO, "Invalid input: Please enter a valid integer\n", 45);
        return 1;
    }

    if(sscanf(buffer, "%d", &num) != 1){
        write(STDERR_FILENO, "Error con el sscanf\n", 20);
        return 1;
    }

    int res = num * 2;
    printf("num = %d\nres = 2*%d = %d\n", num, num, res); // Cambié '%n' por '%d'

    return 0;
}
