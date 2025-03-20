#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

int fd1[2]; // Padre -> Hijo
int fd2[2]; // Hijo -> Padre

int main(int argc, char *argv[]){
    
    pipe(fd1);
    pipe(fd2);

    int n = fork();
    if(n < 0){
        printf("Error en el fork\n");
        return 1;
    }
    else if(n > 0){ // Padre
        close(fd1[0]);
        close(fd2[1]);
        int fd = open(argv[1], O_RDONLY, 0644);
        char buffer[100];
        while(read(fd, buffer, sizeof(buffer)) > 0){
            write(fd1[1], &buffer, sizeof(buffer));
        }
        int total;
        read(fd2[0], &total, sizeof(int));
        write(1, &total, sizeof(int));
        printf("palabras: %d\n", total);

        close(fd1[1]);
        close(fd2[0]);
    }
    else{ // Hijo
        close(fd1[1]);
        close(fd2[0]);
        char buffer2[100];
        int count = 1;
        while(read(fd1[0], buffer2, sizeof(buffer2)) > 0){
            for(int i = 0; i < 100; i++){
                if(buffer2[i] == ' '){
                    count++;
                }
            }
        }
        write(fd2[1], &count, sizeof(int));
    }
    close(fd1[0]);
    close(fd2[1]);
}