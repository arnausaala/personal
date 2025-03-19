#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#define N 10

int main(){
    int buffer[N];
    int fd = open("nums.dat", O_RDONLY | O_TRUNC, 0644);
    if(fd == -1){
        printf("Error abriendo el archivo\n");
        return 1;
    }
    for(int i = 0; i < 10; i++){
        read(fd, &buffer, sizeof(int));
        printf("%d\n", buffer[0]);
        //write(1, &buffer, sizeof(int));
    }
    close(fd);
    return 0;
}