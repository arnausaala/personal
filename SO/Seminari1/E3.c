#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

int main(){
    int nums[100];
    ssize_t bytesRead;

    int writeFd = open("nums.dat", O_CREAT | O_WRONLY, 0644);
    if(writeFd == -1){
        printf("Error abriendo el archivo writeFd\n");
        return 1;
    }

    for(int i = 0; i < 100; i++){
        nums[i] = i+1;
    }

    ssize_t bytesWritten = write(writeFd, nums, sizeof(nums));
    if (bytesWritten != sizeof(nums)) {
        printf("Error escribiendo en el archivo. Escribí %ld bytes, pero esperábamos %ld\n", bytesWritten, sizeof(nums));
        close(writeFd);
        return 1;
    }
    close(writeFd);

    int readFd = open("nums.dat", O_RDONLY);
    if(readFd == -1){
        printf("Error abriendo el archivo readFd\n");
        return 1;
    }

    bytesRead = read(readFd, nums, sizeof(nums));
    close(readFd);
    
    printf("\n\nbytes read = %ld\n\n", bytesRead);

    for(int i = 0; i < 100; i++){
        if(i < 9){
            printf("0%d, ", nums[i]);
        }
        else if(i % 10 == 0){
            printf("\n%d, ", nums[i]);
        }
        else{
            printf("%d, ", nums[i]);
        }
    }

    if(bytesRead != sizeof(nums)){
        printf("\nSe ha cometido un error intentando leer los datos\n");
        return 1;
    }

    printf("\nSe han guardado todos los valores en memoria correctamente\n");
    return 0;
}