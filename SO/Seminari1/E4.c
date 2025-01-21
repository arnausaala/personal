#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]){
    if(argc < 2){
        printf("Argumentos insuficientes\n");
        return 1;
    }

    char *msg = argv[1];
    
    if(strlen(msg) > 0){
        printf("todo correcto\n%zu\n", strlen(msg));
        return 0;
    }
    printf("El texto esta vacío\n");
    return 1;
}