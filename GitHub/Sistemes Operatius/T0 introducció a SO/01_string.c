#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    printf("hola {argc = %d}\n", argc);
    for(int i = 0; i < argc; i++) {
        printf("argv[%d] = %s\n", i, argv[i]);
        if(strcmp(argv[i], "hola") == 0) {
            printf("hi ha hola\n");
        }
    }
}