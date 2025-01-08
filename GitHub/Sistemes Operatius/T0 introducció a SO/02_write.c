#include<stdio.h>
#include<string.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    char *s = "hola\n";
    printf(s);
    write(1, s, strlen(s)); // 1 is the standard output

    int n = 875770417;
    printf("%d\n", n);
    printf("%x\n", n);

    write(1, &n, sizeof(n));

    char sMine[100];
    sprintf(sMine, "%d", n);

    write(1, sMine, strlen(sMine));

    printf("\n");
}