#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int n;
    scanf("%d", &n);
    printf("Read number n = %d\n", n);
    int nBytesRead;

    char s[100];
    // Abans de llegir
    // \0\0\0\0\0\0\0\0\0\0\0\0\0
    nBytesRead= read(0, s, 100); //Read from standard input
    s[nBytesRead] = '\0';
    printf("Bytes read = %d\n", nBytesRead);
    // bon dia\n\0
    printf("%s\n", s);

    nBytesRead= read(10, s, 100); //Read another text from standard input
    printf("Bytes read = %d\n", nBytesRead);
    //s[nBytesRead] = '\0';

    //After read
    // adeu\nia\n\0
    printf("%s\n", s);

}