#include <unistd.h>
#include <stdio.h>

int main(){
    char input;
    ssize_t res;

    res = read(0, &input, 1);

    if(res == -1){
        write(2, "Error\n", 6);
        return 1;
    }
    printf("\n");
    write(1, &input, 1);
    printf("\n\n");
    return 0;
}