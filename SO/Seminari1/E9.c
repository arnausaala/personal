#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int main() {
    const char *filename = "Proves/nums.dat";
    int buffer[1024];
    ssize_t bytesRead;

    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Error al abrir el archivo");
        return 1;
    }
    int nums[10];
    int count = 0;
    bytesRead = 2*read(fd, nums, sizeof(nums));
    if (bytesRead == -1) {
        perror("Error al leer el archivo");
        close(fd);
        return 1;
    }
    printf("n = %zd\n", bytesRead);
    printf("s = %lu\n", sizeof(nums));

    if (bytesRead != sizeof(nums)) {
        fprintf(stderr, "El archivo no contiene exactamente 10 enteros\n");
        close(fd);
        return 1;
    }

    printf("Números multiplicados por 2:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", nums[i] * 2);
    }
    printf("\n");

    close(fd);
    return 0;
}
