#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>

#define BUFFER_SIZE 10

int fds[2];

void readFunction(const char *inputFile) {
    int fd = open(inputFile, O_RDONLY);
    if (fd < 0) {
        perror("Error opening input file");
        exit(1);
    }
    printf("1. Opened input file: %s\n", inputFile);
    
    char buffer[BUFFER_SIZE];
    ssize_t bytesRead;
    while ((bytesRead = read(fd, buffer, BUFFER_SIZE - 1)) > 0) {
        buffer[bytesRead] = '\0'; // Null-terminate the buffer
        int number = atoi(buffer); // Convert ASCII to integer
        write(fds[1], &number, sizeof(int));
    }
    
    close(fd);
    close(fds[1]);
}

void writeFunction(const char *outputFile) {
    int fd = open(outputFile, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("Error opening output file");
        exit(1);
    }
    printf("2. Opened output file: %s\n", outputFile);
    
    int number;
    while (read(fds[0], &number, sizeof(int)) > 0) {
        write(fd, &number, sizeof(int)); // Write binary integer to file
    }
    
    close(fd);
    close(fds[0]);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input file> <output file>\n", argv[0]);
        return 1;
    }
    
    if (pipe(fds) == -1) {
        perror("Error creating pipe");
        return 1;
    }
    
    int pid = fork();
    if (pid < 0) {
        perror("Error creating child process");
        return 1;
    }
    else if (pid > 0) { // Parent process
        close(fds[0]);
        readFunction(argv[1]);
        wait(NULL);
    }
    else { // Child process
        close(fds[1]);
        writeFunction(argv[2]);
    }
    
    printf("Program finished.\n");
    return 0;
}
