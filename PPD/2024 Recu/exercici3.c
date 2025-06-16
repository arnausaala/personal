#define ind(i, j, n) (j*n + i)

int main(int argc, char** argv) {
    int M = 1024;
    int N = 1024;
    double *im = (double *)malloc(N*M*sizeof(double));
    double *im_new = (double *)malloc(N*M*sizeof(double));

    initialize_image(im, M, N);

    double change = 1000;
    int iter = 0;

    while (change > 0.00001 && iter < 5000) {
        
        #pragma acc parallel loop collapse(2) copyin(im[0:N*M]) create(im_new[0:N*M])
        for (int i = 1; i < M-1; i++) {
            for (int j = 1; j < N-1; j++) {
                im_new[ind(i, j, N)] = (im[ind(i , j , N)] + im[ind(i , j-1, N)] +
                                        im[ind(i-1, j-1, N)] + im[ind(i+1, j , N)] +
                                        im[ind(i , j+1, N)]) / 5.0;
            }
        }

        change = 0;

        #pragma acc wait
        #pragma acc update device(im_new)

        #pragma acc parallel loop collapse(2) present(im[0:N*M], im_new[0:N*M])
        for (int i = 1; i < M-1; i++) {
            for (int j = 1; j < N-1; j++) {
                change += fabs(im_new[ind(i, j, N)] - im[ind(i, j , N)]);
                im[ind(i, j, N)] = im_new[ind(i, j, N)];
            }
        }

        if ((iter % 50) == 0) {
            #pragma acc update host(im[0:N*M])
            save_image(im, M, N, iter);
        }

        iter++;
    }

    #pragma acc wait
    #pragma acc exit data copyout(im[0:N*M]) delete(im_new[0:N*M])
    save_image(im, M, N, iter);

    free(im);
    free(im_new);
    return 0;
}