void invert_horitzontal(int *img, int Nx, int Ny){
    for(int offset = 0; offset < Nx*Ny; offset+=Nx*Ny/4){
        #pragma acc enter data copyin(img[offset:(Nx/4)*Ny])
        #pragma acc parallel loop collapse(2) present(img[offset:Nx*Ny/4])
        for(int i = 0; i < Nx/4; i++){
            for(int j = 0; j < Ny/2; j++){
                int aux = img[offset+i*Ny+j];
                img[offset+i*Ny+j] = img[offset+i*Ny+(Ny-1-j)];
                img[offset+i*Ny+(Ny-1-j)] = aux;
            }
        }
        #pragma acc exit data copyout(img[offset:Nx*Ny/4])
    }
    #pragma acc wait
}