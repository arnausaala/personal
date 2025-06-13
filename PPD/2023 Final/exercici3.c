void invert(int *img, int Nx, int Ny)
{
    for (int i = 0; i < Nx / 2; ++i){
        for (int j = 0; j < Ny; ++j){
            
            // aux = img[i, j]
            int aux = img[i * Ny + j]; 

            // img[i, j] = img[Nx-1-i,j]
            img[i * Ny + j] = img[(Nx - 1 - i) * Ny + j];
            
            // img[Nx-1-i, j] = aux
            img[(Nx - 1 - i) * Ny + j] = aux; 
        }
    }
}

void invertGPU(int *img, int Nx, int Ny){
    int aux[Nx][Ny];

    // Paralelitzat amb 4 threads
    for (int i = 0; i < Nx / 2; ++i){
        for (int j = 0; j < Ny; ++j){
            int aux = img[i * Ny + j]; 
            img[i * Ny + j] = img[(Nx - 1 - i) * Ny + j];
            img[(Nx - 1 - i) * Ny + j] = aux;
        }
    }

    for(int i = 0; i < Nx; i++){
        for(int i = 0; i < Ny; i++)

    }

}