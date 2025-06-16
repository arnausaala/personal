void blockInvert(int* img_in, int * img_out , int nx, int ny){

    double bloque = nx/8;

    #pragma acc enter data copyin(img_in[0:nx*ny]) create(img_out[0:nx*ny])
    #pragma acc parallel loop collapse(2)
    
    for(int i = 0; i < nx; i+=bloque){
        for(int k = i*bloque; k < (i+1)*bloque; k++){
            for(int j = 0; j < ny; j++){
                if(i % 2 == 0){
                    img_out[k*nx+j] = 255 - img_in[k*nx+j];
                }
                else{
                    img_out[k*nx+j] = img_in[k*nx+j];
                }
            }
        }
    }
    #pragma acc wait 
    #pragma acc exit data copyout(img_out[0:nx*ny]) delete(img_in[0:nx*ny])
}