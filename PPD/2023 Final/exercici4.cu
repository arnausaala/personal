void MaxEvenOdd(unsigned* h_array, unsigned* h_maxEven, unsigned* h_maxOdd){
    *h_maxEven = 0;
    *h_maxOdd = 0;
    for(int i=0; i<512; ++i){
        unsigned val = h_array[i];
        if(i%2==0 && maxEven < val){
            *h_maxEven = val;
        }
        else if( maxOdd < val){
            *h_maxOdd = val;
        }
    }
}


__global__ MaxEvenOdd(unsigned* d_array, unsigned* d_maxEven, unsigned* d_maxOdd){
    
    
    
}

MaxEvenOdd<<1,512,512*sizeof(unsigned)>>(d_array, d_maxEven, d_maxOdd);