#include "lector.h"
#include <stdio.h>

int main(){
    float NF = comparar(correccio, respostes);
    if(PREGUNTES == 10){
        printf("\nNF = %.2f\n", NF);
    }
    else{
        printf("\nNF = %.2f / %d\n", NF, PREGUNTES);
        printf("\nNF NORMALITZADA = %.2f\n\n", NF * 10/(float)PREGUNTES);
    }
}