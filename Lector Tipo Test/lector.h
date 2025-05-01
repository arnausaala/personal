#include "correccio.h"
#include "respostes.h"
#include "macros.h"


float comparar(int corr[PREGUNTES][RESPOSTES], int resp[2*PREGUNTES][RESPOSTES]){

    int correctes = 0, incorrectes = 0, no_contestades = 0, anul·lades = 0, contestades;
    float puntuacioPregunta, NF = 0;
    char resposta;

    for(int i = 0; i < 2*PREGUNTES; i+=2){
        printf("pregunta %d\n", (i/2)+1);
        contestades = 0;
        puntuacioPregunta = 0;
        resposta = '-';
        for(int j = 0; j < RESPOSTES; j++){
            if(resp[i][j] == 1 && resp[i+1][j] == 0){
                contestades++;
                if(contestades > 1){
                    puntuacioPregunta = 0;
                    printf("    ANUL·LADA      puntuació: 0.00\n");
                    anul·lades++;
                    goto pregunta_anulada;
                }
                
                switch (j)
                {
                case 0:
                    resposta = 'a';
                    break;
                case 1:
                    resposta = 'b';
                    break;
                case 2:
                    resposta = 'c';
                    break;
                case 3:
                    resposta = 'd';
                    break;
                case 4:
                    resposta = 'e';
                    break;
                case 5:
                    resposta = 'f';
                    break;
                }

                if(resp[i][j] == corr[i/2][j]){ // correcte
                    puntuacioPregunta += 1;
                }

                else{ // incorrecte
                    puntuacioPregunta -= 1/(float)PENALITZACIO;
                }
            }
        }
        if(contestades == 0){no_contestades++;}
        else if(puntuacioPregunta > 0){correctes++;}
        else if(puntuacioPregunta < 0){incorrectes++;}
        NF += puntuacioPregunta;
        printf("    resposta: %c", resposta);
        printf("    puntuació: %.2f\n", puntuacioPregunta);

        
    pregunta_anulada:
        continue;
    }

    if(NF < 0){NF = 0;}
    printf("\nRESUM\n    Correctes: %d\n    Incorrectes: %d\n    No contestades: %d\n    Anul·lades: %d\n",
        correctes, incorrectes, no_contestades, anul·lades);

    return NF;
}