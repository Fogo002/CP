#include <math.h>

#include "../include/utils.h"

//Versão da função da distância euclidiana sem a raiz quadrada uma vez que os valoers
//não precisam de ser exatos, apenas que nos dê uma ordem de comparação da dsitância 
//dos pontos.
float euclidean_distance(float ponto1_x,float ponto1_y,float ponto2_x,float ponto2_y){
    return  (float) (ponto2_x - ponto1_x)*(ponto2_x - ponto1_x)+ (ponto2_y - ponto1_y)*(ponto2_y - ponto1_y);
}