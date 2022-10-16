#include <math.h>

#include "../include/utils.h"

float euclidean_distance(float ponto1_x,float ponto1_y,float ponto2_x,float ponto2_y){
    return  (float) sqrt(pow(ponto2_x - ponto1_x, 2)+ pow(ponto2_y - ponto1_y, 2));
}