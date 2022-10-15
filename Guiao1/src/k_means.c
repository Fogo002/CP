#include "../include/k_means.h"

#define N 10000000 
#define K 4


void inicializa(float** pontos_x,float** pontos_y,float** cluster_x,float** cluster_y) {

    *pontos_x = malloc(N*sizeof(float));
    *pontos_y = malloc(N*sizeof(float));
    *cluster_x = malloc(K*sizeof(float));
    *cluster_y = malloc(K*sizeof(float));

    srand(10);
    for(int i = 0; i < N; i++) {
        (*pontos_x)[i] = (float) rand() / RAND_MAX;
        (*pontos_y)[i] = (float) rand() / RAND_MAX;
    }
    for(int i = 0; i < K; i++) {
        (*cluster_x)[i] = (*pontos_x)[i];
        (*cluster_y)[i] = (*pontos_y)[i];
    }
}

int main(){
    float* pontos_x;
    float* pontos_y;
    float* cluster_x;
    float* cluster_y;
    inicializa(&pontos_x,&pontos_y,&cluster_x,&cluster_y);
    
    for(int i = 0; i < N; i++) {
        //printf("Ponto %d :%f , %f \n",i,pontos_x[i],pontos_y[i]); 
    }

    return 0;
}