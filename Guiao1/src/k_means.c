#include "../include/k_means.h"

#define N 10000000 
#define K 4



//Podemos alterar o pontos_x e pontos_y para serem apenas um array de tamanho N*2
void inicializa(float** pontos_x,float** pontos_y,float** cluster_x,float** cluster_y,int** cluster_atribution) {
    int cluster_atual=0;
    float min_dist,tmp;
    *pontos_x = malloc(N*sizeof(float));
    *pontos_y = malloc(N*sizeof(float));
    *cluster_x = malloc(K*sizeof(float));
    *cluster_y = malloc(K*sizeof(float));
    *cluster_atribution= malloc(N*sizeof(int));

    srand(10);
    for(int i = 0; i < N; i++) {
        (*pontos_x)[i] = (float) rand() / RAND_MAX;
        (*pontos_y)[i] = (float) rand() / RAND_MAX;
    }
    for(int i = 0; i < K; i++) {
        (*cluster_x)[i] = (*pontos_x)[i];
        (*cluster_y)[i] = (*pontos_y)[i];
    }
    for(int i = 0; i < N; i++) {
        cluster_atual=0;
        min_dist = euclidean_distance((*cluster_x)[0],(*cluster_y)[0],(*pontos_x)[i],(*pontos_y)[i]);
        for(int j = 1; j < K ; j++){

            tmp = euclidean_distance((*cluster_x)[j],(*cluster_y)[j],(*pontos_x)[i],(*pontos_y)[i]);
            if(tmp < min_dist){
                min_dist = tmp;
                cluster_atual = j;
            }
        }
        (*cluster_atribution)[i] = cluster_atual;

    }
}

int main(){
    float* pontos_x;
    float* pontos_y;
    float* cluster_x;
    float* cluster_y;
    int* cluster_atribution;
    inicializa(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution);
    /*
    for(int i = 0; i < N; i++) {
        printf("Ponto %d :%f , %f \nCluster: %d\n\n",i,pontos_x[i],pontos_y[i],cluster_atribution[i]); 
    }
    */
    return 0;
}