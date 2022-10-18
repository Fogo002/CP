#include "../include/k_means.h"

#define N 10000000
#define K 4


void cluster_distrib(float** pontos_x,float** pontos_y,float** old_centroid_x,float** old_centroid_y,int** cluster_atribution,float** centroids_x,float** centroids_y,int** cluster_size) {

    int cluster_atual=0;
    float min_dist,tmp;

    for(int i = 0; i < K; i++){
        (*cluster_size)[i] = 0.0;
    }

    for(int i = 0; i < N; i++) {
        cluster_atual=0;
        min_dist = 2;
        for(int j = 0; j < K ; j++){
            
            tmp = euclidean_distance((*centroids_x)[j],(*centroids_y)[j],(*pontos_x)[i],(*pontos_y)[i]);

            if(tmp < min_dist){
                min_dist = tmp;
                cluster_atual = j;
            }
        }
        (*cluster_atribution)[i] = cluster_atual;
        (*cluster_size)[cluster_atual]++;
    }
}


//Podemos alterar o pontos_x e pontos_y para serem apenas um array de tamanho N*2 e acrescentar a o cluster_atibution
// Seria algo [x,y,att,x1,x2,att2...xn,yn,attn] de N*3
void inicializa(float** pontos_x,float** pontos_y,float** old_centroid_x,float** old_centroid_y,int** cluster_atribution,
                float** centroids_x,float** centroids_y,int** cluster_size) {

    int cluster_atual=0;
    float min_dist,tmp;

    *pontos_x = malloc(N*sizeof(float));
    *pontos_y = malloc(N*sizeof(float));
    *old_centroid_x = malloc(K*sizeof(float));
    *old_centroid_y = malloc(K*sizeof(float));
    *cluster_atribution = malloc(N*sizeof(int));
    *cluster_size= malloc(K*sizeof(int));
    *centroids_x = malloc(K*sizeof(float));
    *centroids_y = malloc(K*sizeof(float));

    srand(10);
    for(int i = 0; i < N; i++) {
        (*pontos_x)[i] = (float) rand() / RAND_MAX;
        (*pontos_y)[i] = (float) rand() / RAND_MAX;
    }
    for(int i = 0; i < K; i++) {
        (*old_centroid_x)[i] = (*pontos_x)[i];
        (*old_centroid_y)[i] = (*pontos_y)[i];

        (*centroids_x)[i] = (*pontos_x)[i];
        (*centroids_y)[i] = (*pontos_y)[i];

        (*cluster_size)[i] = 0;

    }
    cluster_distrib(&(*pontos_x),&(*pontos_y),&(*old_centroid_x),&(*old_centroid_y),&(*cluster_atribution),&(*centroids_x),&(*centroids_y),&(*cluster_size));
}

//Melhor percorrer um vez ou K vezes evitando acessos a memoria? Testar
int calculate_centroid(float** pontos_x,float** pontos_y,float** old_centroid_x,float** old_centroid_y,int** cluster_atribution,
                        float** centroids_x,float** centroids_y,int** cluster_size){
    int end = 1;
    for(int i = 0; i < K; i++){
        (*old_centroid_x)[i] = (*centroids_x)[i];
        (*old_centroid_y)[i] = (*centroids_y)[i];
        (*centroids_x)[i] = 0.0;
        (*centroids_y)[i] = 0.0;
        
    }


    for(int i = 0; i < N; i++){
        int cluster = (*cluster_atribution)[i];
        (*centroids_x)[cluster] += (*pontos_x)[i];
        (*centroids_y)[cluster] += (*pontos_y)[i];
    }

    for(int i = 0; i < K; i++){
        (*centroids_x)[i] = (*centroids_x)[i]/(*cluster_size)[i];
        (*centroids_y)[i] = (*centroids_y)[i]/(*cluster_size)[i];
        if(((*centroids_x)[i] != (*old_centroid_x)[i]) || ((*centroids_y)[i] !=(*old_centroid_y)[i])) end = 0;
    }
    return end;
}

int main(){
    float* pontos_x;
    float* pontos_y;
    float* old_centroid_x;
    float* old_centroid_y;
    int* cluster_atribution;
    int* cluster_size;
    float* centroids_x;
    float* centroids_y;
    inicializa(&pontos_x,&pontos_y,&old_centroid_x,&old_centroid_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);

    
    int end = 0;
    int iteracoes = 0;
    while(end == 0){
        printf("Iteração nº %d\n",iteracoes);
        end = calculate_centroid(&pontos_x,&pontos_y,&old_centroid_x,&old_centroid_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
        cluster_distrib(&pontos_x,&pontos_y,&old_centroid_x,&old_centroid_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
        iteracoes++;
    }
    for(int i = 0; i < K; i++) {
        printf("Cluster %d :%f , %f \nCentroide: %f %f\nSize: %d\n\n",i,old_centroid_x[i],old_centroid_y[i],centroids_x[i],centroids_y[i],cluster_size[i]); 
    }
    return 0;
}