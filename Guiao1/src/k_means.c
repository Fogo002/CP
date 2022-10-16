#include "../include/k_means.h"

#define N 10000000
#define K 4


void cluster_distrib(float** pontos_x,float** pontos_y,float** cluster_x,float** cluster_y,int** cluster_atribution,float** centroids_x,float** centroids_y,int** cluster_size) {

    int cluster_atual=0;
    float min_dist,tmp;

    for(int i = 0; i < K; i++){
        (*cluster_size)[i] = 0.0;
        
    }
    //printf("Distribuição pelos clusters\n");

    for(int i = 0; i < N; i++) {
        cluster_atual=0;
        min_dist = euclidean_distance((*cluster_x)[0],(*cluster_y)[0],(*pontos_x)[i],(*pontos_y)[i]);
        //printf("Ponto x: %f , y:%f\n",(*pontos_x)[i],(*pontos_y)[i]);
        //printf("Tentar cluster 0. Dist: %f\n",min_dist);

        for(int j = 1; j < K ; j++){
            
            tmp = euclidean_distance((*cluster_x)[j],(*cluster_y)[j],(*pontos_x)[i],(*pontos_y)[i]);
            //printf("Tentar cluster %d. Dist: %f\n",j,tmp);
            if(tmp < min_dist){
                min_dist = tmp;
                cluster_atual = j;
            }
        }
        (*cluster_atribution)[i] = cluster_atual;
        (*cluster_size)[cluster_atual]++;
        //printf("Cluster: %d\n\n\n",(*cluster_atribution)[i]);
    }
    //printf("Distribuição atual\n");
    for(int i = 0; i < N; i++) {
        //printf("Ponto x: %f , y: %f --->%d\n",(*pontos_x)[i],(*pontos_y)[i],(*cluster_atribution)[i]);

    }


}


//Podemos alterar o pontos_x e pontos_y para serem apenas um array de tamanho N*2 e acrescentar a o cluster_atibution
// Seria algo [x,y,att,x1,x2,att2...xn,yn,attn] de N*3
void inicializa(float** pontos_x,float** pontos_y,float** cluster_x,float** cluster_y,int** cluster_atribution,
                float** centroids_x,float** centroids_y,int** cluster_size) {

    int cluster_atual=0;
    float min_dist,tmp;

    *pontos_x = malloc(N*sizeof(float));
    *pontos_y = malloc(N*sizeof(float));
    *cluster_x = malloc(K*sizeof(float));
    *cluster_y = malloc(K*sizeof(float));
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
        (*cluster_x)[i] = (*pontos_x)[i];
        (*cluster_y)[i] = (*pontos_y)[i];

        (*centroids_x)[i] = 0.0;
        (*centroids_y)[i] = 0.0;

        (*cluster_size)[i] = 0;

    }
    cluster_distrib(&(*pontos_x),&(*pontos_y),&(*cluster_x),&(*cluster_y),&(*cluster_atribution),&(*centroids_x),&(*centroids_y),&(*cluster_size));
}
//Melhor percorrer um vez ou K vezes evitando acessos a memoria? Testar
void calculate_centroid(float** pontos_x,float** pontos_y,float** cluster_x,float** cluster_y,int** cluster_atribution,
                        float** centroids_x,float** centroids_y,int** cluster_size){
    for(int i = 0; i < K; i++){
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
        //printf("CLuster: %d\nCentroid calculado x: %f, y: %f\n",i,(*centroids_x)[i],(*centroids_y)[i]);
    }
}

int new_clusters(float** pontos_x,float** pontos_y,float** cluster_x,float** cluster_y,int** cluster_atribution,
                  float** centroids_x,float** centroids_y,int** cluster_size){
    int boolean = 1;
    float x,y,tmp,new_x,new_y,min_dist=2;
    for(int i = 0; i < K; i++) {
        x = (*centroids_x)[i];
        y = (*centroids_y)[i];

        min_dist=2;
        for(int j = 0; j<N;j++){ 
            tmp = euclidean_distance(x,y,(*pontos_x)[j],(*pontos_y)[j]);
            //printf("Cluster: %d\nPonto x: %f , y:%f\nDist: atual->%f menor-> %f\n\n",i,(*pontos_x)[j],(*pontos_y)[j],tmp,min_dist);
            if(tmp < min_dist){
                min_dist = tmp;
                new_x = (*pontos_x)[j];
                new_y = (*pontos_y)[j];
                //printf("Dist menor atualizado %f\n\n",min_dist);
            }

        }
        if(!((*cluster_x)[i] == new_x && (*cluster_y)[i] == new_y)) boolean *=  0;
        //printf("Cluster %d\nAntigo (%f,%f)\nNovo:(%f,%f)\n\n",i,(*cluster_x)[i],(*cluster_y)[i],new_x,new_y);
        (*cluster_x)[i] = new_x;
        (*cluster_y)[i] = new_y;
    }
    return boolean;
}
/*
int k_means(float** pontos_x,float** pontos_y,float** cluster_x,float** cluster_y,int** cluster_atribution,
            float** centroids_x,float** centroids_y,int** cluster_size){

    int end = 0;
    while(end == 0){
        calculate_centroid(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
        end = new_clusters(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
    }
}
*/
int main(){
    float* pontos_x;
    float* pontos_y;
    float* cluster_x;
    float* cluster_y;
    int* cluster_atribution;
    int* cluster_size;
    float* centroids_x;
    float* centroids_y;
    inicializa(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
    for(int i = 0; i < N; i++) {
        //printf("Ponto %f %f\n",pontos_x[i],pontos_y[i]); 
    }  
/*
    printf("Iteração nº 1\n");

    for(int i = 0; i < K; i++) {
        printf("Cluster %d :%f , %f \nSize: %d\n\n",i,cluster_x[i],cluster_y[i],cluster_size[i]); 
    }    
    calculate_centroid(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
    for(int i = 0; i < K; i++) {
        printf("Cluster %d :%f , %f \nCentroide: %f %f\nSize: %d\n\n",i,cluster_x[i],cluster_y[i],centroids_x[i],centroids_y[i],cluster_size[i]); 
    }


    new_clusters(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
    cluster_distrib(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
    
    for(int i = 0; i < K; i++) {
        printf("New Cluster %d :%f , %f \nCentroide: %f %f\nSize: %d\n\n",i,cluster_x[i],cluster_y[i],centroids_x[i],centroids_y[i],cluster_size[i]); 
    }


    printf("Iteração nº 2\n");
    
    for(int i = 0; i < K; i++) {
        printf("Cluster %d :%f , %f \nSize: %d\n\n",i,cluster_x[i],cluster_y[i],cluster_size[i]); 
    }    
    calculate_centroid(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
    for(int i = 0; i < K; i++) {
        printf("Cluster %d :%f , %f \nCentroide: %f %f\nSize: %d\n\n",i,cluster_x[i],cluster_y[i],centroids_x[i],centroids_y[i],cluster_size[i]); 
    }


    new_clusters(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
    cluster_distrib(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
    
    for(int i = 0; i < K; i++) {
        printf("New Cluster %d :%f , %f \nCentroide: %f %f\nSize: %d\n\n",i,cluster_x[i],cluster_y[i],centroids_x[i],centroids_y[i],cluster_size[i]); 
    }*/

    
    int end = 0;
    int iteracoes = 0;
    while(end == 0){
        printf("Iteração nº %d\n",iteracoes);
        calculate_centroid(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
        end = new_clusters(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
        cluster_distrib(&pontos_x,&pontos_y,&cluster_x,&cluster_y,&cluster_atribution,&centroids_x,&centroids_y,&cluster_size);
        for(int i = 0; i < K; i++) {
            //printf("New Cluster %d :%f , %f \nCentroide: %f %f\nSize: %d\n\n",i,cluster_x[i],cluster_y[i],centroids_x[i],centroids_y[i],cluster_size[i]); 
        }
        iteracoes++;
    }
    for(int i = 0; i < K; i++) {
        printf("Cluster %d :%f , %f \nCentroide: %f %f\nSize: %d\n\n",i,cluster_x[i],cluster_y[i],centroids_x[i],centroids_y[i],cluster_size[i]); 
    }





    return 0;
}