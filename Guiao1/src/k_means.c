#include "../include/k_means.h"

#define N 10000000
#define K 4



void cluster_distrib(float** pontos,float** old_centroid,int** cluster_atribution,float** centroids,int** cluster_size) {

    int cluster_atual,k=0,clust;
    float min_dist,tmp;
    float x1,x2,y1,y2;

    for(int i = 0; i < K; i++){
        (*cluster_size)[i] = 0.0;
    }

    for(int i = 0; i < N*2; i+=2) {
        cluster_atual=0;
        min_dist = 2;
        clust=0;
        for(int j = 0; j < K*2 ; j+=2){
            x1 = (*centroids)[j];
            y1 = (*centroids)[j+1];
            x2 = (*pontos)[i];
            y2 = (*pontos)[i+1];
            tmp = euclidean_distance(x1,y1,x2,y2);

            if(tmp < min_dist){
                min_dist = tmp;
                cluster_atual = clust;
            }
            clust++;
        }
        (*cluster_atribution)[k] = cluster_atual;
        (*cluster_size)[cluster_atual]++;
        k++;
    }
}


//Podemos alterar o pontos e pontos_y para serem apenas um array de tamanho N*2 e acrescentar a o cluster_atibution
// Seria algo [x,y,att,x1,x2,att2...xn,yn,attn] de N*3
void inicializa(float** pontos,float** old_centroid,int** cluster_atribution,float** centroids,int** cluster_size) {

    int k=0;


    *pontos = malloc(N*2*sizeof(float));
    *old_centroid = malloc(K*2*sizeof(float));
    *cluster_atribution = malloc(N*sizeof(int));
    *cluster_size= malloc(K*sizeof(int));
    *centroids = malloc(K*2*sizeof(float));


    srand(10);
    for(int i = 0; i < N*2; i+=2) {
        (*pontos)[i] = (float) rand() / RAND_MAX;
        (*pontos)[i+1] = (float) rand() / RAND_MAX;
    }
    for(int i = 0; i < K*2; i+=2) {
        (*centroids)[i] = (*pontos)[i];
        (*centroids)[i+1] = (*pontos)[i+1];
        //printf("Centroids iniciais: %f, %f\n",(*centroids)[i],(*centroids)[i+1]);

        (*cluster_size)[k] = 0;
        k++;

    }
    cluster_distrib(&(*pontos),&(*old_centroid),&(*cluster_atribution),&(*centroids),&(*cluster_size));
}

//Melhor percorrer um vez ou K vezes evitando acessos a memoria? Testar
int calculate_centroid(float** pontos,float** old_centroid,int** cluster_atribution,float** centroids,int** cluster_size){
    int end = 1;
    int k = 0;
    for(int i = 0; i < K*2; i+=2){
        (*old_centroid)[i] = (*centroids)[i];
        (*old_centroid)[i+1] = (*centroids)[i+1];

        (*centroids)[i] = 0.0;
        (*centroids)[i+1] = 0.0;

    }


    for(int i = 0; i < N*2; i+=2){
        int cluster = (*cluster_atribution)[k];
        (*centroids)[cluster*2] += (*pontos)[i];
        (*centroids)[cluster*2+1] += (*pontos)[i+1];
        k++;
    }
    k=0;
    for(int i = 0; i < K*2; i+=2){
        (*centroids)[i] = (*centroids)[i]/(*cluster_size)[k];
        (*centroids)[i+1] = (*centroids)[i+1]/(*cluster_size)[k];

        if(((*centroids)[i] != (*old_centroid)[i]) || ((*centroids)[i+1] !=(*old_centroid)[i+1])) end = 0;
        k++;
    }

    return end;
}

void k_means(float** pontos,float** old_centroid,int** cluster_atribution,float** centroids,int** cluster_size){
    int end = 0;
    int iteracoes = -1;
    while(end == 0){
        //printf("Iteração nº %d\n",iteracoes);
        end = calculate_centroid(&pontos,&old_centroid,&cluster_atribution,&centroids,&cluster_size);
        cluster_distrib(&pontos,&old_centroid,&cluster_atribution,&centroids,&cluster_size);
        iteracoes++;
    }
    printf("N = %d, K = %d\n",N,K);
    for(int i = 0; i < K; i++) {
        printf("Center: (%0.3f, %0.3f) : Size: %d\n",centroids[i],centroids[i],cluster_size[i]); 
    }
    printf("Iterations: %d\n",iteracoes);
}

int main(){
    float* pontos;            //Array de tamanho N*2 com x,y em sequencia de cada ponto
    float* old_centroid;      //Array de tamanho K   que guarda os antigos centroids calculados
    int* cluster_atribution;  //Array de tamanho N   index de cada ponto para o seu cluster
    int* cluster_size;        //Array de tamnho  K   tamanho de cada cluster
    float* centroids;         //Array de tamnho  N*2 com x,y em sequencia de cada centroid atual

    inicializa(&pontos,&old_centroid,&cluster_atribution,&centroids,&cluster_size);
    k_means(&(*pontos),&(*old_centroid),&(*cluster_atribution),&(*centroids),&(*cluster_size));
    
    return 0;
}