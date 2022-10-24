#include "../include/k_means.h"

#define N 10000000
#define K 4




//Distribui os pontos pelos diferentes clusters tendo em conta qual o centroids mais próximo
void cluster_distrib(float** pontos,int** cluster_atribution,float** centroids,int** cluster_size) {

    int cluster_atual,k=0,clust,k_size=K*4,n_size = N*2;
    float min_dist,tmp;
    float x1,x2,y1,y2;
    
    for(int i = 0; i < K; i++){
        (*cluster_size)[i] = 0.0;
    }
    //A cada ponto verifica qual o centroid mais próximo e atualiza o cluster_size 
    //do cluster mais próximo e o cluster_distrib para conter o novo index desse ponto
    for(int i = 0; i < n_size; i+=2) {
        cluster_atual=0;
        min_dist = 2;
        clust=0;
        x2 = (*pontos)[i];
        y2 = (*pontos)[i+1];
        for(int j = 0; j < k_size ; j+=4){
            x1 = (*centroids)[j];
            y1 = (*centroids)[j+1];

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


//Inicializa os diferentes arrays
void inicializa(float** pontos,int** cluster_atribution,float** centroids,int** cluster_size) {

    int k=0;


    *pontos = malloc(N*2*sizeof(float));
    *cluster_atribution = malloc(N*sizeof(int));
    *cluster_size= malloc(K*sizeof(int));
    *centroids = malloc(K*4*sizeof(float));


    srand(10);
    for(int i = 0; i < N*2; i+=2) {
        (*pontos)[i] = (float) rand() / RAND_MAX;
        (*pontos)[i+1] = (float) rand() / RAND_MAX;
    }
    for(int i = 0; i < K*2*2; i+=2) {
        (*centroids)[i+i] = (*pontos)[i];     //Centroid ponto x
        (*centroids)[i+i+1] = (*pontos)[i+1]; //Centroid ponto y
        (*cluster_size)[k] = 0;
        k++;

    }
    cluster_distrib(&(*pontos),&(*cluster_atribution),&(*centroids),&(*cluster_size));
}

//Calcula os novos valores de centroid de cada cluster
int calculate_centroid(float** pontos,int** cluster_atribution,float** centroids,int** cluster_size){
    int end = 1;
    int k = 0;
    //Guarda os valores antigos dos centroids e inicializa os novos a 0
    //io = [x_centroid,y_centroid,x_centroid_old,y_centroid_old,...]
    //i1 = [0.0 , 0.0 , x_centroid,y_centroid,.... ]
    for(int i = 0; i < K*4; i+=4){
        (*centroids)[i+2] = (*centroids)[i];
        (*centroids)[i+3] = (*centroids)[i+1];

        (*centroids)[i] = 0.0;
        (*centroids)[i+1] = 0.0;

    }

    //No cluster B é feita o somatório dos pontos(x,y) no centroid que representa esse cluster, sendo B um cluster
    for(int i = 0; i < N*2; i+=2){
        int cluster = (*cluster_atribution)[k];
        (*centroids)[cluster*4] += (*pontos)[i];
        (*centroids)[cluster*4+1] += (*pontos)[i+1];
        k++;
    }
    k=0;
    
    //Divisão dos valores dos centroids pelo tamanho do seu respetivo cluster, para obter o novo centroid do cluster
    for(int i = 0; i < K*4; i+=4){
        (*centroids)[i] = (*centroids)[i]/(*cluster_size)[k];
        (*centroids)[i+1] = (*centroids)[i+1]/(*cluster_size)[k];
        
        //Verificação se os valores foram alterados para continuar o algoritmo
        if(((*centroids)[i] != (*centroids)[i+2]) || ((*centroids)[i+1] !=(*centroids)[i+3])) end = 0;
        k++;
    }

    return end;
}

//Executa o algoritmo K-means, tendo em conta que as variáveis estejam inicializadas
void k_means(float** pontos,int** cluster_atribution,float** centroids,int** cluster_size){
    int end = 0;
    int iteracoes = -1;

    //Ciclo que para quando os centroids não variam
    while(end == 0){
        //printf("Iteração nº %d\n",iteracoes);
        end = calculate_centroid(&(*pontos),&(*cluster_atribution),&(*centroids),&(*cluster_size));
        cluster_distrib(&(*pontos),&(*cluster_atribution),&(*centroids),&(*cluster_size));
        iteracoes++;
    }
    printf("N = %d, K = %d\n",N,K);
    for(int i = 0; i < K; i++) {
        printf("Center: (%0.3f, %0.3f) : Size: %d\n",(*centroids)[i],(*centroids)[i],(*cluster_size)[i]); 
    }
    printf("Iterations: %d\n",iteracoes);
}

int main(){
    float* pontos;            //Array de tamanho N*2 com x,y em sequencia de cada ponto
    int* cluster_atribution;  //Array de tamanho N   indexa cada ponto para o seu cluster
    int* cluster_size;        //Array de tamnho  K   tamanho de cada cluster
    float* centroids;         //Array de tamnho  K*2*2 com [x,y,old_x,old_y,....] em sequencia de cada centroid atual e do centroid antigo
                              //Sendo que o centroid i = 0 diz respeito ao cluster 0, i = 4*1 diz respeito ao cluster 1, etc 
    inicializa(&pontos,&cluster_atribution,&centroids,&cluster_size);
    k_means(&pontos,&cluster_atribution,&centroids,&cluster_size);
    
    return 0;
}