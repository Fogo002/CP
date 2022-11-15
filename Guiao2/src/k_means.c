#include<omp.h>
#include "../include/k_means.h"

// #define N 10000000
// #define K 4

float* pontos;            //Array de tamanho N*2 com x,y em sequencia de cada ponto
int* cluster_atribution;  //Array de tamanho N   indexa cada ponto para o seu cluster
int* cluster_size;        //Array de tamnho  K   tamanho de cada cluster
float* centroids;         //Array de tamnho  K*2*2 com [x,y,old_x,old_y,....] em sequencia de cada centroid atual e do centroid antigo
                          //Sendo que o centroid i = 0 diz respeito ao cluster 0, i = 4*1 diz respeito ao cluster 1, etc 

int N;
int K;
int T;


//Distribui os pontos pelos diferentes clusters tendo em conta qual o centroids mais próximo
void cluster_distrib() {

    int cluster_atual,clust,k_size=K*4;
    float min_dist,tmp;
    float x1,x2,y1,y2;
    
    for(int i = 0; i < K; i++){
        cluster_size[i] = 0.0;
    }

    //A cada ponto verifica qual o centroid mais próximo e atualiza o cluster_size 
    //do cluster mais próximo e o cluster_distrib para conter o novo index desse ponto , com nivel 2 de unrolls

    // area de multithread  (3)
    
    #pragma omp parallel num_threads(T) private(cluster_atual,min_dist,clust,tmp,x1,x2,y1,y2)
    {
        # pragma omp for reduction(+:cluster_size[:K])
        for(int i = 0; i < N; i++) {
            cluster_atual=0;
            min_dist = 2;
            clust=0;
            x2 = pontos[i*2];
            y2 = pontos[i*2+1];

            for(int j = 0; j < k_size ; j+=4){
                x1 = centroids[j];
                y1 = centroids[j+1];

                tmp = euclidean_distance(x1,y1,x2,y2);

                if(tmp < min_dist){
                    min_dist = tmp;
                    cluster_atual = clust;
                }
                clust++;
            }

            cluster_atribution[i] = cluster_atual;
            
            cluster_size[cluster_atual]++;
        }
    }

    // area de multithread  (3)
}


//Inicializa os diferentes arrays
void inicializa() {

    int k=0;

    pontos = malloc(N*2*sizeof(float));
    cluster_atribution = malloc(N*sizeof(int));
    cluster_size= malloc(K*sizeof(int));
    centroids = malloc(K*4*sizeof(float));

    srand(10);

    // area de multithread  (1)

    //#pragma omp parallel num_threads(T)
    {
        //#pragma omp for
        for(int i = 0; i < N*2; i+=2) {
            pontos[i] = (float) rand() / RAND_MAX;
            pontos[i+1] = (float) rand() / RAND_MAX;
        }
    }

    // area de multithread  (1)
    
    for(int i = 0; i < K*2*2; i+=2) {
        centroids[i+i] = pontos[i];     //Centroid ponto x
        centroids[i+i+1] = pontos[i+1]; //Centroid ponto y
        cluster_size[k] = 0;
        k++;

    }
    cluster_distrib();
}

//Calcula os novos valores de centroid de cada cluster
int calculate_centroid(){
    int end = 1;
    int k = -1;
    //Guarda os valores antigos dos centroids e inicializa os novos a 0
    //io = [x_centroid,y_centroid,x_centroid_old,y_centroid_old,...]
    //i1 = [0.0 , 0.0 , x_centroid,y_centroid,.... ]
    for(int i = 0; i < K*4; i+=4){
        centroids[i+2] = centroids[i];
        centroids[i+3] = centroids[i+1];

        centroids[i] = 0.0;
        centroids[i+1] = 0.0;

    }

    // area de multithread  (2)

    #pragma omp parallel num_threads(T)
    {
        #pragma omp for reduction(+:centroids[:K*4])
        for(int i = 0; i < N; i++){
            int cluster = cluster_atribution[i];
            centroids[cluster*4] += pontos[i*2];
            centroids[cluster*4+1] += pontos[i*2+1];
        }
    }
    
    // area de multithread  (2)

    k=0;
    
    //Divisão dos valores dos centroids pelo tamanho do seu respetivo cluster, para obter o novo centroid do cluster
    for(int i = 0; i < K*4; i+=4){
        if (cluster_size[k]>0){
            centroids[i] = centroids[i]/cluster_size[k];
            centroids[i+1] = centroids[i+1]/cluster_size[k];
        }
        
        //Verificação se os valores foram alterados para continuar o algoritmo
        if((centroids[i] != centroids[i+2]) || (centroids[i+1] !=centroids[i+3])) end = 0;
        k++;
    }

    return end;
}

//Executa o algoritmo K-means, tendo em conta que as variáveis estejam inicializadas
void k_means(){
    int end = 0;
    int iteracoes = -1;

    //Ciclo que para quando os centroids não variam
    while(end == 0 && iteracoes <20){
        end = calculate_centroid();
        cluster_distrib();
        iteracoes++;
    }
    printf("N = %d, K = %d\n",N,K);
    for(int i = 0; i < K; i++) {
        printf("Center: (%0.3f, %0.3f) : Size: %d\n",centroids[i*4],centroids[i*4+1],cluster_size[i]); 
    }
    printf("Iterations: %d\n",iteracoes);
}

int main(int argc, char **argv){
    N = atoi(argv[1]);
    K = atoi(argv[2]);
    if(argc>3){
        T = atoi(argv[3]);
    } else{
        T = 1;
    }
    inicializa();
    k_means();
    
    return 0;
}
