#include "stencil.h"

using namespace std;

#define NUM_BLOCKS 1024
#define NUM_THREADS_PER_BLOCK 1024
#define SIZE NUM_BLOCKS*NUM_THREADS_PER_BLOCK
#define N NUM_BLOCKS*NUM_THREADS_PER_BLOCK
#define K 4

float* pontos;            //Array de tamanho N*2 com x,y em sequencia de cada ponto
int* cluster_atribution;  //Array de tamanho N   indexa cada ponto para o seu cluster
int* cluster_size;        //Array de tamnho  K   tamanho de cada cluster
float* centroids;         //Array de tamnho  K*2*2 com [x,y,old_x,old_y,....] em sequencia de cada centroid atual e do centroid antigo
                          //Sendo que o centroid i = 0 diz respeito ao cluster 0, i = 4*1 diz respeito ao cluster 1, etc 
float* d_pontos;            
int* d_cluster_atribution; 
int* d_cluster_size;       
float* d_centroids; 
int* d_end;
int* d_iteracoes;

//Distribui os pontos pelos diferentes clusters tendo em conta qual o centroids mais próximo
__global__  void cluster_distrib(int*d_cluster_size,float*d_centroids,float*d_pontos,int* d_cluster_atribution) {        
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    int cluster_atual,clust,k_size=K*4;
    float min_dist,tmp;
    float x1,x2,y1,y2;
    __shared__ float block_centroids[K*4];
    if(lid<k_size){
    // carrega os pontos e atribuicoes para memoria partilhada que serão somados a seguir
        block_centroids[lid]= d_centroids[lid];
        block_centroids[lid]= d_centroids[lid];
    }
    // sincronizar bloco para garantir que todos os elementos a usar pelo bloco foram carregados
	__syncthreads();

    cluster_atual=0;
    min_dist = 2;
    clust=0;
    x2 = d_pontos[id*2];
    y2 = d_pontos[id*2+1];

    for(int j = 0; j < k_size ; j+=4){
        x1 = block_centroids[j];
        y1 = block_centroids[j+1];
                
        tmp = (x2 - x1)*(x2 - x1)+ (y2 - y1)*(y2 - y1);

        if(tmp < min_dist){
            min_dist = tmp;
            cluster_atual = clust;
        }
        clust++;
    }
    d_cluster_atribution[id] = cluster_atual;
}
// Parte 1 : faz as somas todas
__global__  void calculate_centroid(int*d_cluster_size,float*d_centroids,float*d_pontos,int* d_cluster_atribution){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    //if (id >= N) return;
    // init centroids e tamanhos dos clusters
    if(id<K){
        d_centroids[id*4+2] = d_centroids[id*4];
        d_centroids[id*4+3] = d_centroids[id*4+1];
        d_centroids[id*4] = 0.0;
        d_centroids[id*4+1] = 0.0;
        d_cluster_size[id] = 0;
    }
    // carrega os pontos e atribuicoes para memoria partilhada que serão somados a seguir
	__shared__ float block_pontos_x[NUM_THREADS_PER_BLOCK];
	block_pontos_x[lid]= d_pontos[id*2];
    __shared__ float block_pontos_y[NUM_THREADS_PER_BLOCK];
	block_pontos_y[lid]= d_pontos[id*2+1];
	__shared__ int block_cluster_atribution[NUM_THREADS_PER_BLOCK];
	block_cluster_atribution[lid] = d_cluster_atribution[id];

    // sincronizar bloco para garantir que todos os elementos a usar pelo bloco foram carregados
	__syncthreads();

	// cada thread com id = 0 do seu bloco faz primeiro a soma local e depois e depois soma isso no global (reduction-like)
	if(lid==0){
		float block_sum_x[K]={0};
        float block_sum_y[K]={0};
		int block_clust_size[K]={0};
        // soma local (bloco)
		for(int j=0; j< blockDim.x; ++j){
			int cluster_id = block_cluster_atribution[j];
			block_sum_x[cluster_id]+=block_pontos_x[j];
            block_sum_y[cluster_id] += block_pontos_y[j];
			block_clust_size[cluster_id]+=1;
		}
		// soma global
		for(int z=0; z < K; ++z){
			atomicAdd(&d_centroids[z*4],block_sum_x[z]);
            atomicAdd(&d_centroids[z*4+1],block_sum_y[z]);
			atomicAdd(&d_cluster_size[z],block_clust_size[z]);
		}
	}
}

// Parte 2 : calcula centroids atuais e verifica condição de saida (separado em 2 funcões para haver sincronização total dos calculos anteriores)
__global__  void calculate_centroid_2(int*d_end,int*d_cluster_size,float*d_centroids){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //int lid = threadIdx.x;
    if(id<K){
        if(d_cluster_size[id]>0){
            d_centroids[id*4] = d_centroids[id*4]/d_cluster_size[id];
            d_centroids[id*4+1] = d_centroids[id*4+1]/d_cluster_size[id];
        }
        d_end[0] = 1;

        __syncthreads();

        if((d_centroids[id*4] != d_centroids[id*4+2]) || (d_centroids[id*4+1] != d_centroids[id*4+3])) d_end[0] = 0;
    }
}

__global__ void k_means_gpu(int*d_end,int*d_cluster_size,float*d_centroids,float*d_pontos,int* d_cluster_atribution,int*d_iteracoes){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id==0){
        int iteracoes = -1;
        while(d_end[0] == 0 && iteracoes <20){
            calculate_centroid<<< NUM_THREADS_PER_BLOCK,NUM_BLOCKS >>> (d_cluster_size,d_centroids,d_pontos,d_cluster_atribution);
            calculate_centroid_2<<< NUM_THREADS_PER_BLOCK,NUM_BLOCKS >>> (d_end,d_cluster_size,d_centroids);
            cluster_distrib<<< NUM_THREADS_PER_BLOCK,NUM_BLOCKS >>> (d_cluster_size,d_centroids,d_pontos,d_cluster_atribution);
            iteracoes++;
        }
        d_iteracoes[0]=iteracoes;
    }
}

//Inicializa os diferentes arrays
void inicializa() {
    int k=0;

    pontos = (float*) malloc(N*2*sizeof(float));
    cluster_atribution = (int*) malloc(N*sizeof(int));
    cluster_size= (int*) malloc(K*sizeof(int));
    centroids = (float*) malloc(K*4*sizeof(float));

    srand(10);
   
    for(int i = 0; i < N*2; i+=2) {
        pontos[i] = (float) rand() / RAND_MAX;
        pontos[i+1] = (float) rand() / RAND_MAX;
    }
    for(int i = 0; i < K*4; i+=4) {
        centroids[i+i] = pontos[i];     //Centroid ponto x
        centroids[i+i+1] = pontos[i+1]; //Centroid ponto y
        cluster_size[k] = 0;
        k++;
    }
    cudaMalloc ((void**) &d_pontos,  N*2*(sizeof(float)) );
    cudaMalloc ((void**) &d_centroids, K*4*sizeof(float));
    cudaMalloc ((void**) &d_cluster_atribution, N*sizeof(int));
    cudaMalloc ((void**) &d_cluster_size, K*sizeof(int));
    cudaMalloc ((void**) &d_end, sizeof(int));
    cudaMalloc ((void**) &d_iteracoes, sizeof(int));

    cudaMemcpy(d_centroids, centroids, K*4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pontos, pontos, N*2*sizeof(float), cudaMemcpyHostToDevice);

    cluster_distrib<<< NUM_THREADS_PER_BLOCK,NUM_BLOCKS >>> (d_cluster_size,d_centroids,d_pontos,d_cluster_atribution);
}

//Executa o algoritmo K-means, tendo em conta que as variáveis estejam inicializadas
void k_means(){
    int iteracoes = -1;

    k_means_gpu<<< NUM_THREADS_PER_BLOCK,NUM_BLOCKS >>> (d_end,d_cluster_size,d_centroids,d_pontos,d_cluster_atribution,d_iteracoes);
   
    cudaMemcpy(centroids, d_centroids, K*4*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cluster_size, d_cluster_size, K*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&iteracoes, d_iteracoes,sizeof(int), cudaMemcpyDeviceToHost);

    printf("N = %d, K = %d\n",N,K);
    for(int i = 0; i < K; i++) {
        printf("Center: (%0.3f, %0.3f) : Size: %d\n",centroids[i*4],centroids[i*4+1],cluster_size[i]); 
    }
    printf("Iterations: %d\n",iteracoes);
}

int main(int argc, char **argv){
    startKernelTime();
    inicializa();
    k_means();
    stopKernelTime();
    return 0;
}
