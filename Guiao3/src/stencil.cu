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
    //int lid = threadIdx.x;

    //__shared__ float temp[NUM_THREADS_PER_BLOCK*2];
    //temp[lid*2] = pontos[id];
    //temp[lid*2+1] = pontos[id+1];

    int cluster_atual,clust,k_size=K*4;
    float min_dist,tmp;
    float x1,x2,y1,y2;

    cluster_atual=0;
    min_dist = 2;
    clust=0;
    x2 = d_pontos[id*2];
    y2 = d_pontos[id*2+1];

    for(int j = 0; j < k_size ; j+=4){
        x1 = d_centroids[j];
        y1 = d_centroids[j+1];
                
        tmp = (x2 - x1)*(x2 - x1)+ (y2 - y1)*(y2 - y1);

        if(tmp < min_dist){
            min_dist = tmp;
            cluster_atual = clust;
        }
        clust++;
    }
    d_cluster_atribution[id] = cluster_atual;
}

__global__  void calculate_centroid(int*d_cluster_size,float*d_centroids,float*d_pontos,int* d_cluster_atribution){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    //if (id >= N) return;

    if(id<K){
        d_centroids[id*4+2] = d_centroids[id*4];
        d_centroids[id*4+3] = d_centroids[id*4+1];
        d_centroids[id*4] = 0.0;
        d_centroids[id*4+1] = 0.0;
        d_cluster_size[id] = 0;
    }

    //put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
	__shared__ float block_pontos_x[NUM_THREADS_PER_BLOCK];
	block_pontos_x[lid]= d_pontos[id*2];
    __shared__ float block_pontos_y[NUM_THREADS_PER_BLOCK];
	block_pontos_y[lid]= d_pontos[id*2+1];
	__shared__ int block_cluster_atribution[NUM_THREADS_PER_BLOCK];
	block_cluster_atribution[lid] = d_cluster_atribution[id];

	__syncthreads();

	//it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
	if(lid==0){
		float block_sum_x[K]={0};
        float block_sum_y[K]={0};
		int block_clust_size[K]={0};

		for(int j=0; j< blockDim.x; ++j){
			int cluster_id = block_cluster_atribution[j];
			block_sum_x[cluster_id]+=block_pontos_x[j];
            block_sum_y[cluster_id] += block_pontos_y[j];
			block_clust_size[cluster_id]+=1;
		}
		//Now we add the sums to the global centroids and add the counts to the global counts.
		for(int z=0; z < K; ++z){
			atomicAdd(&d_centroids[z*4],block_sum_x[z]);
            atomicAdd(&d_centroids[z*4+1],block_sum_y[z]);
			atomicAdd(&d_cluster_size[z],block_clust_size[z]);
		}
	}
}

__global__  void calculate_centroid_2(int*d_end,int*d_cluster_size,float*d_centroids){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //int lid = threadIdx.x;

    if(id<K){
        d_centroids[id*4] = d_centroids[id*4]/d_cluster_size[id];
        d_centroids[id*4+1] = d_centroids[id*4+1]/d_cluster_size[id];

        d_end[0] = 1;

        __syncthreads();

        if((d_centroids[id*4] != d_centroids[id*4+2]) || (d_centroids[id*4+1] != d_centroids[id*4+3])) d_end[0] = 0;
    }
}

__global__ void k_means_gpu(int*d_end,int*d_cluster_size,float*d_centroids,float*d_pontos,int* d_cluster_atribution,int*d_iteracoes){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int iteracoes = -1;
    if(id==0){
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
    for(int i = 0; i < K*2*2; i+=2) {
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
    //Ciclo que para quando os centroids não variam

    k_means_gpu<<< NUM_THREADS_PER_BLOCK,NUM_BLOCKS >>> (d_end,d_cluster_size,d_centroids,d_pontos,d_cluster_atribution,d_iteracoes);
   
    
    cudaMemcpy(centroids, d_centroids, K*4*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cluster_size, d_cluster_size, K*sizeof(int), cudaMemcpyDeviceToHost);
    printf("N = %d, K = %d\n",N,K);
    for(int i = 0; i < K; i++) {
        printf("Center: (%0.3f, %0.3f) : Size: %d\n",centroids[i*4],centroids[i*4+1],cluster_size[i]); 
    }
    cudaMemcpy(&iteracoes, d_iteracoes,sizeof(int), cudaMemcpyDeviceToHost);
    printf("Iterations: %d\n",iteracoes);
}

int main(int argc, char **argv){
    inicializa();
    k_means();
    return 0;
}
