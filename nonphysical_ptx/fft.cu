
#include <cuda.h>
#include <stdio.h>
const int NFFT = 4096;

__shared__ float sub_data[NFFT*2];
__constant__ float twiddles[NFFT];
__device__ float reverse(int n){
    int v = n;
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
    v = (v >> 16) | (v << 16);
    v = v >> min(31, __clz(v));
    v = (v - 1) / 2;
    return sub_data[v];
}
__device__ void fft_n(int step, int t, int ridx){
    int dist = 1<<t;
    int chunk_size = dist<<1;
    int sub_idx = ridx>>t;
    int inner_idx = ridx%dist;

    float ics00 = sub_data[sub_idx*chunk_size*2+inner_idx*2];
    float ics01 = sub_data[sub_idx*chunk_size*2+inner_idx*2+1];

    float ics10 = sub_data[sub_idx*chunk_size*2+dist*2+inner_idx*2];
    float ics11 = sub_data[sub_idx*chunk_size*2+dist*2+inner_idx*2+1];

    float w0 = twiddles[inner_idx*step*2];
    float w1 = twiddles[inner_idx*step*2+1];


    float tmp0 = ics00-ics10;
    float tmp1 = ics01-ics11;

    sub_data[sub_idx*chunk_size*2+inner_idx*2] += ics10;
    sub_data[sub_idx*chunk_size*2+inner_idx*2+1] += ics11;

    sub_data[sub_idx*chunk_size*2+dist*2+inner_idx*2] = tmp0*w0-tmp1*w1;
    sub_data[sub_idx*chunk_size*2+dist*2+inner_idx*2+1] = tmp0*w1+tmp1*w0;

    
}
__device__ void fft_4(int ridx){
    int sub_idx = ridx>>1;
    if (ridx & 1 == 0){
        float tmp1 = sub_data[sub_idx*8];
        float tmp2 = sub_data[sub_idx*8+1];
        sub_data[sub_idx*8]+= sub_data[sub_idx*8+4];
        sub_data[sub_idx*8+1]+= sub_data[sub_idx*8+5];
        sub_data[sub_idx*8+4] = tmp1-sub_data[sub_idx*8+4];
        sub_data[sub_idx*8+5] = tmp2-sub_data[sub_idx*8+5];
    }else{
        float tmp1 = sub_data[sub_idx*8+2];
        float tmp2 = sub_data[sub_idx*8+3];
        sub_data[sub_idx*8+2]+= sub_data[sub_idx*8+6];
        sub_data[sub_idx*8+3]+= sub_data[sub_idx*8+7];
        sub_data[sub_idx*8+6] = tmp1-sub_data[sub_idx*8+7];
        sub_data[sub_idx*8+7] = -(tmp2-sub_data[sub_idx*8+6]);
    }
}

__device__ void fft_2(int ridx){
    float tmp1 = sub_data[ridx*4];
    float tmp2 = sub_data[ridx*4+1];
    sub_data[ridx*4]+= sub_data[ridx*4+2];
    sub_data[ridx*4+1]+= sub_data[ridx*4+3];
    sub_data[ridx*4+2] = tmp1-sub_data[ridx*4+2];
    sub_data[ridx*4+3] = tmp2-sub_data[ridx*4+3];
}
extern "C" __global__ void fft(float * data) {
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int block_dim = blockDim.x;
    //grap the start address
    sub_data[thread_idx*4] = data[block_idx+thread_idx*4];
    sub_data[thread_idx*4+1] = data[block_idx+thread_idx*4+1];
    sub_data[thread_idx*4+2] = data[block_idx+thread_idx*4+2];
    sub_data[thread_idx*4+3] = data[block_idx+thread_idx*4+3];
    __syncthreads();
    
    int step = 1;
    int n = (int)log2((float)NFFT);

    for(int t = n; t>2;t--){
        for (int ridx = 0; ridx<NFFT/2; ridx+=block_dim){
            //fft n the x
        }
        step <<=1;
        __syncthreads();
    }
    
    for (int ridx = 0; ridx<NFFT/2; ridx+=block_dim){
        //fft 4 the x
        fft_4(ridx);

    }
    __syncthreads();

    for (int ridx = 0; ridx<NFFT/2; ridx+=block_dim){
        //fft 2 the x
        fft_2(ridx);
    }
    __syncthreads();
    ;
    float local[16] = {0.0};
    int j = 0;
    for (int i = 0; i<NFFT;i+=block_dim){
        int n = i+NFFT;
        local[j] = reverse(n);
        local[j+1] = reverse(n+1);
        j+=2;
    }
    
    __syncthreads();
    int k = 0;
    //local[0]=local[1];
    //sub_data[0]= local[0];
    k+=2;
    
    for (int i = 0; i<NFFT;i+=block_dim){
        sub_data[2*i] = local[k];
        sub_data[2*i+1]= local[k+1];
        k+=2;
    }
    
    
    
    
    __syncthreads();
    data[block_idx+thread_idx*4]=sub_data[thread_idx*4];
    data[block_idx+thread_idx*4+1]=sub_data[thread_idx*4+1] ;
    data[block_idx+thread_idx*4+2]=sub_data[thread_idx*4+2] ;
    data[block_idx+thread_idx*4+3]=sub_data[thread_idx*4+3];
    return;
}


void generate_twiddles(float * twiddles){
    float angle = -3.1415/NFFT;
    for(int i = 0; i<NFFT/2; i++){
        float phase = angle*i;
        twiddles[i*2]=cos(phase);
        twiddles[i*2+1]=sin(phase);
    }
}

int main(){
    int N = NFFT*2048;
    //fake out all of the data 
    float *data, *d_data;
    data = (float*)malloc(N*2*sizeof(float));
    printf("Malloc complete\n");
    for (int i = 0; i< N; i++){
        data[i*2]=i;
        data[i*2+1]=i+1;
    }

    //malloc it on device
    cudaMalloc(&d_data, N*2*sizeof(float));
    //start the transfer
    cudaMemcpyAsync(d_data, data, N*2*sizeof(float), cudaMemcpyHostToDevice);
    printf("Copy complete\n");

    //generate the twiddles
    float * twiddles;
    twiddles = (float*)malloc(NFFT*sizeof(float));
    generate_twiddles(twiddles);
    printf("Twiddles complete\n");
    //copy them them to constant
    cudaMemcpyToSymbol("twiddles",twiddles,NFFT*sizeof(float),0,cudaMemcpyHostToDevice);
    printf("Const complete\n");

    //launch the fft
    dim3 xthread = dim3{1024,0,0};
    dim3 xblock = dim3(2048,0,0);
    fft<<<2048,1024>>>(d_data);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    //sync
    printf("launch complete\n");

    //collect complete
    cudaMemcpy(data,d_data,N*2*sizeof(float), cudaMemcpyDeviceToHost);
    printf("data complete\n");

}


