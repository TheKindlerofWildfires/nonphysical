#include<cuda.h>

//__shared__ float block[4096];
extern "C" __device__ void test(float * x,float * y, int z) {

}
extern "C" __global__ void gabor_kernel(float * x){
    __shared__ float local[8192];
    if (threadIdx.x == 0){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i = 0; i<8192;i++){
            local[i]=x[i];
        }
        for (int i = 0; i<8192;i++){
            local[i]+=1.0;
        }
        for (int i = 0; i<8192;i++){
            x[i]=local[i];
        }
    }

}

