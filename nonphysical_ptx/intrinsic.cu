#include<cuda.h>

//

__shared__ int block[4096];

extern "C" __global__ void test(unsigned int * x,float * y, int z) {
    block[threadIdx.x]=x[0];
    atomicCAS(&block[threadIdx.x],block[threadIdx.y],block[threadIdx.z]);
    x[1]=block[threadIdx.x];


}