#include<cuda.h>

//__shared__ float block[4096];

/*
__shared__ double abc[30];
extern "C" __device__ void test(float * x,float * y, int z) {
    for (int i = 0; i<30;i++){
        abc[threadIdx.x]=y[i];
    }
    for (int i = 0; i<30;i++){
        x[i]=abc[i];
    }

}*/
extern "C" __device__ void test(unsigned int * x,float * y, int z) {
    /*
    __shared__ int max[3];
    __shared__ int min[3];
    for (int i = 0; i<3; i++){
        max[i]=0;
        min[i]=0;
    }
    for (int i = 0; i<3; i++){
        atomicMax(&max[i],x[i]);
        atomicMin(&min[i],x[i]);
    }
    y[0]=max[0];
    y[1]=min[0];*/
    y[3]=__clzll(z);

}