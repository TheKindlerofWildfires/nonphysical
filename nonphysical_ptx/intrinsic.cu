#include<cuda.h>

//__shared__ float block[4096];

__constant__ float abc[3];
extern "C" __device__ void test(float * x,float * y, int z) {
    x[0]=abc[0];
    x[1]=abc[1];
    x[2]=abc[2];

}
