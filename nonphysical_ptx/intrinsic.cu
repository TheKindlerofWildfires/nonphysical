#include<cuda.h>
extern "C" __device__ float test(float x) {
    return sinh(x);
}
extern "C" __global__ void sin_approx_f32(float * x){
    
}

