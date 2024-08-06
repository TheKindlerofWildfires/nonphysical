#include <chrono> 
#include <time.h>

using namespace std::chrono;

using namespace std;

__device__ float M_PI = 3.1415;

typedef float2 Complex_my;

static __device__ __host__ inline Complex_my Add(Complex_my A, Complex_my B)
{
    Complex_my C;
    C.x = A.x + B.x;
    C.y = A.y + B.y;
    return C;
}

/**
 *  Inverse of Complex_my Number
 */
static __device__ __host__ inline Complex_my Inverse(Complex_my A)
{
    Complex_my C;
    C.x = -A.x;
    C.y = -A.y;
    return C;
}

/**
 *  Multipication of Complex_my Numbers
 */
static __device__ __host__ inline Complex_my Multiply(Complex_my A, Complex_my B)
{
    Complex_my C;
    C.x = A.x * B.x - A.y * B.y;
    C.y = A.y * B.x + A.x * B.y;
    return C;
}

/**
* Parallel Functions for performing various tasks
*/

/**
*  Dividing by constant for inverse fft transform
*/
__global__ void inplace_divide_invert(Complex_my *A, int n, int threads)
{
    int i = blockIdx.x * threads + threadIdx.x;
    if (i < n)
    {
        // printf("in divide");
        A[i].x /= n;
        A[i].y /= n;
    }
    else
    {
        // printf("else in divide");
        // printf("i=%d, n=%d", i, n);
    }
}

/**
* Reorders array by bit-reversing the indexes.
*/
__global__ void bitrev_reorder(Complex_my *__restrict__ r, Complex_my *__restrict__ d, int s, size_t nthr, int n)
{
    int id = blockIdx.x * nthr + threadIdx.x;
    //r[id].x = -1;
    if ((id < n) && (__brev(id) >> (32 - s) < n))
        r[__brev(id) >> (32 - s)] = d[id];
}

/**
* Inner part of the for loop
*/
__device__ void inplace_fft_inner(Complex_my *__restrict__ A, int i, int j, int len, int n, bool invert)
{
    if ((i + j + len / 2 < n) && (j < len / 2))
    {
        Complex_my u, v;

        float angle = (2 * M_PI * j) / (len * (invert ? 1.0 : -1.0));
        v.x = cos(angle);
        v.y = sin(angle);

        u = A[i + j];
        v = Multiply(A[i + j + len / 2], v);
        // printf("i:%d j:%d u_x:%f u_y:%f    v_x:%f v_y:%f\n", i, j, u.x, u.y, v.x, v.y);
        A[i + j] = Add(u, v);
        A[i + j + len / 2] = Add(u, Inverse(v));
    }
}

/**
* FFT if number of threads are sufficient.
*/
__global__ void inplace_fft(Complex_my *__restrict__ A, int i, int len, int n, int threads, bool invert)
{
    int j = blockIdx.x * threads + threadIdx.x;
    inplace_fft_inner(A, i, j, len, n, invert);
}

/**
* FFt if number of threads are not sufficient.
*/
__global__ void inplace_fft_outer(Complex_my *__restrict__ A, int len, int n, int threads, bool invert)
{
    int i = (blockIdx.x * threads + threadIdx.x)*len;
    for (int j = 0; j < len / 2; j++)
    {
        inplace_fft_inner(A, i, j, len, n, invert);
    }
}

/**
* parallel FFT transform and inverse transform
* Arguments vector of complex numbers, invert, balance, number of threads
* Perform inplace transform
*/
void fft(bool invert, int balance = 10, int threads = 32)
{
    // Creating array from vector
    int n = 4096;
    int data_size = n * sizeof(Complex_my);
    Complex_my *data_array = (Complex_my *)malloc(data_size);
    for (int i = 0; i < n; i++)
    {
        data_array[i].x = i;
        data_array[i].y = i+1;
    }
    
    // Copying data to GPU
    Complex_my *A, *dn;
    cudaMalloc((void **)&A, data_size);
    cudaMalloc((void **)&dn, data_size);
    cudaMemcpy(dn, data_array, data_size, cudaMemcpyHostToDevice);
    // Bit reversal reordering
    int s = log2(n);

    bitrev_reorder<<<ceil(float(n) / threads), threads>>>(A, dn, s, threads, n);

    
    // Synchronize
    cudaDeviceSynchronize();
    // Iterative FFT with loop parallelism balancing
    int count = 0;
    for (int len = 2; len <= n; len <<= 1)
    {
        if (n / len > balance)
        {

            inplace_fft_outer<<<ceil((float)n / threads / len), threads>>>(A, len, n, threads, invert);
        }
        else
        {
            for (int i = 0; i < n; i += len)
            {
                float repeats = len / 2;
                inplace_fft<<<ceil(repeats / threads), threads>>>(A, i, len, n, threads, invert);
                count +=1;
            }
        }
    }
    if (invert)
        inplace_divide_invert<<<ceil(n * 1.00 / threads), threads>>>(A, n, threads);

    // Copy data from GPU
    Complex_my *result;
    result = (Complex_my *)malloc(data_size);
    cudaMemcpy(result, A, data_size, cudaMemcpyDeviceToHost);
    
    // Saving data to vector<complex> in input.
    // Free the memory blocks
    free(data_array);
    cudaFree(A);
    cudaFree(dn);
    return;
}


#define BALANCE 1024

int main()
{
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    fft(false,BALANCE,32);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print the result
    printf("The function took %f seconds to execute.\n", cpu_time_used);
    return 0;
}