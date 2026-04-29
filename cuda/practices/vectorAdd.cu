/*
The program intends to practice the "Hello World!" in GPU programming, vector addition
given the input vector, x, y. return the vector addition's output z
where z = x + y
*/

/*
TODOs:
- compare the time running on CPU and GPU
*/
#include <stdio.h>
#include <stdlib.h>

__global__ void vecAdd_kernel(float *x_d, float *y_d, float *z_d, unsigned int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N)
    {
        z_d[i] = x_d[i] + y_d[i];
    }
}

void vecAdd_gpu(float *x_h, float *y_h, float *z_h, unsigned int N)
{
    // allocate x, y, z memory on cuda device
    float *x_d, *y_d, *z_d;
    size_t mem_size = N * sizeof(float);
    cudaMalloc((void **)&x_d, mem_size);
    cudaMalloc((void **)&y_d, mem_size);
    cudaMalloc((void **)&z_d, mem_size);
    printf("Successfully allocate the device memory\n");
    // copy x and y from host to device
    cudaMemcpy(x_d, x_h, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, mem_size, cudaMemcpyHostToDevice);
    printf("Copy the data from Host to Device\n");
    // launch the vecAdd kernel to do the addition operation
    int BLOCKSIZE = 1024;
    int BLOCKNUM = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    printf("Running the kernel ...\n");
    vecAdd_kernel<<<BLOCKNUM, BLOCKSIZE>>>(x_d, y_d, z_d, N);
    // copy z from device back to host
    cudaMemcpy(z_h, z_d, mem_size, cudaMemcpyDeviceToHost);
    // freeup x, y, z memory on cuda device
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main(void)
{
    // allocate the host memory for vector x, y, z
    unsigned int len_vec = 3000000;
    size_t mem_size = len_vec * sizeof(float);
    printf("the vector length is: %d and the memory size: %zu\n", len_vec, mem_size);
    float *x_h = (float *)malloc(mem_size);
    float *y_h = (float *)malloc(mem_size);
    float *z_h = (float *)malloc(mem_size);
    printf("Successfully allocate host x, y, z\n");
    // initiate x and y
    for (int i = 0 ; i < len_vec; ++i)
    {
        x_h[i] = rand() / (float)RAND_MAX;
        y_h[i] = rand() / (float)RAND_MAX;
    }
    printf("Randomly assigned each values of x and y vector\n");
    // run the launch function 
    printf("Running the launch function ...\n");
    vecAdd_gpu(x_h, y_h, z_h, len_vec);

    // freeup memory
    free(x_h);
    free(y_h);
    free(z_h);
    printf("Done.");
    return 0;

}