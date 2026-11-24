#include <stdio.h>

__global__ void hello() {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block %d, Thread %d, Global ID %d\n", blockIdx.x, threadIdx.x, id);
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    printf("GPU Name             : %s\n", p.name);
    printf("Compute Capability   : %d.%d\n", p.major, p.minor);
    printf("Multiprocessors (SM) : %d\n", p.multiProcessorCount);
    printf("Warp Size            : %d\n", p.warpSize);
    printf("Max Threads/Block    : %d\n", p.maxThreadsPerBlock);

    printf("\n<<<1 block, 4 threads>>>\n");
    hello<<<1, 4>>>();
    cudaDeviceSynchronize();

    printf("\n<<<2 blocks, 4 threads>>>\n");
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();

    printf("\n<<<4 blocks, 2 threads>>>\n");
    hello<<<4, 2>>>();
    cudaDeviceSynchronize();
}
