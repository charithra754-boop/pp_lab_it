#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 10000000

__global__ void add(float *a, float *b, float *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

int main() {
    int sz = N * sizeof(float);
    float *a = (float*)malloc(sz), *b = (float*)malloc(sz), *c = (float*)malloc(sz), *da, *db, *dc;

    for (int i = 0; i < N; i++) { a[i] = i; b[i] = i * 2; }

    clock_t start = clock();
    for (int i = 0; i < N; i++) c[i] = a[i] + b[i];
    double cpu = (double)(clock() - start) / CLOCKS_PER_SEC;

    cudaMalloc((void**)&da, sz);
    cudaMalloc((void**)&db, sz);
    cudaMalloc((void**)&dc, sz);
    cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sz, cudaMemcpyHostToDevice);

    start = clock();
    add<<<(N + 255) / 256, 256>>>(da, db, dc);
    cudaDeviceSynchronize();
    double gpu = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("CPU time = %f s\nGPU time = %f s\nSpeedup  = %.2f\n", cpu, gpu, cpu / gpu);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(a); free(b); free(c);
}
