#include <stdio.h>
#define N 10

__global__ void add(int *a, int *b, int *c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main() {
    int a[N], b[N], c[N], *da, *db, *dc, sz = N * sizeof(int);
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    cudaMalloc((void**)&da, sz);
    cudaMalloc((void**)&db, sz);
    cudaMalloc((void**)&dc, sz);
    cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sz, cudaMemcpyHostToDevice);
    add<<<1, N>>>(da, db, dc);
    cudaMemcpy(c, dc, sz, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}
