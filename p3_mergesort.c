#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 2000000

int orig[N], a[N], tmp[N];

void merge(int l, int m, int r) {
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) {
        tmp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    }
    while (i <= m) {
        tmp[k++] = a[i++];
    }
    while (j <= r) {
        tmp[k++] = a[j++];
    }
    for (i = l; i <= r; i++) {
        a[i] = tmp[i];
    }
}

void mergeSort(int l, int r) {
    if (l >= r) return;
    int m = (l + r) / 2;

    #pragma omp task if(r - l > 10000)
    mergeSort(l, m);
    #pragma omp task if(r - l > 10000)
    mergeSort(m + 1, r);
    #pragma omp taskwait
    merge(l, m, r);
}

void reset() {
    for (int i = 0; i < N; i++) {
        a[i] = orig[i];
    }
}

int main() {
    for (int i = 0; i < N; i++) {
        orig[i] = rand();
    }

    reset();
    double start = omp_get_wtime();
    mergeSort(0, N - 1);
    double seq = omp_get_wtime() - start;
    printf("Sequential time = %f s\n\n", seq);

    for (int t = 1; t <= 8; t *= 2) {
        reset();
        omp_set_num_threads(t);
        start = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            mergeSort(0, N - 1);
        }
        double time = omp_get_wtime() - start;
        printf("Threads = %d   Time = %f s   Speedup = %.2f\n", t, time, seq / time);
    }

    for (int i = 1; i < N; i++) {
        if (a[i - 1] > a[i]) {
            printf("NOT sorted!\n");
            return 1;
        }
    }
    printf("\nArray sorted correctly\n");
    return 0;
}

// gcc -fopenmp p3_mergesort.c -o p3
// ./p3
