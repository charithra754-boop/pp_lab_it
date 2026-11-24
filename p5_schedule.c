#include <stdio.h>
#include <omp.h>
#define N 16

void work(int i) {
    for (long j = 0; j < (i + 1) * 5000000L; j++);
    printf("  Iteration %2d -> Thread %d\n", i, omp_get_thread_num());
}

int main() {
    omp_set_num_threads(4);
    double t;

    printf("STATIC:\n");
    t = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) work(i);
    printf("Time = %f s\n\n", omp_get_wtime() - t);

    printf("DYNAMIC:\n");
    t = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) work(i);
    printf("Time = %f s\n\n", omp_get_wtime() - t);

    printf("GUIDED:\n");
    t = omp_get_wtime();
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) work(i);
    printf("Time = %f s\n", omp_get_wtime() - t);
}
