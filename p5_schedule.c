#include <stdio.h>
#include <omp.h>

#define N 16

void work(int i) {
    double s = 0;
    for (long j = 0; j < (i + 1) * 5000000L; j++) {
        s += 1;
    }
    printf("  Iteration %2d -> Thread %d  (work = %.0f)\n", i, omp_get_thread_num(), s);
}

int main() {
    double start;
    omp_set_num_threads(4);

    printf("STATIC (equal blocks of iterations, fixed in advance):\n");
    start = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        work(i);
    }
    printf("Time = %f s\n\n", omp_get_wtime() - start);

    printf("DYNAMIC (free thread takes the next iteration):\n");
    start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        work(i);
    }
    printf("Time = %f s\n\n", omp_get_wtime() - start);

    printf("GUIDED (big chunks first, then smaller chunks):\n");
    start = omp_get_wtime();
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        work(i);
    }
    printf("Time = %f s\n", omp_get_wtime() - start);

    return 0;
}

// gcc -fopenmp p5_schedule.c -o p5
// ./p5
