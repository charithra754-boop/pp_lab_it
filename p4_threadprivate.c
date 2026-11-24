#include <stdio.h>
#include <omp.h>

int x;
#pragma omp threadprivate(x)

int main() {
    omp_set_num_threads(4);
    x = 10;

    printf("Region 1 (without copyin): other threads still have x = 0\n");
    #pragma omp parallel
    {
        printf("  Thread %d: x = %d\n", omp_get_thread_num(), x);
    }

    printf("\nRegion 2 (with copyin): master's x is copied to all threads\n");
    #pragma omp parallel copyin(x)
    {
        int id = omp_get_thread_num();
        printf("  Thread %d: x = %d\n", id, x);
        x = x + id;
    }

    printf("\nRegion 3: each thread's copy keeps its value from region 2\n");
    #pragma omp parallel
    {
        printf("  Thread %d: x = %d\n", omp_get_thread_num(), x);
    }

    printf("\nMaster thread after all regions: x = %d\n", x);
    return 0;
}

// gcc -fopenmp p4_threadprivate.c -o p4
// ./p4
