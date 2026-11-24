#include <stdio.h>
#include <omp.h>

int x;
#pragma omp threadprivate(x)

int main() {
    omp_set_num_threads(4);
    x = 10;

    printf("Region 1 (without copyin):\n");
    #pragma omp parallel
    printf("  Thread %d: x = %d\n", omp_get_thread_num(), x);

    printf("\nRegion 2 (with copyin):\n");
    #pragma omp parallel copyin(x)
    {
        int id = omp_get_thread_num();
        printf("  Thread %d: x = %d\n", id, x);
        x += id;
    }

    printf("\nRegion 3 (persistence):\n");
    #pragma omp parallel
    printf("  Thread %d: x = %d\n", omp_get_thread_num(), x);

    printf("\nMaster thread after all regions: x = %d\n", x);
    return 0;
}
