#include <stdio.h>
#include <omp.h>
#define N 2000000

int isPrime(int n) {
    for (int i = 2; i * i <= n; i++)
        if (n % i == 0) return 0;
    return n > 1;
}

int main() {
    int count = 0;
    double start = omp_get_wtime();
    for (int i = 2; i <= N; i++) count += isPrime(i);
    double seq = omp_get_wtime() - start;
    printf("Sequential: primes = %d, time = %f s\n\nThreads   Time(s)    Speedup   Efficiency\n", count, seq);

    for (int t = 1; t <= 8; t *= 2) {
        count = 0;
        omp_set_num_threads(t);
        start = omp_get_wtime();
        #pragma omp parallel for reduction(+:count) schedule(dynamic, 1000)
        for (int i = 2; i <= N; i++) count += isPrime(i);
        double time = omp_get_wtime() - start, sp = seq / time;
        printf("%4d    %f    %.2f      %.2f\n", t, time, sp, sp / t);
    }
    return 0;
}
