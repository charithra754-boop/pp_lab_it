#include <stdio.h>
#include <omp.h>
#define N 500

double A[N][N], B[N][N], C[N][N];

int main() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = 1;
            B[i][j] = 2;
        }

    double t1 = 0;
    for (int t = 1; t <= 8; t *= 2) {
        omp_set_num_threads(t);
        double start = omp_get_wtime();
        #pragma omp parallel for
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double sum = 0;
                for (int k = 0; k < N; k++) sum += A[i][k] * B[k][j];
                C[i][j] = sum;
            }
        double time = omp_get_wtime() - start;
        if (t == 1) t1 = time;
        printf("Threads = %d   Time = %f s   Speedup = %.2f\n", t, time, t1 / time);
    }
    printf("C[0][0] = %.0f (expected %d)\n", C[0][0], 2 * N);
    return 0;
}
