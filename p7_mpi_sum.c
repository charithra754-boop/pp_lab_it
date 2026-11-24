#include <stdio.h>
#include <mpi.h>
#define N 1000000

int a[N], part[N];

int main(int argc, char *argv[]) {
    int rank, size, local = 0, total = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = N / size;
    if (rank == 0)
        for (int i = 0; i < N; i++) a[i] = 1;

    MPI_Scatter(a, chunk, MPI_INT, part, chunk, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < chunk; i++) local += part[i];
    printf("Process %d: partial sum = %d\n", rank, local);

    MPI_Reduce(&local, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total sum = %d (expected %d)\n", total, N);

    MPI_Finalize();
    return 0;
}
