#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, arr[5] = {0};
    char msg[20] = "Hello from P0";

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        MPI_Send(msg, 20, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent: %s\n", msg);
        for (int i = 0; i < 5; i++) arr[i] = (i + 1) * 10;
    } else if (rank == 1) {
        MPI_Recv(msg, 20, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received: %s\n", msg);
    }

    MPI_Bcast(arr, 5, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d after Bcast: %d %d %d %d %d\n", rank, arr[0], arr[1], arr[2], arr[3], arr[4]);

    MPI_Finalize();
}

// mpicc p8_mpi_comm.c -o p8
// mpirun -np 4 ./p8
