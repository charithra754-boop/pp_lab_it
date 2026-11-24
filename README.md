# Parallel Programming Lab (23AIL57)

| # | File | Topic |
|---|---|---|
| 1 | `p1_array_add.cu` | CUDA: addition of two arrays |
| 2 | `p2_matmul.c` | OpenMP: matrix multiplication, time vs. thread count |
| 3 | `p3_mergesort.c` | OpenMP: parallel merge sort, speedup vs. sequential |
| 4 | `p4_threadprivate.c` | OpenMP: `threadprivate` and `copyin` |
| 5 | `p5_schedule.c` | OpenMP: `static` / `dynamic` / `guided` scheduling |
| 6 | `p6_primes.c` | OpenMP: prime generation, time / speedup / efficiency |
| 7 | `p7_mpi_sum.c` | MPI: array sum with `MPI_Scatter` + `MPI_Reduce` |
| 8 | `p8_mpi_comm.c` | MPI: `MPI_Send` / `MPI_Recv` + `MPI_Bcast` |
| 9 | `p9_cuda_props.cu` | CUDA: GPU properties + varying blocks/threads |
| 10 | `p10_vec_add_time.cu` | CUDA: vector addition, CPU vs. GPU time |
| 11 | `p11_linreg.py` | PyTorch: linear regression (gradients + SGD) |
| 12 | `p12_ffnn.py` | PyTorch: feedforward neural network on random data |

Sample outputs for all programs are available in `sample_outputs.txt`.

## How to compile and run

At the bottom of each program file, the exact commands to compile and execute that program are also included.

### OpenMP (Programs 2–6)
```sh
gcc -fopenmp p2_matmul.c -o p2
./p2
```

### MPI (Programs 7–8)
```sh
mpicc p7_mpi_sum.c -o p7
mpirun -np 4 ./p7
```

### CUDA (Programs 1, 9, 10)
```sh
nvcc p1_array_add.cu -o p1
./p1
```

### PyTorch (Programs 11–12)
```sh
python3 p11_linreg.py
python3 p12_ffnn.py
```

## Setup

### Local environment
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install mpich nvidia-cuda-nvcc nvidia-cuda-runtime numpy torch
```

On systems where `mpicc`, `nvcc`, and `gcc` are installed system-wide (e.g. lab workstations), the standard commands work directly.

### Google Colab
Open `pp_lab.ipynb` in [Google Colab](https://colab.research.google.com), set the runtime to **T4 GPU** (**Runtime → Change runtime type → T4 GPU**), and select **Runtime → Run all**.
