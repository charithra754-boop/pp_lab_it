# Parallel Programming Laboratory (23AIL57) — Complete Explanatory Guide

This comprehensive reference manual combines all 12 laboratory programs from the Parallel Programming Laboratory course into a single structured, explanatory document. For every program, this guide provides the problem statement, full source code, compilation and execution commands, line-by-line technical explanations, sample outputs, performance analyses, and essential viva/conceptual questions.

---

## Table of Contents

- [Overview of Parallel Computing Paradigms](#overview-of-parallel-computing-paradigms)
- [Program 1: CUDA Addition of Two Arrays (`p1_array_add.cu`)](#program-1-cuda-addition-of-two-arrays-p1_array_addcu)
- [Program 2: OpenMP Parallel Matrix Multiplication (`p2_matmul.c`)](#program-2-openmp-parallel-matrix-multiplication-p2_matmulc)
- [Program 3: OpenMP Parallel Merge Sort with Tasks (`p3_mergesort.c`)](#program-3-openmp-parallel-merge-sort-with-tasks-p3_mergesortc)
- [Program 4: OpenMP Threadprivate & Copyin Directives (`p4_threadprivate.c`)](#program-4-openmp-threadprivate--copyin-directives-p4_threadprivatec)
- [Program 5: OpenMP Loop Scheduling Strategies (`p5_schedule.c`)](#program-5-openmp-loop-scheduling-strategies-p5_schedulec)
- [Program 6: OpenMP Parallel Prime Generation (`p6_primes.c`)](#program-6-openmp-parallel-prime-generation-p6_primesc)
- [Program 7: MPI Parallel Summation using Scatter and Reduce (`p7_mpi_sum.c`)](#program-7-mpi-parallel-summation-using-scatter-and-reduce-p7_mpi_sumc)
- [Program 8: MPI Point-to-Point and Collective Communication (`p8_mpi_comm.c`)](#program-8-mpi-point-to-point-and-collective-communication-p8_mpi_commc)
- [Program 9: CUDA GPU Device Properties & Kernel Grid/Block Hierarchy (`p9_cuda_props.cu`)](#program-9-cuda-gpu-device-properties--kernel-gridblock-hierarchy-p9_cuda_propscu)
- [Program 10: CUDA Vector Addition Performance vs. CPU (`p10_vec_add_time.cu`)](#program-10-cuda-vector-addition-performance-vs-cpu-p10_vec_add_timecu)
- [Program 11: PyTorch Linear Regression & Optimization (`p11_linreg.py`)](#program-11-pytorch-linear-regression--optimization-p11_linregpy)
- [Program 12: PyTorch Feedforward Neural Network (`p12_ffnn.py`)](#program-12-pytorch-feedforward-neural-network-p12_ffnnpy)
- [Comparative Architectural Summary](#comparative-architectural-summary)
- [Quick Execution Cheatsheet](#quick-execution-cheatsheet)

---

## Overview of Parallel Computing Paradigms

This lab explores four distinct computing paradigms that address different architectural levels of parallel hardware:

| Paradigm | Target Hardware | Memory Architecture | Communication Mechanism | Key Abstraction |
|---|---|---|---|---|
| **OpenMP** | Multi-core CPUs | Uniform / Shared Memory | Shared variables, compiler directives (`#pragma omp`) | Threads, worksharing loops, tasking |
| **MPI** | Clusters, distributed nodes | Distributed Memory | Explicit message passing via interconnect | Processes, ranks, communicators, scatter/reduce |
| **CUDA** | NVIDIA GPUs | Heterogeneous / Device Memory | Host-Device transfers via PCIe, Shared/Global GPU memory | Kernels, grids, blocks, threads, SIMT execution |
| **PyTorch** | CPU / GPU / TPU | Tensors with Autograd | High-level tensor operations & backward graph dispatch | Dynamic computational graphs, autograd, optimizers |

---

## Program 1: CUDA Addition of Two Arrays (`p1_array_add.cu`)

### Objective
Write a CUDA program to perform addition of two arrays on an NVIDIA GPU using a basic 1D grid/block configuration.

### Complete Source Code
```cuda
#include <stdio.h>
#define N 10

__global__ void add(int *a, int *b, int *c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main() {
    int a[N], b[N], c[N], *da, *db, *dc, sz = N * sizeof(int);
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    cudaMalloc((void**)&da, sz);
    cudaMalloc((void**)&db, sz);
    cudaMalloc((void**)&dc, sz);
    cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sz, cudaMemcpyHostToDevice);
    add<<<1, N>>>(da, db, dc);
    cudaMemcpy(c, dc, sz, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

// nvcc p1_array_add.cu -o p1
// ./p1
```

### Compilation and Execution
```bash
nvcc p1_array_add.cu -o p1
./p1
```

### Detailed Explanation

1. **Kernel Declaration (`__global__ void add(...)`)**:
   - `__global__` is a CUDA execution space specifier indicating a function that runs on the device (GPU) and is callable from the host (CPU).
   - In CUDA's Single Instruction, Multiple Threads (SIMT) model, each GPU thread executes this function concurrently.
   - `threadIdx.x` is a built-in CUDA variable providing the 0-based index of the calling thread within its thread block.
   - Line `c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];` ensures thread $i$ computes the sum of the $i$-th elements without any loop construct.

2. **Host Data Initialization**:
   - Host arrays `a[N]`, `b[N]`, and `c[N]` reside in CPU system RAM.
   - `a[i] = i` and `b[i] = 2 * i` initialize sample values.

3. **Device Memory Allocation (`cudaMalloc`)**:
   - `cudaMalloc((void**)&da, sz)` allocates linear physical memory directly on the GPU's onboard High Bandwidth Memory (VRAM).
   - `da`, `db`, and `dc` are device pointers stored on the CPU that point to GPU memory addresses.

4. **Data Transfer (`cudaMemcpy`)**:
   - `cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice)` transfers memory from host RAM to device VRAM across the PCIe bus.
   - Both inputs `a` and `b` must be placed in GPU memory before kernel launch.

5. **Kernel Launch Execution Configuration (`add<<<1, N>>>(da, db, dc)`)**:
   - The triple chevron syntax `<<<gridDim, blockDim>>>` configures how GPU resources are dispatched.
   - `1`: The grid consists of exactly 1 thread block.
   - `N` (10): The block contains 10 threads running in parallel.

6. **Retrieving Results and Cleanup**:
   - `cudaMemcpy(c, dc, sz, cudaMemcpyDeviceToHost)` copies the computed results back into CPU memory.
   - `cudaFree` releases the device allocations to prevent VRAM memory leaks.

### Sample Output
```text
0 + 0 = 0
1 + 2 = 3
2 + 4 = 6
3 + 6 = 9
4 + 8 = 12
5 + 10 = 15
6 + 12 = 18
7 + 14 = 21
8 + 16 = 24
9 + 18 = 27
```

### Viva & Conceptual Questions
- **Q: What is the difference between `__global__`, `__device__`, and `__host__` in CUDA?**
  - `__global__`: Runs on device (GPU), invoked from host (CPU) or device (CUDA dynamic parallelism). Returns `void`.
  - `__device__`: Runs on device and can only be called by other device functions.
  - `__host__`: Runs on host and can only be called from host (default C/C++ function).
- **Q: Why can't CPU functions directly dereference device pointers like `da`?**
  - Device memory addresses exist in an isolated virtual address space physically mapped to GPU VRAM, causing segmentation faults if accessed directly by the CPU without Unified Memory.

---

## Program 2: OpenMP Parallel Matrix Multiplication (`p2_matmul.c`)

### Objective
Implement parallel matrix multiplication using OpenMP and measure execution time and speedup across varying thread counts ($t = 1, 2, 4, 8$).

### Complete Source Code
```c
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

// gcc -fopenmp p2_matmul.c -o p2
// ./p2
```

### Compilation and Execution
```bash
gcc -fopenmp p2_matmul.c -o p2
./p2
```

### Detailed Explanation

1. **Matrix Storage & Size**:
   - `A`, `B`, and `C` are statically allocated global 2D arrays of size $500 \times 500$ doubles.
   - Global allocation prevents stack overflow, which would occur if large matrices were declared on the local thread stack.

2. **Computational Complexity**:
   - Standard matrix multiplication has $O(N^3)$ arithmetic operations. For $N=500$, there are $500^3 = 125,000,000$ multiply-accumulate operations per test run.

3. **Loop Parallelization with OpenMP (`#pragma omp parallel for`)**:
   - This directive instructs the compiler to fork a team of $t$ threads and partition the outer loop iterations (`i` from $0$ to $N-1$) among the threads.
   - Loop index `i` is implicitly made `private` to each thread.
   - Inner indices `j` and `k`, as well as accumulator `double sum`, are scoped inside the outer loop body, ensuring they are thread-local and eliminating race conditions.
   - Because each thread writes to distinct rows `C[i][*]`, no two threads write to the same memory location, avoiding lock contention and false sharing on row boundaries.

4. **Timing & Speedup Calculation**:
   - `omp_get_wtime()` returns high-resolution wall-clock time in seconds.
   - When $t = 1$, execution time $t_1$ is captured as the sequential baseline.
   - Speedup is computed as $S_t = \frac{T_1}{T_t}$.

5. **Verification**:
   - Every element of $A$ is $1$ and every element of $B$ is $2$.
   - The dot product of row $i$ with column $j$ is $\sum_{k=0}^{N-1} (1 \times 2) = 2N = 1000$. Checking `C[0][0] == 1000` verifies numerical correctness.

### Sample Output
```text
Threads = 1   Time = 0.349141 s   Speedup = 1.00
Threads = 2   Time = 0.179182 s   Speedup = 1.95
Threads = 4   Time = 0.092682 s   Speedup = 3.77
Threads = 8   Time = 0.046766 s   Speedup = 7.47
C[0][0] = 1000 (expected 1000)
```

### Viva & Conceptual Questions
- **Q: Why parallelize the outer loop (`i`) instead of the inner loops (`j` or `k`)?**
  - Parallelizing the outer loop maximizes the granularity of work assigned to each thread, minimizing thread creation/synchronization overhead. If `k` were parallelized, threads would race on `sum` or require expensive reduction synchronization on every iteration.
- **Q: What is speedup and why does it rarely reach an ideal $t\times$ factor?**
  - Amdahl's Law states that maximum speedup is bounded by the sequential portion of the code. Thread synchronization overhead, memory bus saturation, and CPU cache contention prevent ideal linear speedup.

---

## Program 3: OpenMP Parallel Merge Sort with Tasks (`p3_mergesort.c`)

### Objective
Develop a parallel recursive merge sort algorithm using OpenMP task-based parallelism (`#pragma omp task`) and analyze speedup compared to sequential execution.

### Complete Source Code
```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 2000000

int a[N], tmp[N], orig[N];

void merge(int l, int m, int r) {
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) tmp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i <= m) tmp[k++] = a[i++];
    while (j <= r) tmp[k++] = a[j++];
    for (i = l; i <= r; i++) a[i] = tmp[i];
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

int main() {
    for (int i = 0; i < N; i++) a[i] = orig[i] = rand();
    double start = omp_get_wtime();
    mergeSort(0, N - 1);
    double seq = omp_get_wtime() - start;
    printf("Sequential time = %f s\n\n", seq);

    for (int t = 1; t <= 8; t *= 2) {
        for (int i = 0; i < N; i++) a[i] = orig[i];
        omp_set_num_threads(t);
        start = omp_get_wtime();
        #pragma omp parallel
        #pragma omp single
        mergeSort(0, N - 1);
        double time = omp_get_wtime() - start;
        printf("Threads = %d   Time = %f s   Speedup = %.2f\n", t, time, seq / time);
    }

    for (int i = 1; i < N; i++)
        if (a[i - 1] > a[i]) { printf("NOT sorted!\n"); return 1; }
    printf("\nArray sorted correctly\n");
    return 0;
}

// gcc -fopenmp p3_mergesort.c -o p3
// ./p3
```

### Compilation and Execution
```bash
gcc -fopenmp p3_mergesort.c -o p3
./p3
```

### Detailed Explanation

1. **Divide and Conquer with OpenMP Tasks**:
   - Merge sort naturally forms a binary tree of sub-problems.
   - `#pragma omp task` packages the recursive call into an independent unit of work (task) placed into an OpenMP thread pool queue. Idle threads in the team can pick up and execute tasks asynchronously.

2. **The Cutoff Threshold (`if(r - l > 10000)`)**:
   - Creating an OpenMP task incurs internal memory allocation and queueing overhead.
   - If tasks were generated down to sub-arrays of size 1 or 2, the runtime overhead would drastically exceed the sorting computation.
   - The clause `if(r - l > 10000)` ensures tasks are only created for sub-problems larger than 10,000 items. Smaller sub-problems execute synchronously on the current thread.

3. **Synchronization Barrier (`#pragma omp taskwait`)**:
   - Merging two halves requires both sub-arrays to be completely sorted first.
   - `#pragma omp taskwait` suspends the parent task until both child tasks (left and right recursive sorts) have finished execution.

4. **Task Seeding Pattern (`#pragma omp parallel` + `#pragma omp single`)**:
   - `#pragma omp parallel` spawns a team of $t$ worker threads.
   - `#pragma omp single` directs **one single thread** to initiate the initial `mergeSort(0, N - 1)` call.
   - As that single thread encounters `#pragma omp task` directives, it generates tasks and pushes them to the shared queue, from which all other threads in the team steal and execute work.

5. **Sorting Verification**:
   - The loop checks `a[i - 1] <= a[i]` across all $2,000,000$ integers to guarantee sorting correctness before terminating.

### Sample Output
```text
Sequential time = 0.303634 s

Threads = 1   Time = 0.304487 s   Speedup = 1.00
Threads = 2   Time = 0.165270 s   Speedup = 1.84
Threads = 4   Time = 0.136939 s   Speedup = 2.22
Threads = 8   Time = 0.071637 s   Speedup = 4.24

Array sorted correctly
```

### Viva & Conceptual Questions
- **Q: What happens if `#pragma omp taskwait` is omitted in parallel merge sort?**
  - The `merge()` function would execute concurrently with incomplete `mergeSort()` tasks, reading partially sorted or unsorted segments and corrupting array order.
- **Q: Why is `#pragma omp single` essential when kicking off tasks?**
  - Without `#pragma omp single`, all $t$ threads in the parallel region would invoke `mergeSort(0, N - 1)` simultaneously, leading to duplicate redundant sorting trees and race conditions.

---

## Program 4: OpenMP Threadprivate & Copyin Directives (`p4_threadprivate.c`)

### Objective
Demonstrate how to use `threadprivate` and `copyin` directives in OpenMP to make file-scope/global variables local to each thread and observe data persistence across parallel regions.

### Complete Source Code
```c
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

// gcc -fopenmp p4_threadprivate.c -o p4
// ./p4
```

### Compilation and Execution
```bash
gcc -fopenmp p4_threadprivate.c -o p4
./p4
```

### Detailed Explanation

1. **`#pragma omp threadprivate(x)`**:
   - `x` is declared at file scope. The `threadprivate` directive specifies that every thread gets its own private, persistent copy of `x` stored in thread-local storage (TLS).
   - Unlike standard `private(x)` variables that are created and destroyed within a parallel block, `threadprivate` variables **persist across multiple consecutive parallel regions**.

2. **Region 1 (Without `copyin`)**:
   - The master thread (thread 0) sets its copy of `x = 10` in sequential code.
   - When entering the first parallel region without `copyin`, non-master threads (1, 2, 3) retain their uninitialized default storage (which is `0` for global variables).
   - Output shows Thread 0 sees `10`, while Threads 1, 2, and 3 see `0`.

3. **Region 2 (With `copyin(x)`)**:
   - The `copyin(x)` clause broadcasts the value from the master thread's instance (`10`) to the private instances of all other threads at the start of the parallel region.
   - All threads now start with `x = 10`.
   - Each thread adds its thread ID to its local copy: `x += id`.
     - Thread 0: $10 + 0 = 10$
     - Thread 1: $10 + 1 = 11$
     - Thread 2: $10 + 2 = 12$
     - Thread 3: $10 + 3 = 13$

4. **Region 3 (Data Persistence Demonstration)**:
   - Entered without `copyin`.
   - Each thread retains the modified value from Region 2.
   - Proves that `threadprivate` memory is preserved between distinct `#pragma omp parallel` blocks for identical thread counts.

5. **Sequential Code Post-Region**:
   - In sequential execution outside the parallel regions, only the master thread executes.
   - `x` references thread 0's private instance, retaining `10`.

### Sample Output
```text
Region 1 (without copyin):
  Thread 3: x = 0
  Thread 1: x = 0
  Thread 0: x = 10
  Thread 2: x = 0

Region 2 (with copyin):
  Thread 1: x = 10
  Thread 0: x = 10
  Thread 3: x = 10
  Thread 2: x = 10

Region 3 (persistence):
  Thread 0: x = 10
  Thread 2: x = 12
  Thread 1: x = 11
  Thread 3: x = 13

Master thread after all regions: x = 10
```

### Viva & Conceptual Questions
- **Q: What is the key distinction between `private` and `threadprivate` in OpenMP?**
  - A `private` variable is allocated upon entering a parallel construct and destroyed at construct exit. A `threadprivate` variable persists in memory throughout the lifetime of the process across multiple parallel constructs.
- **Q: What does `copyin` do?**
  - It copies the master thread's current value of a `threadprivate` variable into the corresponding threadprivate variables of all other threads upon entering a parallel region.

---

## Program 5: OpenMP Loop Scheduling Strategies (`p5_schedule.c`)

### Objective
Develop an OpenMP program using various scheduling strategies (`static`, `dynamic`, `guided`) to observe their effect on loop performance and load balancing when workload varies significantly per iteration.

### Complete Source Code
```c
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

// gcc -fopenmp p5_schedule.c -o p5
// ./p5
```

### Compilation and Execution
```bash
gcc -fopenmp p5_schedule.c -o p5
./p5
```

### Detailed Explanation

1. **Synthetic Imbalanced Workload**:
   - `work(i)` executes a loop with $(i + 1) \times 5,000,000$ iterations.
   - Iteration $0$ takes $5 \times 10^6$ operations, while iteration $15$ takes $80 \times 10^6$ operations (a 16x workload disparity).
   - This provides the ideal benchmark for evaluating load balancing across scheduling algorithms.

2. **Static Scheduling (`schedule(static)`)**:
   - Loop iterations are statically divided into equal contiguous chunks prior to loop execution:
     - Thread 0: Iterations 0, 1, 2, 3 (Total work units: $1 + 2 + 3 + 4 = 10$)
     - Thread 1: Iterations 4, 5, 6, 7 (Total work units: $5 + 6 + 7 + 8 = 26$)
     - Thread 2: Iterations 8, 9, 10, 11 (Total work units: $9 + 10 + 11 + 12 = 42$)
     - Thread 3: Iterations 12, 13, 14, 15 (Total work units: $13 + 14 + 15 + 16 = 58$)
   - **Flaw**: Thread 0 finishes almost instantly and sits idle waiting at the implicit barrier, while Thread 3 is overwhelmed. Overall runtime is bounded by the slowest thread ($0.82\text{ s}$).

3. **Dynamic Scheduling (`schedule(dynamic)`)**:
   - Iterations (default chunk size = 1) are placed in a shared queue.
   - When a thread finishes an iteration, it requests the next available iteration dynamically.
   - Because early iterations are short, threads finish them rapidly and share the heavier remaining iterations evenly.
   - **Result**: Drastic reduction in wall-clock time from $0.82\text{ s}$ to $0.57\text{ s}$.

4. **Guided Scheduling (`schedule(guided)`)**:
   - Allocates exponentially decreasing chunk sizes over time.
   - Initial chunk size is proportional to $\frac{\text{unassigned iterations}}{\text{number of threads}}$.
   - Decreases chunk size toward 1 for later iterations to smooth out load imbalance while minimizing lock/queue contention.
   - **Result**: Lowest runtime ($0.56\text{ s}$), providing optimal balance between low scheduling overhead and dynamic load distribution.

### Sample Output
```text
STATIC:
  Iteration  0 -> Thread 0
  Iteration  1 -> Thread 0
  Iteration  4 -> Thread 1
  Iteration  2 -> Thread 0
  Iteration  8 -> Thread 2
  Iteration  3 -> Thread 0
  Iteration  5 -> Thread 1
  Iteration 12 -> Thread 3
  Iteration  6 -> Thread 1
  Iteration  9 -> Thread 2
  Iteration  7 -> Thread 1
  Iteration 13 -> Thread 3
  Iteration 10 -> Thread 2
  Iteration 11 -> Thread 2
  Iteration 14 -> Thread 3
  Iteration 15 -> Thread 3
Time = 0.821918 s

DYNAMIC:
  Iteration  0 -> Thread 3
  Iteration  1 -> Thread 0
  Iteration  2 -> Thread 2
  Iteration  3 -> Thread 1
  Iteration  4 -> Thread 3
  Iteration  5 -> Thread 0
  Iteration  6 -> Thread 2
  Iteration  7 -> Thread 1
  Iteration  8 -> Thread 3
  Iteration  9 -> Thread 0
  Iteration 10 -> Thread 2
  Iteration 11 -> Thread 1
  Iteration 12 -> Thread 3
  Iteration 13 -> Thread 0
  Iteration 14 -> Thread 2
  Iteration 15 -> Thread 1
Time = 0.574418 s

GUIDED:
  Iteration  0 -> Thread 3
  Iteration  1 -> Thread 3
  Iteration  4 -> Thread 0
  Iteration  2 -> Thread 3
  Iteration  7 -> Thread 1
  Iteration  3 -> Thread 3
  Iteration 10 -> Thread 2
  Iteration  5 -> Thread 0
  Iteration  8 -> Thread 1
  Iteration  6 -> Thread 0
  Iteration 12 -> Thread 3
  Iteration 11 -> Thread 2
  Iteration  9 -> Thread 1
  Iteration 13 -> Thread 0
  Iteration 14 -> Thread 3
  Iteration 15 -> Thread 2
Time = 0.562561 s
```

### Viva & Conceptual Questions
- **Q: When should `schedule(static)` be chosen over `dynamic`?**
  - When loop iterations have nearly identical computation time (uniform workload). In that case, static scheduling has zero runtime scheduling overhead and optimal cache locality.
- **Q: What is the main drawback of `dynamic` scheduling?**
  - Thread synchronization overhead: every time a thread needs a new chunk, it must access a shared queue protected by internal synchronization, which degrades performance for trivial loop bodies.

---

## Program 6: OpenMP Parallel Prime Generation (`p6_primes.c`)

### Objective
Implement a parallel prime number generation algorithm using OpenMP. Compare the performance of sequential and parallel executions by varying thread counts and analyze execution time, speedup, and efficiency.

### Complete Source Code
```c
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

// gcc -fopenmp p6_primes.c -o p6
// ./p6
```

### Compilation and Execution
```bash
gcc -fopenmp p6_primes.c -o p6
./p6
```

### Detailed Explanation

1. **Prime Testing Function (`isPrime`)**:
   - Tests primality using trial division up to $\lfloor\sqrt{n}\rfloor$.
   - Testing larger numbers takes significantly more loop iterations than testing smaller ones (e.g. testing $n = 1,999,993$ tests up to $\sqrt{n} \approx 1414$, while testing $n = 100$ stops at $10$).

2. **Reduction Clause (`reduction(+:count)`)**:
   - If multiple threads update a shared variable `count += ...` directly, a race condition occurs, causing incorrect counts.
   - The `reduction(+:count)` clause creates a private accumulator for each thread initialized to `0`.
   - At the completion of the parallel loop, OpenMP atomically sums the private accumulators of all threads into the master `count` variable.

3. **Dynamic Chunk Scheduling (`schedule(dynamic, 1000)`)**:
   - Because primality testing time increases with $n$, static scheduling would allocate harder numbers to later threads.
   - `schedule(dynamic, 1000)` dishes out chunks of 1000 consecutive integers to whatever thread becomes available, maintaining even CPU utilization. A chunk size of 1000 amortizes scheduling overhead while preserving load balance.

4. **Metrics**:
   - **Speedup ($S_t$)**: $S_t = \frac{T_{\text{seq}}}{T_t}$
   - **Efficiency ($E_t$)**: $E_t = \frac{S_t}{t} = \frac{T_{\text{seq}}}{t \times T_t}$. Ideal efficiency is $1.00$ ($100\%$).

### Sample Output
```text
Sequential: primes = 148933, time = 0.257054 s

Threads   Time(s)    Speedup   Efficiency
   1    0.254630    1.01      1.01
   2    0.132397    1.94      0.97
   4    0.067926    3.78      0.95
   8    0.034402    7.47      0.93
```

### Viva & Conceptual Questions
- **Q: Why does efficiency decline slightly from 0.97 at 2 threads to 0.93 at 8 threads?**
  - Overhead from thread synchronization, task dispatching for dynamic chunks, memory bus contention, and reduction accumulation increases as thread count grows.
- **Q: How does OpenMP implement a reduction under the hood?**
  - It creates a thread-local copy of the variable for each thread, performs local updates without locking, and applies a binary tree or atomic addition to combine the local copies at the end.

---

## Program 7: MPI Parallel Summation using Scatter and Reduce (`p7_mpi_sum.c`)

### Objective
Implement parallel summation of a large array using MPI. Use `MPI_Scatter` to distribute segments across processes, compute local partial sums, and use `MPI_Reduce` to combine the results.

### Complete Source Code
```c
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
}

// mpicc p7_mpi_sum.c -o p7
// mpirun -np 4 ./p7
```

### Compilation and Execution
```bash
mpicc p7_mpi_sum.c -o p7
mpirun -np 4 ./p7
```

### Detailed Explanation

1. **MPI Architecture & Setup**:
   - `MPI_Init(&argc, &argv)` initializes the MPI execution environment.
   - `MPI_COMM_WORLD` is the default communicator containing all processes launched by `mpirun`.
   - `MPI_Comm_rank` returns the calling process's unique ID (`0` to `size - 1`).
   - `MPI_Comm_size` returns the total number of processes (e.g. 4).

2. **Distributed Memory Model**:
   - Unlike OpenMP (where threads share the same address space), each MPI process runs as an independent OS process with its own private virtual memory space.
   - Array `a` only needs to be initialized on Process 0 (`rank == 0`).

3. **Data Distribution with `MPI_Scatter`**:
   - Signature:
     ```c
     MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm)
     ```
   - Process 0 (`root = 0`) divides its buffer `a` into contiguous chunks of size `chunk = N / size` ($1,000,000 / 4 = 250,000$) and sends one chunk to each process in `MPI_COMM_WORLD`.
   - Each process receives its portion into local buffer `part`.

4. **Local Computation**:
   - Each process computes `local += part[i]` for its $250,000$ elements completely independently, without any network traffic or cross-process locks.

5. **Result Aggregation with `MPI_Reduce`**:
   - Signature:
     ```c
     MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm)
     ```
   - Each process sends its scalar `local` value.
   - The MPI runtime applies the reduction operation `MPI_SUM` across all values and delivers the aggregated scalar result `total` exclusively to the root process (`rank == 0`).
   - Only Process 0 prints the final total sum.

6. **Cleanup**:
   - `MPI_Finalize()` cleans up MPI state and terminates communication channels before process exit.

### Sample Output
```text
Process 1: partial sum = 250000
Process 3: partial sum = 250000
Process 0: partial sum = 250000
Process 2: partial sum = 250000
Total sum = 1000000 (expected 1000000)
```
*(Note: Output order of the partial sum lines is non-deterministic because processes execute concurrently).*

### Viva & Conceptual Questions
- **Q: What is the difference between `MPI_Reduce` and `MPI_Allreduce`?**
  - `MPI_Reduce` returns the combined result only to the designated `root` process. `MPI_Allreduce` combines the results and broadcasts the final value to **all** processes in the communicator.
- **Q: What is a collective communication in MPI?**
  - A collective operation (like `MPI_Scatter`, `MPI_Bcast`, `MPI_Reduce`, or `MPI_Barrier`) is one that requires **every** process in the communicator to invoke the function.

---

## Program 8: MPI Point-to-Point and Collective Communication (`p8_mpi_comm.c`)

### Objective
Create an MPI program to demonstrate both point-to-point communication (`MPI_Send`, `MPI_Recv`) and collective broadcast communication (`MPI_Bcast`).

### Complete Source Code
```c
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
```

### Compilation and Execution
```bash
mpicc p8_mpi_comm.c -o p8
mpirun -np 4 ./p8
```

### Detailed Explanation

1. **Point-to-Point Communication (`MPI_Send` and `MPI_Recv`)**:
   - Point-to-point communication involves an explicit sender and a single designated receiver.
   - `MPI_Send(msg, 20, MPI_CHAR, 1, 0, MPI_COMM_WORLD)`:
     - Process 0 transmits a 20-byte character array to destination process `1` using message tag `0`.
   - `MPI_Recv(msg, 20, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE)`:
     - Process 1 explicitly waits for a message from source `0` with tag `0`.
     - Processes 2 and 3 do not participate in this exchange and skip the `if/else if` block.

2. **Collective Broadcast (`MPI_Bcast`)**:
   - Signature:
     ```c
     MPI_Bcast(buffer, count, datatype, root, comm)
     ```
   - In contrast to point-to-point, `MPI_Bcast` **must be called by all processes** within the communicator.
   - Process 0 acts as `root`: its `arr` contents (`{10, 20, 30, 40, 50}`) are copied over the network to the `arr` buffers of Processes 1, 2, and 3.
   - For non-root processes, the same call acts as a receive operation.

3. **Comparison of Data Transfer**:
   - `MPI_Send` / `MPI_Recv` sends 1-to-1.
   - `MPI_Bcast` sends 1-to-All (replicating identical data to every process).
   - `MPI_Scatter` sends 1-to-All (partitioning distinct chunks of data to each process).

### Sample Output
```text
Process 0 sent: Hello from P0
Process 0 after Bcast: 10 20 30 40 50
Process 1 received: Hello from P0
Process 1 after Bcast: 10 20 30 40 50
Process 2 after Bcast: 10 20 30 40 50
Process 3 after Bcast: 10 20 30 40 50
```

### Viva & Conceptual Questions
- **Q: What is a deadlock in MPI point-to-point communication?**
  - If two processes both call blocking `MPI_Recv` waiting for each other before calling `MPI_Send`, both wait indefinitely, causing a deadlock.
- **Q: What does `MPI_STATUS_IGNORE` mean?**
  - It indicates that the programmer does not need message metadata (such as actual bytes received, error code, or source rank) that would otherwise be stored in an `MPI_Status` structure.

---

## Program 9: CUDA GPU Device Properties & Kernel Grid/Block Hierarchy (`p9_cuda_props.cu`)

### Objective
Query and display hardware properties of the host GPU and demonstrate CUDA kernel execution under various 1D grid and block dimensions to explore thread coordinate mapping.

### Complete Source Code
```cuda
#include <stdio.h>

__global__ void hello() {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block %d, Thread %d, Global ID %d\n", blockIdx.x, threadIdx.x, id);
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    printf("GPU Name             : %s\n", p.name);
    printf("Compute Capability   : %d.%d\n", p.major, p.minor);
    printf("Multiprocessors (SM) : %d\n", p.multiProcessorCount);
    printf("Warp Size            : %d\n", p.warpSize);
    printf("Max Threads/Block    : %d\n", p.maxThreadsPerBlock);

    printf("\n<<<1 block, 4 threads>>>\n");
    hello<<<1, 4>>>();
    cudaDeviceSynchronize();

    printf("\n<<<2 blocks, 4 threads>>>\n");
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();

    printf("\n<<<4 blocks, 2 threads>>>\n");
    hello<<<4, 2>>>();
    cudaDeviceSynchronize();
}

// nvcc p9_cuda_props.cu -o p9
// ./p9
```

### Compilation and Execution
```bash
nvcc p9_cuda_props.cu -o p9
./p9
```

### Detailed Explanation

1. **Hardware Querying with `cudaGetDeviceProperties`**:
   - `cudaDeviceProp` is a CUDA runtime struct containing device attributes.
   - `cudaGetDeviceProperties(&p, 0)` inspects GPU device 0.
   - Key attributes printed:
     - `p.name`: Hardware product branding (e.g. NVIDIA GeForce RTX 3050 Laptop GPU).
     - `p.major` & `p.minor`: Compute capability version (e.g. 8.6 for Ampere architecture).
     - `p.multiProcessorCount`: Number of Streaming Multiprocessors (SMs) on the chip.
     - `p.warpSize`: Number of threads executed in lockstep (always 32 on modern NVIDIA GPUs).
     - `p.maxThreadsPerBlock`: Maximum threads allowed in a single thread block (typically 1024).

2. **The CUDA Thread Indexing Formula**:
   - In 1D grid/block layout, the unique global thread index is:
     $$\text{Global ID} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
   - `blockIdx.x`: Index of the current block within the grid.
   - `blockDim.x`: Dimension (number of threads) of each block.
   - `threadIdx.x`: Local index of the thread within its block.

3. **Comparison of Execution Configurations**:
   - Configuration 1: `<<<1, 4>>>` $\rightarrow$ 1 block with 4 threads $\implies$ Global IDs 0, 1, 2, 3.
   - Configuration 2: `<<<2, 4>>>` $\rightarrow$ 2 blocks of 4 threads $\implies$ Total 8 threads (Block 0: IDs 0..3; Block 1: IDs 4..7).
   - Configuration 3: `<<<4, 2>>>` $\rightarrow$ 4 blocks of 2 threads $\implies$ Total 8 threads (Block 0: IDs 0..1; Block 1: IDs 2..3; Block 2: IDs 4..5; Block 3: IDs 6..7).

4. **Synchronization (`cudaDeviceSynchronize`)**:
   - Kernel launches are asynchronous: the CPU issues the launch command and immediately resumes execution.
   - `printf` output from GPU threads is stored in an internal device FIFO buffer.
   - `cudaDeviceSynchronize()` forces the host CPU to block until all active GPU kernels have finished and flushes the device output buffer to stdout.

### Sample Output
```text
GPU Name             : NVIDIA GeForce RTX 3050 Laptop GPU
Compute Capability   : 8.6
Multiprocessors (SM) : 16
Warp Size            : 32
Max Threads/Block    : 1024

<<<1 block, 4 threads>>>
Block 0, Thread 0, Global ID 0
Block 0, Thread 1, Global ID 1
Block 0, Thread 2, Global ID 2
Block 0, Thread 3, Global ID 3

<<<2 blocks, 4 threads>>>
Block 0, Thread 0, Global ID 0
Block 0, Thread 1, Global ID 1
Block 0, Thread 2, Global ID 2
Block 0, Thread 3, Global ID 3
Block 1, Thread 0, Global ID 4
Block 1, Thread 1, Global ID 5
Block 1, Thread 2, Global ID 6
Block 1, Thread 3, Global ID 7

<<<4 blocks, 2 threads>>>
Block 0, Thread 0, Global ID 0
Block 0, Thread 1, Global ID 1
Block 1, Thread 0, Global ID 2
Block 1, Thread 1, Global ID 3
Block 2, Thread 0, Global ID 4
Block 2, Thread 1, Global ID 5
Block 3, Thread 0, Global ID 6
Block 3, Thread 1, Global ID 7
```

### Viva & Conceptual Questions
- **Q: What is a Warp in CUDA?**
  - A warp is a group of 32 consecutive threads within a block that execute instructions simultaneously in lockstep on the hardware (SIMT).
- **Q: Why should the number of threads per block typically be a multiple of 32?**
  - Because thread scheduling on the GPU occurs at warp granularity. If block size is not a multiple of 32, the final warp contains idle/inactive threads that consume SM warp slots without doing useful work.

---

## Program 10: CUDA Vector Addition Performance vs. CPU (`p10_vec_add_time.cu`)

### Objective
Write a CUDA program for large-scale vector addition ($N = 10,000,000$ elements) and compare execution time and speedup against a single-threaded CPU implementation.

### Complete Source Code
```cuda
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 10000000

__global__ void add(float *a, float *b, float *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

int main() {
    int sz = N * sizeof(float);
    float *a = (float*)malloc(sz), *b = (float*)malloc(sz), *c = (float*)malloc(sz), *da, *db, *dc;

    for (int i = 0; i < N; i++) { a[i] = i; b[i] = i * 2; }

    clock_t start = clock();
    for (int i = 0; i < N; i++) c[i] = a[i] + b[i];
    double cpu = (double)(clock() - start) / CLOCKS_PER_SEC;

    cudaMalloc((void**)&da, sz);
    cudaMalloc((void**)&db, sz);
    cudaMalloc((void**)&dc, sz);
    cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sz, cudaMemcpyHostToDevice);

    start = clock();
    add<<<(N + 255) / 256, 256>>>(da, db, dc);
    cudaDeviceSynchronize();
    double gpu = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("CPU time = %f s\nGPU time = %f s\nSpeedup  = %.2f\n", cpu, gpu, cpu / gpu);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(a); free(b); free(c);
}

// nvcc p10_vec_add_time.cu -o p10
// ./p10
```

### Compilation and Execution
```bash
nvcc p10_vec_add_time.cu -o p10
./p10
```

### Detailed Explanation

1. **Problem Scale**:
   - $N = 10,000,000$ single-precision floats $\implies 10^7 \times 4\text{ bytes} \approx 40\text{ MB}$ per array.
   - Total memory footprint for three vectors: $\approx 120\text{ MB}$.
   - Large data size ensures the GPU can saturate its memory bandwidth and overcome invocation latencies.

2. **Boundary Check (`if (i < N)`)**:
   - When launching grid dimensions that do not divide $N$ evenly, the total number of launched threads ($\text{blocks} \times \text{threadsPerBlock}$) will exceed $N$.
   - The condition `if (i < N)` prevents threads with out-of-bounds indices from accessing invalid memory and corrupting state.

3. **Ceiling Division for Grid Size**:
   - Configuration: `add<<<(N + 255) / 256, 256>>>(da, db, dc);`
   - Block size is chosen as 256 (a multiple of warp size 32).
   - `(N + 255) / 256` implements integer ceiling division $\lceil N / 256 \rceil$, producing $39,063$ blocks, which provides enough threads ($39,063 \times 256 = 10,000,128$) to cover all $10,000,000$ elements.

4. **Performance Comparison**:
   - The CPU executes vector addition sequentially in $O(N)$ serial steps, limited by single-core clock frequency and L3 cache misses.
   - The GPU launches $10^7$ threads scheduled across dozens of Streaming Multiprocessors, executing addition across thousands of cores concurrently with high memory bandwidth.
   - **Result**: GPU executes in $\approx 0.00088\text{ s}$ compared to $\approx 0.0311\text{ s}$ on CPU, achieving over $35\times$ speedup.

### Sample Output
```text
Results match
CPU time = 0.031129 s
GPU time = 0.000880 s
Speedup  = 35.37
```

### Viva & Conceptual Questions
- **Q: Why is integer ceiling division written as `(N + blockDim - 1) / blockDim` in CUDA?**
  - Standard integer division truncates fractions toward zero. Adding `blockDim - 1` ensures any fractional block rounds up to a complete block, ensuring no array elements are left unprocessed.
- **Q: What is the primary bottleneck for simple vector addition kernels on GPUs?**
  - Memory bandwidth (Memory-bound kernel). The arithmetic intensity is very low (one addition per two 4-byte reads and one 4-byte write), so execution time is dictated by how fast the GPU memory bus can transfer data.

---

## Program 11: PyTorch Linear Regression & Optimization (`p11_linreg.py`)

### Objective
Develop a Linear Regression model using PyTorch to demonstrate automatic gradient computation (`autograd`), mean squared error loss, and parameter optimization using Stochastic Gradient Descent (SGD).

### Complete Source Code
```python
import torch

torch.manual_seed(0)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = 2 * x + 1

model = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}  Loss = {loss.item():.4f}  dL/dw = {model.weight.grad.item():.4f}  dL/db = {model.bias.grad.item():.4f}")

print(f"\nLearned: w = {model.weight.item():.3f}, b = {model.bias.item():.3f}  (true: w = 2, b = 1)")

# python3 p11_linreg.py
```

### Compilation and Execution
```bash
python3 p11_linreg.py
```

### Detailed Explanation

1. **Synthetic Data Setup**:
   - $x \in \mathbb{R}^{5 \times 1}$ with values $[1.0, 2.0, 3.0, 4.0, 5.0]^T$.
   - Ground truth relationship: $y = 2x + 1$ (target weight $w = 2$, target bias $b = 1$).

2. **Model Definition (`nn.Linear(1, 1)`)**:
   - Implements affine transformation $\hat{y} = x \cdot w^T + b$.
   - Contains two trainable parameters: weight tensor $w$ and bias scalar $b$.

3. **Loss Function (`nn.MSELoss`)**:
   - Calculates Mean Squared Error:
     $$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$

4. **The Standard PyTorch Optimization Loop**:
   - `model(x)`: **Forward Pass** — computes predictions $\hat{y}$ and constructs the dynamic computational graph.
   - `optimizer.zero_grad()`: Clears out gradients from the previous iteration. PyTorch **accumulates** gradients by default on `.backward()` calls, so failing to reset gradients causes gradient explosion.
   - `loss.backward()`: **Backward Pass (Autograd)** — calculates the partial derivatives $\frac{\partial \mathcal{L}}{\partial w}$ and $\frac{\partial \mathcal{L}}{\partial b}$ using reverse-mode automatic differentiation.
   - `optimizer.step()`: Updates parameter values using SGD:
     $$w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}, \quad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}$$
     where $\eta = 0.01$ is the learning rate.

5. **Convergence**:
   - Over 1000 epochs, the loss drops from $\approx 50.1$ down to $0.0000$, and parameters converge to $w \approx 1.999$ and $b \approx 1.003$.

### Sample Output
```text
Epoch    0  Loss = 50.1284  dL/dw = -46.9461  dL/db = -12.9720
Epoch  100  Loss = 0.0007  dL/dw = -0.0057  dL/db = 0.0207
Epoch  200  Loss = 0.0003  dL/dw = -0.0041  dL/db = 0.0148
Epoch  300  Loss = 0.0002  dL/dw = -0.0029  dL/db = 0.0105
Epoch  400  Loss = 0.0001  dL/dw = -0.0021  dL/db = 0.0075
Epoch  500  Loss = 0.0000  dL/dw = -0.0015  dL/db = 0.0053
Epoch  600  Loss = 0.0000  dL/dw = -0.0011  dL/db = 0.0038
Epoch  700  Loss = 0.0000  dL/dw = -0.0008  dL/db = 0.0027
Epoch  800  Loss = 0.0000  dL/dw = -0.0005  dL/db = 0.0019
Epoch  900  Loss = 0.0000  dL/dw = -0.0004  dL/db = 0.0014

Learned: w = 1.999, b = 1.003  (true: w = 2, b = 1)
```

### Viva & Conceptual Questions
- **Q: Why does PyTorch accumulate gradients instead of overwriting them?**
  - Gradient accumulation facilitates training large models when mini-batch sizes exceed GPU memory limits: gradients can be accumulated across several micro-batches before calling `optimizer.step()`.
- **Q: What is the purpose of `loss.item()`?**
  - `loss.item()` extracts the Python floating-point scalar from a 1-element PyTorch tensor, detaching it from the computation graph and preventing memory buildup.

---

## Program 12: PyTorch Feedforward Neural Network (`p12_ffnn.py`)

### Objective
Develop a Multi-Layer Perceptron (Feedforward Neural Network) on random synthetic data using PyTorch with non-linear activation functions to demonstrate end-to-end network training.

### Complete Source Code
```python
import torch

torch.manual_seed(0)
X, y = torch.randn(100, 10), torch.randn(100, 1)

model = torch.nn.Sequential(
    torch.nn.Linear(10, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1)
)
print(model)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    loss = loss_fn(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}  Loss = {loss.item():.4f}")

print(f"Final Loss = {loss.item():.4f}")

# python3 p12_ffnn.py
```

### Compilation and Execution
```bash
python3 p12_ffnn.py
```

### Detailed Explanation

1. **Synthetic Multi-Dimensional Dataset**:
   - `X = torch.randn(100, 10)` generates 100 samples with 10 continuous features sampled from a standard normal distribution $\mathcal{N}(0, 1)$.
   - `y = torch.randn(100, 1)` generates 100 continuous regression target values.

2. **Architecture Definition (`nn.Sequential`)**:
   - **Input Layer to Hidden Layer**: `nn.Linear(10, 16)` applies $Z^{[1]} = X W_1 + b_1$, mapping 10 inputs to 16 hidden neurons ($10 \times 16 = 160$ weights $+ 16$ biases $= 176$ parameters).
   - **Activation Function**: `nn.ReLU()` applies element-wise non-linearity:
     $$\text{ReLU}(z) = \max(0, z)$$
     Without a non-linear activation function, stacking linear layers merely reduces to a single linear transformation regardless of depth.
   - **Hidden Layer to Output Layer**: `nn.Linear(16, 1)` applies $\hat{y} = A^{[1]} W_2 + b_2$, mapping 16 features to 1 scalar output ($16 \times 1 = 16$ weights $+ 1$ bias $= 17$ parameters).
   - **Total Model Parameters**: $176 + 17 = 193$ trainable parameters.

3. **Loss & Optimization**:
   - Loss: Mean Squared Error (`nn.MSELoss`).
   - Optimizer: Stochastic Gradient Descent (`lr=0.1`).
   - Each epoch evaluates the entire batch of 100 samples (Batch Gradient Descent), computes gradients across all 193 parameters via backpropagation, and updates the weights.
   - Loss decreases steadily from $0.7986$ to $0.3639$ across 100 epochs as the network fits the random representations.

### Sample Output
```text
Sequential(
  (0): Linear(in_features=10, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=1, bias=True)
)
Epoch   0  Loss = 0.7986
Epoch  10  Loss = 0.6657
Epoch  20  Loss = 0.6245
Epoch  30  Loss = 0.5862
Epoch  40  Loss = 0.5489
Epoch  50  Loss = 0.5133
Epoch  60  Loss = 0.4796
Epoch  70  Loss = 0.4472
Epoch  80  Loss = 0.4162
Epoch  90  Loss = 0.3875
Final Loss = 0.3639
```

### Viva & Conceptual Questions
- **Q: Why is non-linear activation (like ReLU) required in multi-layer neural networks?**
  - If activations were linear ($f(z) = z$), the composition of two linear layers $W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W' x + b'$ would just be another single linear layer, incapable of learning non-linear functions.
- **Q: What is the difference between batch gradient descent and stochastic gradient descent?**
  - Batch Gradient Descent computes the loss and gradients across the entire training dataset before each parameter update. Stochastic Gradient Descent (SGD) updates parameters using a single random training sample (or mini-batch) at a time.

---

## Comparative Architectural Summary

To prepare for oral evaluations and technical interviews, review how these frameworks compare:

```
                    ┌──────────────────────────────────────────────┐
                    │    PARALLEL PROGRAMMING PARADIGMS            │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌───────────────────┬─────────────┴───────┬──────────────────┐
         ▼                   ▼                     ▼                  ▼
   ┌───────────┐       ┌───────────┐         ┌───────────┐      ┌───────────┐
   │  OpenMP   │       │    MPI    │         │   CUDA    │      │  PyTorch  │
   ├───────────┤       ├───────────┤         ├───────────┤      ├───────────┤
   │Shared Mem │       │Dist. Mem  │         │Heterogen. │      │Tensors &  │
   │Multi-Core │       │Cluster/Net│         │GPU SIMT   │      │Autograd   │
   │Threads    │       │Processes  │         │Grids/Warps│      │Graphs     │
   └───────────┘       └───────────┘         └───────────┘      └───────────┘
```

| Feature | OpenMP | MPI | CUDA | PyTorch |
|---|---|---|---|---|
| **Execution Model** | Fork-Join Threads | SPMD (Single Program Multiple Data) Processes | SIMT (Single Instruction Multiple Threads) | Dynamic Computation Graph Dispatch |
| **Memory Space** | Single Shared Address Space | Private Memory per Process | Separate Host and Device VRAM | Device Tensors (Managed by PyTorch) |
| **Communication** | Direct memory reads/writes, cache coherence | Explicit Network Messages (`Send`/`Recv`/`Bcast`) | PCIe Bus (`cudaMemcpy`) / Shared Mem | High-level tensor operations / NCCL |
| **Hardware** | Multi-core CPU socket | Cluster nodes, LAN, Infiniband | NVIDIA GPUs | CPUs, NVIDIA GPUs, Apple Silicon, TPUs |
| **Synchronization**| `#pragma omp barrier`, `taskwait` | `MPI_Barrier`, Collective functions | `cudaDeviceSynchronize()`, `__syncthreads()` | Synchronous CUDA stream execution |
| **Fault Isolation**| Low (a segfault kills all threads) | High (isolated process memory spaces) | Low (GPU kernel trap halts context) | High (Python exceptions handled gracefully) |

---

## Quick Execution Cheatsheet

Save time during lab exams and demonstrations by keeping these compilation templates handy:

### 1. OpenMP Compilation (`gcc`)
```bash
# Compile with OpenMP flag
gcc -fopenmp <filename>.c -o <output_binary>

# Run binary
./<output_binary>
```

### 2. MPI Compilation & Execution (`mpicc` & `mpirun`)
```bash
# Compile MPI C program
mpicc <filename>.c -o <output_binary>

# Run on 4 processes
mpirun -np 4 ./<output_binary>

# Run oversubscribed (if CPU has fewer physical cores than requested)
mpirun --oversubscribe -np 4 ./<output_binary>
```

### 3. CUDA Compilation (`nvcc`)
```bash
# Compile CUDA program for active GPU architecture
nvcc <filename>.cu -o <output_binary>

# Run compiled GPU binary
./<output_binary>
```

### 4. PyTorch Execution (`python3`)
```bash
python3 <filename>.py
```
