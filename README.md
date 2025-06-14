## Research on General Matrix Multiplication

## **Overview**

This project is qualified for participation in the "National Pioneer Cup on Intelligent Computing – Shandong University of Science and Technology Selection. The primary focus is optimizing matrix-related operations on DCU, specifically including the following three tasks:
1. **General Matrix Multiplication (GEMM) Optimization**: Implement parallelized GEMM on DCU and apply performance optimization theories and methods such as memory access optimization, tiling techniques, register optimization, loop unrolling, etc., to achieve efficient matrix multiplication. Verify the accuracy of computation results and output runtime and speedup ratios.
2. **Sparse Matrix Optimization**: Optimize SpMM on the DCU platform, including sparse scheduling strategies and parallelization strategies.
3. **Matrix Transposition**: Perform and optimize matrix transposition on the DCU platform.

## **Runtime Environment**

1. **Hardware Environment**:
CPU: Intel 8458P, with 20 available CPU cores.
One DCU heterogeneous accelerator (k100AI) with 64GB of VRAM.
2. **Software Environment**:
hip5.4.23416, devtoolset-7.3.1, dtk-24.04, rocSPARSE library.

## **Execution Steps**

1.  Modify the compilation script `build.sh` and the job submission script `dcutest.slurm`.
2.  Execute the command `bash build.sh` to compile.
3.  Execute the command `sbatch dcutest.slurm` to submit the job.
4.  Check the generated `.out` file in the `out` directory to view the runtime results.
## 
The SpMM algorithm in the code draws inspiration from the paper _"GE-SpMM: General-Purpose Sparse Matrix-Matrix Multiplication on GPUs for Graph Neural Networks"_ and its associated code（[https://github.com/hgyhungry/ge-spmm](https://github.com/hgyhungry/ge-spmm)），with modifications applied。
