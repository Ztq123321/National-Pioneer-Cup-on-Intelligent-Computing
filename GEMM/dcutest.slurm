#!/bin/bash
#SBATCH -o ./out/%j.out
#SBATCH -J test_gemm
#SBATCH -p wzidnormal
#SBATCH --gres=dcu:1		#指定每个节点使用1块DCU卡
#SBATCH -w xdb4

module purge
module load compiler/devtoolset/7.3.1
module load compiler/dtk/24.04

./src/Gemm_dcu_float < input.txt