#!/bin/bash
#SBATCH -o ./out/%j.out
#SBATCH -J test_spmm
#SBATCH -p wzidnormal
#SBATCH --gres=dcu:1		#指定每个节点使用1块DCU卡
#SBATCH -w xdb5

module purge
module load compiler/devtoolset/7.3.1
module load compiler/dtk/24.04

./src/spmm_dcu ./data/cora.mtx
# ./src/spmm_dcu ./data/pdb1HYS.mtx
# ./src/spmm_dcu ./data/cant.mtx
# ./src/spmm_dcu ./data/mip1.mtx
# ./src/spmm_dcu ./data/consph.mtx
# ./src/spmm_dcu ./data/dc2.mtx
# ./src/spmm_dcu ./data/cop20k_A.mtx
# ./src/spmm_dcu ./data/shipsec1.mtx