module purge
module load compiler/devtoolset/7.3.1    # （如果需要 GNU 工具链）
module load compiler/dtk/24.04          # DTK 24.04 环境

hipcc ./src/spmm_dcu.cpp -o ./src/spmm_dcu -lrocsparse

# echo "Build succeeded"