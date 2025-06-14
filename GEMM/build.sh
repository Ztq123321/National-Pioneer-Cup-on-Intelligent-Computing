#!/bin/bash

module purge
module load compiler/devtoolset/7.3.1
module load compiler/dtk/24.04



hipcc ./src/Gemm_dcu_float.cpp -o ./src/Gemm_dcu_float -O3 -mfma --offload-arch=gfx908 -ffast-math


echo "Build succeeded"
