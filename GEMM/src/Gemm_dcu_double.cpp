#include <stdio.h>
#include <iostream>
#include <cmath>
#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define data_type double
#define TILE_SIZE 16

#define HIP_CHECK(cmd) do {                                  \
    hipError_t e = (cmd);                                    \
    if (e != hipSuccess) {                                   \
        std::cerr << "HIP Error: " << hipGetErrorString(e) \
                  << " at " << __FILE__ << ":" << __LINE__ \
                  << std::endl;                              \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while(0)

inline double seconds() {
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return tp.tv_sec + tp.tv_nsec * 1.0e-9;
}

bool checkResult(const data_type* hostRef, const data_type* dcuRef, const int N) {
    const data_type epsilon = 1.0E-1;//再进行规模为1024 1024 1024的情况下精度可以达到1.0E-3，2048 2048 2048的情况下精度可以达到1.0E-2
    for (int i = 0; i < N; i++) {
        if (fabs(hostRef[i] - dcuRef[i]) > epsilon) {
            printf("Mismatch at %d: host %f dcu %f\n", i, hostRef[i], dcuRef[i]);
            return false;
        }
    }
    return true;
}

void MatrixMulOnHost(const data_type* A, const data_type* B, data_type* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            data_type sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void MatrixMul_Shared16x16(const data_type* A, const data_type* B, data_type* C, int M, int N, int K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ data_type sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ data_type sharedB[TILE_SIZE][TILE_SIZE+1];
    data_type value = 0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;

        sharedA[threadIdx.y][threadIdx.x] = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0;
        sharedB[threadIdx.y][threadIdx.x] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

int main() {
    int t=1;
    std::cout << "请输入测试组数 t: ";
    std::cin >> t;
    std::cout << t << std::endl;
   
    for (int test = 0; test < t; ++test) {
        int M, N, K;
        std::cout << "请输入第 " << test + 1 << " 组的 M N K：";
        std::cin >> M >> N >> K;
        std::cout << M <<" "<< N <<" "<< K << std::endl;
        size_t sizeA = M * K * sizeof(data_type);
        size_t sizeB = K * N * sizeof(data_type);
        size_t sizeC = M * N * sizeof(data_type);

        // allocate host
        data_type *hA = (data_type*)malloc(sizeA);
        data_type *hB = (data_type*)malloc(sizeB);
        data_type *hC = (data_type*)malloc(sizeC);
        data_type *hC_dcu = (data_type*)malloc(sizeC);

        srand((unsigned)time(nullptr));
        for (int i = 0; i < M * K; ++i) hA[i] = rand() / (double)RAND_MAX;
        for (int i = 0; i < K * N; ++i) hB[i] = rand() / (double)RAND_MAX;

        double t_host = seconds();
        MatrixMulOnHost(hA, hB, hC, M, N, K);
        t_host = seconds() - t_host;
        printf("Host time: %f s\n", t_host);

        // allocate device
        data_type *dA, *dB, *dC;
        HIP_CHECK(hipMalloc(&dA, sizeA));
        HIP_CHECK(hipMalloc(&dB, sizeB));
        HIP_CHECK(hipMalloc(&dC, sizeC));
        HIP_CHECK(hipMemcpy(dA, hA, sizeA, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, hB, sizeB, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(dC, 0, sizeC));

        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);

        double t_dcu = seconds();
        hipLaunchKernelGGL(MatrixMul_Shared16x16, grid, block, 0, 0, dA, dB, dC, M, N, K);
        HIP_CHECK(hipDeviceSynchronize());
        t_dcu = seconds() - t_dcu;

        printf("DCU time: %f s\n", t_dcu);

        HIP_CHECK(hipMemcpy(hC_dcu, dC, sizeC, hipMemcpyDeviceToHost));

        if (!checkResult(hC, hC_dcu, M * N)) {
            printf("计算错误\n");
        } else {
            printf("Speedup: %f\n", t_host / t_dcu);
        }

        // cleanup
        hipFree(dA);
        hipFree(dB);
        hipFree(dC);
        free(hA); free(hB); free(hC); free(hC_dcu);
    }

    return 0;
}
