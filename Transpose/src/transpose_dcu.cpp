#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <hip/hip_runtime.h>
#define TILE_DIM 64
#define BLOCK_ROWS 32
#define HIP_CHECK(cmd) do {                                  \
  hipError_t e = (cmd);                                      \
  if (e != hipSuccess) {                                     \
    std::cerr << "HIP Error: " << hipGetErrorString(e)       \
              << " at " << __FILE__ << ":" << __LINE__       \
              << std::endl;                                  \
    exit(EXIT_FAILURE);                                      \
  }                                                          \
} while(0)
using namespace std;

// 生成随机矩阵
vector<int8_t> generateMatrix(int n) {
    vector<int8_t> matrix(n * n);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int8_t> dist(-128, 127);
    for (int i = 0; i < n * n; ++i)
        matrix[i] = dist(gen);
    return matrix;
}

// 串行矩阵转置
vector<int8_t> serialTranspose(const vector<int8_t>& matrix, int n) {
    vector<int8_t> result(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            result[j * n + i] = matrix[i * n + j];
    return result;
}

// 棋盘划分并行转置
vector<int8_t> checkerboardTranspose(const vector<int8_t>& matrix, int n, int p) {
    vector<int8_t> result(n * n);
    int block_size = (n + p - 1) / p;

    #pragma omp parallel for collapse(2) num_threads(p * p)
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            int start_x = i * block_size, end_x = min((i + 1) * block_size, n);
            int start_y = j * block_size, end_y = min((j + 1) * block_size, n);
            for (int x = start_x; x < end_x; ++x)
                for (int y = start_y; y < end_y; ++y)
                    result[y * n + x] = matrix[x * n + y];
        }
    }
    return result;
}

// 直角划分并行转置
vector<int8_t> rectangularTranspose(const vector<int8_t>& matrix, int n, int p) {
    vector<int8_t> result(n * n);
    int rows_per_thread = (n + p - 1) / p;

    #pragma omp parallel for num_threads(p)
    for (int i = 0; i < p; ++i) {
        int start_row = i * rows_per_thread;
        int end_row = min((i + 1) * rows_per_thread, n);
        for (int x = start_row; x < end_row; ++x)
            for (int y = 0; y < n; ++y)
                result[y * n + x] = matrix[x * n + y];
    }
    return result;
}

__global__ void gpuTransposeKernel(const int8_t* __restrict__ input,
                        int8_t* __restrict__ output,
                        int n)      // 一个block转置64*64的小块
{
    // 使用带 +1 Padding 的共享内存避免 Bank Conflict
    __shared__ int8_t tile[TILE_DIM][TILE_DIM + 1];

    // 全局坐标 (x, y) 对应当前线程操作的输入矩阵的列、行
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 以 BLOCK_ROWS 为步长，把一个 TILE_DIM×TILE_DIM 区块分多次从全局读入共享内存
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * n + x];
        }
    }
    __syncthreads();

    // 写回时，将 blockIdx 交换（实现转置），当前线程将其在共享内存中取数时转置位置的值存储到全局内存中对应转置tile的同一位置
    int x_t = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_t = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x_t < n && (y_t + j) < n) {
            output[(y_t + j) * n + x_t] = tile[threadIdx.x][threadIdx.y + j];   // 读取共享内存时转置
        }
    }
}

std::vector<int8_t> hipTranspose(const std::vector<int8_t>& matrix, int n) {
    size_t size = size_t(n) * n * sizeof(int8_t);
    int8_t *d_input = nullptr, *d_output = nullptr;

    HIP_CHECK(hipMalloc(&d_input,  size));
    HIP_CHECK(hipMalloc(&d_output, size));
    HIP_CHECK(hipMemcpy(d_input, matrix.data(), size, hipMemcpyHostToDevice));

    dim3 blockSize(TILE_DIM, BLOCK_ROWS);
    dim3 gridSize((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);
    gpuTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    // HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int8_t> result(n * n);
    HIP_CHECK(hipMemcpy(result.data(), d_output, size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    return result;
}

int main() {
    cout << "矩阵转置算法对比实验\n";
    int n = 25000;
    // cout << "输入矩阵维度 n (n > 0): ";
    // cin >> n;

    auto matrix = generateMatrix(n);
    int max_threads = omp_get_max_threads();
    int p = 20;
    cout << "请输入线程数 (1 - " << max_threads << "): ";
    // cin >> p;

    auto start = chrono::high_resolution_clock::now();
    auto serial_result = serialTranspose(matrix, n);
    auto serial_time = chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();
    cout << "串行 CPU 耗时: " << serial_time << " 秒\n";

    start = chrono::high_resolution_clock::now();
    auto checkerboard_result = checkerboardTranspose(matrix, n, p);
    auto checkerboard_time = chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();
    cout << "棋盘划分并行 CPU 耗时: " << checkerboard_time << " 秒\n";

    start = chrono::high_resolution_clock::now();
    auto rectangular_result = rectangularTranspose(matrix, n, p);
    auto rectangular_time = chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();
    cout << "直角划分并行 CPU 耗时: " << rectangular_time << " 秒\n";

    start = chrono::high_resolution_clock::now();
    auto hip_result = hipTranspose(matrix, n);
    auto hip_time = chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();
    cout << "HIP GPU 耗时: " << hip_time << " 秒\n";

    cout << "\n性能对比:\n";
    cout << "算法\t\t\t时间(秒)\t\t加速比\n";
    cout << "串行 CPU\t\t" << serial_time << "\t1.0\n";
    cout << "棋盘划分并行 CPU\t" << checkerboard_time << "\t" << serial_time / checkerboard_time << "\n";
    cout << "直角划分并行 CPU\t" << rectangular_time << "\t" << serial_time / rectangular_time << "\n";
    cout << "HIP GPU\t\t\t" << hip_time << "\t" << serial_time / hip_time << "\n";

    return 0;
}