#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <hip/hip_runtime.h>
#include <vector>
#include <rocsparse/rocsparse.h>
#include <sys/time.h>
#include "./util.hpp"
#define WARP_SIZE 64

#define HIP_CHECK(cmd) do {                                  \
  hipError_t e = (cmd);                                      \
  if (e != hipSuccess) {                                     \
    std::cerr << "HIP Error: " << hipGetErrorString(e)       \
              << " at " << __FILE__ << ":" << __LINE__       \
              << std::endl;                                  \
    exit(EXIT_FAILURE);                                      \
  }                                                          \
} while(0)

// const int WARP_SIZE = 64; // DCU warp大小

inline double seconds()
{
    struct timeval tp;
    int i = gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void checkResult1(const float *dcuRef0, const float *dcuRef1, int A_nrows, int B_ncols) {       //与rocsparse库对比
    int errors = 0;
    for (int i = 0; i < A_nrows; i++) {
        for (int j = 0; j < B_ncols; j++) {
            float diff = fabs(dcuRef0[i + j * A_nrows] - dcuRef1[i * B_ncols + j]);
            if (diff > 1e-6) {
                if (errors < 10) {
                    printf("Error at [%d,%d]: %.4f vs %.4f (diff=%.4f)\n", 
                           i, j, dcuRef0[i + A_nrows * j], dcuRef1[i * B_ncols + j], diff);
                }
                errors++;
            }
        }
    }
    printf("completed with %d errors\n", errors);
}

void checkResult2(const float *dcuRef0, const float *dcuRef1, int A_nrows, int B_ncols) {       //与DCU基础版SpMM对比
    int errors = 0;
    for (int i = 0; i < A_nrows; i++) {
        for (int j = 0; j < B_ncols; j++) {
            float diff = fabs(dcuRef0[i * B_ncols + j] - dcuRef1[i * B_ncols + j]);
            if (diff > 1e-6) {
                if (errors < 10) {
                    printf("Error at [%d,%d]: %.4f vs %.4f (diff=%.4f)\n", 
                           i, j, dcuRef0[i * B_ncols + j], dcuRef1[i * B_ncols + j], diff);
                }
                errors++;
            }
        }
    }
    printf("completed with %d errors\n", errors);
}

// SpMM在CPU实现
template<typename T>
void spmm_cpu(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, T* A_csrVal,
    T* B_dnVal, T* C_dnVal
) {
    T acc = 0;
    for (int i=0; i<A_nrows; i++) {
        for (int k=0; k < B_ncols; k++) {
            acc = 0;
            for (int ptr=A_csrRowPtr[i]; ptr<A_csrRowPtr[i+1]; ptr++) {
                acc += A_csrVal[ptr] * B_dnVal[(B_ncols * A_csrColInd[ptr] + k)];
            }
            C_dnVal[(B_ncols * i + k)] = acc;
        }
    }
}

template<typename T>
void spmm_rocsparse(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, T* A_csrVal,
    T* B_dnVal, T* C_dnVal, int nnz
) {
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);
    rocsparse_mat_descr descr;
    rocsparse_create_mat_descr(&descr);
    rocsparse_set_mat_index_base(descr, rocsparse_index_base_zero);
    rocsparse_set_mat_type(descr, rocsparse_matrix_type_general);
    // T one=1.0f, zero=0.0f;
    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;
    constexpr rocsparse_operation trans_A = rocsparse_operation_none;
    constexpr rocsparse_operation trans_B = rocsparse_operation_transpose;
    rocsparse_status status = rocsparse_scsrmm(handle,
                    trans_A,   // A不转置
                    trans_B,
                    A_nrows,
                    B_ncols,
                    A_nrows,    // A的行值、列值与B的行值相等
                    nnz,
                    &alpha,
                    descr,
                    A_csrVal,
                    A_csrRowPtr,
                    A_csrColInd,
                    B_dnVal,
                    B_ncols,
                    &beta,
                    C_dnVal,
                    A_nrows);
    std::cout << "status:" << status << std::endl;
    rocsparse_destroy_mat_descr(descr);
    rocsparse_destroy_handle(handle);
}

// 基本SpMM在DCU上实现
template<typename T>
__global__ void spmm_dcu0(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, T* A_csrVal,
    T* B_dnVal, T* C_dnVal
) {
    int rid = blockDim.y * blockIdx.x + threadIdx.y;
    if (rid < A_nrows) {
        int cid = (blockIdx.y * WARP_SIZE) + threadIdx.x;
        int lb = A_csrRowPtr[rid];
        int hb = A_csrRowPtr[rid + 1];
        T acc = 0;
        
        for (int ptr = lb; ptr < hb; ptr++) {
            int offset = A_csrColInd[ptr] * B_ncols + cid;
            acc += A_csrVal[ptr] * B_dnVal[offset];
        }
        
        if (cid < B_ncols) {
            C_dnVal[rid * B_ncols + cid] = acc;
        }
    }
}

// 合并行缓存
template<typename T>
__global__ void spmm_dcu1(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, T* A_csrVal,
    T* B_dnVal, T* C_dnVal
) {
    extern __shared__ int sh[];
    int* colInd_sh = sh;
    T* val_sh = (T*)&sh[blockDim.y * WARP_SIZE];
    int shmem_offset = threadIdx.y * WARP_SIZE;
    int thread_idx = shmem_offset + threadIdx.x;

    int rid = blockDim.y * blockIdx.x + threadIdx.y;
    
    if (rid < A_nrows) {
        int cid = (blockIdx.y * WARP_SIZE) + threadIdx.x;
        int lb = A_csrRowPtr[rid];
        int hb = A_csrRowPtr[rid + 1];
        int ptr = lb + threadIdx.x;
        T acc = 0;

        for (int jj = lb; jj < hb; jj += WARP_SIZE) {
            if (ptr < hb) {
                val_sh[thread_idx] = A_csrVal[ptr];
                colInd_sh[thread_idx] = A_csrColInd[ptr]; // 仅存储列索引
            }
            __syncthreads(); // DCU需要完整的线程块同步

            for (int kk = 0; kk < WARP_SIZE && jj + kk < hb; kk++) {
                int col_idx = colInd_sh[shmem_offset + kk];
                int offset = col_idx * B_ncols + cid;
                acc += val_sh[shmem_offset + kk] * B_dnVal[offset];
            }
            __syncthreads();
            ptr += WARP_SIZE;
        }
        
        if (cid < B_ncols) {
            C_dnVal[rid * B_ncols + cid] = acc;
        }
    }
}

// 线程束融合
template<typename T>
__global__ void spmm_dcu2(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, T* A_csrVal,
    T* B_dnVal, T* C_dnVal
) {
    extern __shared__ int sh[];
    // 新布局：[WARP_SIZE][tile_row] 避免步长1访问
    int* colInd_sh = sh;
    T* val_sh = (T*)&sh[WARP_SIZE * blockDim.y];

    int rid = blockDim.y * blockIdx.x + threadIdx.y;
    if (rid >= A_nrows) return;

    int cid = blockIdx.y * 128 + threadIdx.x; // 每线程处理128列中的1列
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[rid + 1];
    T acc[2] = {0, 0}; // 每个线程计算2个输出

    for (int jj = lb; jj < hb; jj += WARP_SIZE) {
        // 协作加载：线程kk加载元素(jj+kk)
        int kk = threadIdx.x;
        int ptr = jj + kk;
        int col_idx = (ptr < hb) ? A_csrColInd[ptr] : 0;
        T val = (ptr < hb) ? A_csrVal[ptr] : 0.0f;

        // 转置存储：colInd_sh[kk][threadIdx.y]
        colInd_sh[kk * blockDim.y + threadIdx.y] = col_idx;
        val_sh[kk * blockDim.y + threadIdx.y] = val;
        __syncthreads();

        // 处理当前块的非零元素
        for (int i = 0; i < min(WARP_SIZE, hb - jj); i++) {
            int idx = colInd_sh[i * blockDim.y + threadIdx.y]; // 无bank冲突
            T v = val_sh[i * blockDim.y + threadIdx.y];
            int offset = idx * B_ncols + cid;
            acc[0] += v * B_dnVal[offset];
            acc[1] += v * B_dnVal[offset + 64];
        }
        __syncthreads();
    }

    if (cid < B_ncols) 
        C_dnVal[rid * B_ncols + cid] = acc[0];
    if (cid + 64 < B_ncols) 
        C_dnVal[rid * B_ncols + cid + 64] = acc[1];
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_file>\n", argv[0]);
        return 1;
    }
    int A_nrows, A_ncols, nnz;
    std::vector<int> row_indices, col_indices;
    std::vector<float> values;
    double start1, start2, end1, end2;

    // 读取矩阵文件
    readMtx<float>(argv[1], row_indices, col_indices, values, A_nrows, A_ncols, nnz);
    printf("Matrix: %d x %d, nnz: %d\n", A_nrows, A_ncols, nnz);

    // 主机内存分配
    int max_ncols = 128;
    int* A_indptr = new int[A_nrows + 1];   // 行指针
    int* A_indices = new int[nnz];          // 列索引
    float* A_data = new float[nnz];         // 非零值
    const int B_ncols = max_ncols;          // 固定B的列数
    float* B = new float[B_ncols * A_ncols];// 输入稠密矩阵
    float* C = new float[A_nrows * B_ncols];// 输出稠密矩阵
    float* dcuRef0 = new float[A_nrows * B_ncols];
    float* dcuRef1 = new float[A_nrows * B_ncols];
    
    // 转换为CSR格式
    memset(A_indptr, 0, (A_nrows + 1) * sizeof(int));
    for (int n = 0; n < nnz; n++) {
        A_indptr[row_indices[n] + 1]++;
    }
    for (int n = 1; n <= A_nrows; n++) {
        A_indptr[n] += A_indptr[n - 1];
    }
    for (int n = 0; n < nnz; n++) {
        int ptr = A_indptr[row_indices[n]];
        A_indices[ptr] = col_indices[n];
        A_data[ptr] = 1.0f;  // 简化数据
        A_indptr[row_indices[n]] = ptr + 1;
    }
    for (int n = A_nrows; n > 0; n--) {
        A_indptr[n] = A_indptr[n - 1];
    }
    A_indptr[0] = 0;
    // 初始化B矩阵
    srand(time(0));
    for (int i = 0; i < B_ncols * A_ncols; i++) {
        B[i] = static_cast<float>(rand() % 100 - 50) / 100.0f;
    }

    // 设备内存分配
    int *A_indptr_dev, *A_indices_dev;
    float *A_data_dev, *B_dev, *C_dev0, *C_dev1;
    HIP_CHECK(hipMalloc(&A_indptr_dev, (A_nrows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&A_indices_dev, nnz * sizeof(int)));
    HIP_CHECK(hipMalloc(&A_data_dev, nnz * sizeof(float)));
    HIP_CHECK(hipMalloc(&B_dev, B_ncols * A_ncols * sizeof(float)));
    HIP_CHECK(hipMalloc(&C_dev0, A_nrows * B_ncols * sizeof(float)));
    HIP_CHECK(hipMalloc(&C_dev1, A_nrows * B_ncols * sizeof(float)));
    // 数据拷贝到设备
    HIP_CHECK(hipMemcpy(A_indptr_dev, A_indptr, (A_nrows + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(A_indices_dev, A_indices, nnz * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(A_data_dev, A_data, nnz * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B, B_ncols * A_ncols * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev0, 0, A_nrows * B_ncols * sizeof(float)));
    HIP_CHECK(hipMemset(C_dev1, 0, A_nrows * B_ncols * sizeof(float)));

    // 执行配置
    const int tile_row = 8; // 每个block处理的行数
    dim3 grid_dim((A_nrows + tile_row - 1) / tile_row, (B_ncols + WARP_SIZE - 1) / WARP_SIZE);
    dim3 grid_dim2((A_nrows + tile_row - 1) / tile_row, (B_ncols + 2 * WARP_SIZE - 1) / (2 * WARP_SIZE));
    dim3 block_dim(WARP_SIZE, tile_row); // 每个warp64线程，8个warp
    // 共享内存大小
    size_t shared_mem = block_dim.y * WARP_SIZE * (sizeof(int) + sizeof(float));

    int errors = 0;

    // CPU计算
    // start1 = seconds();
    // spmm_cpu<float>(A_nrows, B_ncols, A_indptr, A_indices, A_data, B, C);
    // end1 = seconds();

    // 基准
    start1 = seconds();
    spmm_rocsparse(A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev0, nnz);
    // spmm_dcu0<float><<<grid_dim, block_dim>>>(A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev0);
    HIP_CHECK(hipDeviceSynchronize());
    end1 = seconds();

    // DCU优化后
    start2 = seconds();
    // spmm_dcu0<float><<<grid_dim, block_dim>>>(A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev1);
    // spmm_dcu1<float><<<grid_dim, block_dim, shared_mem>>>(A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev1);
    spmm_dcu2<float><<<grid_dim2, block_dim, shared_mem>>>(A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev1);
    HIP_CHECK(hipDeviceSynchronize());
    end2 = seconds();

    // 验证结果
    HIP_CHECK(hipMemcpy(dcuRef0, C_dev0, A_nrows * B_ncols * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(dcuRef1, C_dev1, A_nrows * B_ncols * sizeof(float), hipMemcpyDeviceToHost));
    // checkResult1(dcuRef0, dcuRef1, A_nrows, B_ncols);
    checkResult1(dcuRef0, dcuRef1, A_nrows, B_ncols);

    // 输出结果
    double S = (end1 - start1) / (end2 - start2);
    printf("优化前:%lf s\n", end1 - start1);
    printf("优化后:%lf s\n", end2 - start2);
    printf("加速比:%lf\n", S);

    // 清理资源
    delete[] A_indptr;
    delete[] A_indices;
    delete[] A_data;
    delete[] B;
    delete[] C;
    delete[] dcuRef0;
    delete[] dcuRef1;
    HIP_CHECK(hipFree(A_indptr_dev));
    HIP_CHECK(hipFree(A_indices_dev));
    HIP_CHECK(hipFree(A_data_dev));
    HIP_CHECK(hipFree(B_dev));
    HIP_CHECK(hipFree(C_dev0));
    HIP_CHECK(hipFree(C_dev1));

    return 0;
}