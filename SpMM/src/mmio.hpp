#ifndef MM_IO_HPP
#define MM_IO_HPP

#include <cstdio>
#include <cstring>
#include <cctype>

#define MM_MAX_LINE_LENGTH 1025
#define MM_MAX_TOKEN_LENGTH 64
#define MatrixMarketBanner "%%MatrixMarket"

typedef char MM_typecode[4];

// typecode
#define mm_clear_typecode(typecode) \
    ((*typecode)[0]=(*typecode)[1]=(*typecode)[2]=' ',(*typecode)[3]='G')
#define mm_set_matrix(typecode)      ((*typecode)[0]='M')
#define mm_set_sparse(typecode)      ((*typecode)[1]='C')
#define mm_set_dense(typecode)       ((*typecode)[1]='A')
#define mm_set_real(typecode)        ((*typecode)[2]='R')
#define mm_set_complex(typecode)     ((*typecode)[2]='C')
#define mm_set_pattern(typecode)     ((*typecode)[2]='P')
#define mm_set_integer(typecode)     ((*typecode)[2]='I')
#define mm_set_general(typecode)     ((*typecode)[3]='G')
#define mm_set_symmetric(typecode)   ((*typecode)[3]='S')
#define mm_set_hermitian(typecode)   ((*typecode)[3]='H')
#define mm_set_skew(typecode)        ((*typecode)[3]='K')

// typecode
#define mm_is_integer(typecode) ((typecode)[2]=='I')
#define mm_is_real(typecode)    ((typecode)[2]=='R')
#define mm_is_pattern(typecode) ((typecode)[2]=='P')
#define mm_is_symmetric(typecode)((typecode)[3]=='S')

// 读取 Matrix Market banner
int mm_read_banner(FILE *f, MM_typecode *matcode)
{
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH];
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;

    mm_clear_typecode(matcode);

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
        return -1;  // 提前 EOF

    if (sscanf(line, "%s %s %s %s %s",
               banner, mtx, crd, data_type, storage_scheme) != 5)
        return -1;

    /* 转小写 */
    for (p=mtx;           *p; ++p) *p = std::tolower(*p);
    for (p=crd;           *p; ++p) *p = std::tolower(*p);
    for (p=data_type;     *p; ++p) *p = std::tolower(*p);
    for (p=storage_scheme;*p; ++p) *p = std::tolower(*p);

    /* 检查 banner */
    if (std::strncmp(banner, MatrixMarketBanner, std::strlen(MatrixMarketBanner)) != 0)
        return -1;
    mm_set_matrix(matcode);

    /* 稀疏 vs 密集 */
    if (std::strcmp(crd, "coordinate") == 0)
        mm_set_sparse(matcode);
    else if (std::strcmp(crd, "array") == 0)
        mm_set_dense(matcode);
    else
        return -1;

    /* 数据类型 */
    if (std::strcmp(data_type, "real") == 0)
        mm_set_real(matcode);
    else if (std::strcmp(data_type, "complex") == 0)
        mm_set_complex(matcode);
    else if (std::strcmp(data_type, "pattern") == 0)
        mm_set_pattern(matcode);
    else if (std::strcmp(data_type, "integer") == 0)
        mm_set_integer(matcode);
    else
        return -1;

    /* 存储方案 */
    if (std::strcmp(storage_scheme, "general") == 0)
        mm_set_general(matcode);
    else if (std::strcmp(storage_scheme, "symmetric") == 0)
        mm_set_symmetric(matcode);
    else if (std::strcmp(storage_scheme, "hermitian") == 0)
        mm_set_hermitian(matcode);
    else if (std::strcmp(storage_scheme, "skew-symmetric") == 0 ||
             std::strcmp(storage_scheme, "skew") == 0)
        mm_set_skew(matcode);
    else
        return -1;

    return 0;
}

// 读取 coordinate 格式时，跳过注释并解析行数、列数、非零元个数
int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz)
{
    char line[MM_MAX_LINE_LENGTH];
    *M = *N = *nz = 0;

    /* 跳过注释行 */
    do {
        if (std::fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
            return -1;
    } while (line[0] == '%');

    /* 尝试解析 */
    if (std::sscanf(line, "%d %d %d", M, N, nz) == 3)
        return 0;

    /* 如果第一行为空或不完整，继续用 fscanf */
    while (std::fscanf(f, "%d %d %d", M, N, nz) != 3) {
        if (std::feof(f)) return -1;
    }
    return 0;
}

#endif  // MM_IO_HPP
