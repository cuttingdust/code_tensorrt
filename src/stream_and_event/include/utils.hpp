#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cstdarg>
#include <cuda_runtime.h>
#include <system_error>

#define CUDA_CHECK(call)    __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK() __kernelCheck(__FILE__, __LINE__)
#define LOG(...)            __log_info(__VA_ARGS__)
#define BLOCKSIZE           16

inline static void __cudaCheck(cudaError_t err, const char* file, const int line)
{
    if (err != cudaSuccess)
    {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

inline static void __kernelCheck(const char* file, const int line)
{
    /// 在编写CUDA是，错误排查非常重要，默认的cuda runtime API中的函数都会返回cudaError_t类型的结果，
    /// 但是在写kernel函数的时候，需要通过cudaPeekAtLastError或者cudaGetLastError来获取错误
    /// 这两个函数的区别在于，前者不会清除错误状态，而后者会清除错误状态，因此在检查kernel函数的错误时，建议使用cudaPeekAtLastError

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

static void __log_info(const char* format, ...)
{
    char    msg[1000];
    va_list args;
    va_start(args, format);

    vsnprintf(msg, sizeof(msg), format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}

void initMatrix(float* data, int size, int low, int high, int seed);
void initMatrixSigned(float* data, int size, float low, float high, int seed);
void printMat(float* data, int size);
void compareMat(float* h_data, float* d_data, int size);

#endif //__UTILS__HPP__
