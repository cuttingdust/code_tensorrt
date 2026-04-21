#include "utils.hpp"

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void MatmulKernel(float *M_device, float *N_device, float *P_device, int width)
{
    /// 我们设定每一个thread负责P中的一个坐标的matmul
    /// 所以一共有width * width个thread并行处理P的计算

    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    float P_element = 0;
    /// 对于每一个P的元素，我们只需要循环遍历width次M和N中的元素就可以了
    for (int k = 0; k < width; k++)
    {
        float M_element = M_device[y * width + k];
        float N_element = N_device[k * width + x];
        P_element += M_element * N_element;
    }

    P_device[y * width + x] = P_element;
}


/// CUDA中使用block对矩阵中某一片区域进行集中计算。这个类似于loop中的tile
/// 感兴趣的同学可以试着改一下blockSize，也就是tileSize，看看速度会发生什么样子的变化
/// 当blockSize达到一个数量的时候，这个程序会出错。下一个案例中我们会分析

/*
    这个实现的问题点：只有一个block
    因为只有一个block，并且又因为SM中的sp数量是有限的，所以不能够全部放下。想要全部放下的话需要缩小矩阵的大小
    有很多次读写，但具体的执行很少(两次读和一次写，一次计算)
    解决办法：使用tile
*/
void MatmulOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize)
{
    ///  设置矩阵大小
    int size = width * width * sizeof(float);

    /// 分配M, N在GPU上的空间
    float *M_device;
    float *N_device;

    CUDA_CHECK(cudaMalloc(&M_device, size));
    CUDA_CHECK(cudaMalloc(&N_device, size));

    /// 分配M, N拷贝到GPU上
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    /// 分配P在GPU上的空间
    float *P_device;
    CUDA_CHECK(cudaMalloc(&P_device, size));

    ///  调用kernel来进行matmul计算, 在这个例子中我们用的方案是：将一个矩阵切分成多个blockSize * blockSize的大小
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);
    MatmulKernel<<<dimGrid, dimBlock>>>(M_device, N_device, P_device, width);


    ///  将结果从device拷贝回host
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /// 注意要在synchronization结束之后排查kernel的错误, 否则错误排查只会检查参数配置
    LAST_KERNEL_CHECK();

    /// Free
    CUDA_CHECK(cudaFree(P_device));
    CUDA_CHECK(cudaFree(N_device));
    CUDA_CHECK(cudaFree(M_device));
}
