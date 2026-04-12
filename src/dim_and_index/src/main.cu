#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// 辅助注释：CUDA 同步函数说明
// ============================================================================
/// cudaDeviceSynchronize: CPU 与 GPU 同步，CPU 等待 GPU 上所有操作完成
/// cudaStreamSynchronize: 针对特定流的同步，只等待该流中的操作完成
/// cudaThreadSynchronize: 已弃用的方法，不要使用
/// __syncthreads:         线程块内同步，同一个 block 中的所有线程等待彼此

// ============================================================================
// 内核函数 1: print_idx_kernel
// 功能：打印每个线程的 block 索引和 thread 索引（三维形式）
// 用途：理解 blockIdx 和 threadIdx 的三维分布
// ============================================================================
__global__ void print_idx_kernel()
{
    // blockIdx.z, blockIdx.y, blockIdx.x: 当前线程所在 block 的三维索引
    // threadIdx.z, threadIdx.y, threadIdx.x: 当前线程在 block 内的三维索引
    printf("block idx: (%3d, %3d, %3d), thread idx: (%3d, %3d, %3d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
           threadIdx.y, threadIdx.x);
}

// ============================================================================
// 内核函数 2: print_dim_kernel
// 功能：打印整个任务的规模（总共有多少 block，每个 block 有多少 thread）
// 用途：理解 gridDim 和 blockDim 的含义
// ============================================================================
__global__ void print_dim_kernel()
{
    // blockDim.z, blockDim.y, blockDim.x: 每个 block 在三个维度上的线程数
    // gridDim.z, gridDim.y, gridDim.x: 整个 grid 在三个维度上的 block 数
    printf("block dim: (%3d, %3d, %3d), grid dim: (%3d, %3d, %3d)\n", blockDim.z, blockDim.y, blockDim.x, gridDim.z,
           gridDim.y, gridDim.x);
}

// ============================================================================
// 内核函数 3: print_thread_idx_per_block_kernel
// 功能：把三维的 threadIdx 转换成一维的局部编号（线程在 block 内的编号）
// 用途：理解三维转一维的公式
// ============================================================================
__global__ void print_thread_idx_per_block_kernel()
{
    // 公式含义：
    // threadIdx.z * blockDim.x * blockDim.y : 跳过前面 z 个完整平面
    // threadIdx.y * blockDim.x              : 在当前平面内跳过前面 y 整行
    // threadIdx.x                           : 加上当前行内的 x 偏移
    int index = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    printf("block idx: (%3d, %3d, %3d), thread idx: %3d\n", blockIdx.z, blockIdx.y, blockIdx.x, index);
}

// ============================================================================
// 内核函数 4: print_thread_idx_per_grid_kernel
// 功能：计算每个线程在整个 grid 中的全局唯一编号
// 用途：理解如何给所有线程统一编号（常用于一维数组访问）
// ============================================================================
__global__ void print_thread_idx_per_grid_kernel()
{
    // bSize: 每个 block 包含的线程总数
    int bSize = blockDim.z * blockDim.y * blockDim.x;

    // bIndex: 当前 block 在整个 grid 中的一维编号（第几个 block）
    // 公式含义：跳过前面 z 层完整平面，再跳过前面 y 行，再加上 x
    int bIndex = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

    // tIndex: 当前线程在所在 block 中的一维编号（第几个线程）
    int tIndex = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    // index: 当前线程在整个 grid 中的全局一维编号
    // 公式含义：前面所有 block 的线程总数 + 当前 block 内的线程编号
    int index = bIndex * bSize + tIndex;

    printf("block idx: %3d, thread idx in block: %3d, thread idx: %3d\n", bIndex, tIndex, index);
}

// ============================================================================
// 测试函数 1: print_one_dim
// 功能：一维配置，处理 8 个元素，每个 block 4 个线程，共 2 个 block
// 配置：grid = (2, 1, 1), block = (4, 1, 1)
// ============================================================================
void print_one_dim()
{
    int inputSize = 8;                    // 总共 8 个元素
    int blockDim  = 4;                    // 每个 block 4 个线程
    int gridDim   = inputSize / blockDim; // 需要 2 个 block

    dim3 block(blockDim); // block = (4, 1, 1)
    dim3 grid(gridDim);   // grid = (2, 1, 1)

    // 取消注释下面的任意一行来测试对应的内核函数
    // print_idx_kernel<<<grid, block>>>();
    // print_dim_kernel<<<grid, block>>>();
    // print_thread_idx_per_block_kernel<<<grid, block>>>();
    print_thread_idx_per_grid_kernel<<<grid, block>>>();
    cudaDeviceSynchronize(); // 等待 GPU 执行完成，确保 printf 输出
}

// ============================================================================
// 测试函数 2: print_two_dim
// 功能：二维配置，处理 4x4 矩阵，每个 block 2x2 个线程，共 2x2 个 block
// 配置：grid = (2, 2, 1), block = (2, 2, 1)
// ============================================================================
void print_two_dim()
{
    int inputWidth = 4;                     // 4x4 矩阵
    int blockDim   = 2;                     // 每个 block 2x2 个线程
    int gridDim    = inputWidth / blockDim; // 需要 2x2 个 block

    dim3 block(blockDim, blockDim); // block = (2, 2, 1)
    dim3 grid(gridDim, gridDim);    // grid = (2, 2, 1)

    print_thread_idx_per_grid_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}

// ============================================================================
// 内核函数 5: print_cord_kernel
// 功能：打印线程的 block 索引、block 内一维编号、以及全局二维坐标 (x, y)
// 用途：理解二维坐标和一维编号之间的映射关系
// ============================================================================
__global__ void print_cord_kernel()
{
    // index: 线程在 block 内的一维编号（0 ~ blockDim.x*blockDim.y-1）
    int index = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    // x: 线程在整个 grid 中的 X 坐标（列）
    // 公式：前面的 block 贡献的列数 + 当前 block 内的列偏移
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // y: 线程在整个 grid 中的 Y 坐标（行）
    // 公式：前面的 block 贡献的行数 + 当前 block 内的行偏移
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 打印：block 索引，block 内线程编号，全局坐标
    printf("block idx: (%3d, %3d, %3d), thread idx: %3d, cord: (%3d, %3d)\n", blockIdx.z, blockIdx.y, blockIdx.x, index,
           x, y);
}

// ============================================================================
// 测试函数 3: print_cord
// 功能：演示二维坐标 (x, y) 的计算方法
// 配置：处理 4x4 矩阵，每个 block 2x2 个线程，共 2x2 个 block
// ============================================================================
void print_cord()
{
    int inputWidth = 4;                     // 4x4 矩阵
    int blockDim   = 2;                     // 每个 block 2x2 个线程
    int gridDim    = inputWidth / blockDim; // 需要 2x2 个 block

    dim3 block(blockDim, blockDim); // block = (2, 2, 1)
    dim3 grid(gridDim, gridDim);    // grid = (2, 2, 1)

    print_cord_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}

// ============================================================================
// 主函数
// ============================================================================
int main()
{
    // 取消注释下面的任意一行来测试不同的功能
    // print_one_dim();   // 测试一维配置
    // print_two_dim();   // 测试二维配置
    print_cord(); // 测试二维坐标计算（当前启用）

    return 0;
}
