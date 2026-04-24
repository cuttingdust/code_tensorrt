#include "utils.hpp"
#include <cuda_runtime.h>

// 定义 Tile（分块）的大小为 16x16
// Tile 是矩阵乘法中一次加载到共享内存中的子块大小
#define BLOCKSIZE 32

/*
    这是一个**故意制造 Bank Conflict** 的矩阵乘法 kernel（静态共享内存版本）
    
    什么是 Bank Conflict？
    - 共享内存被分成 32 个 Bank（体）
    - 每个 Bank 一次只能响应一个线程的访问
    - 当一个 Warp（32个线程）中的多个线程访问同一个 Bank 的不同地址时，就发生了 Bank Conflict
    - Bank Conflict 会导致访问被串行化，严重影响性能
    
    本 kernel 通过以下方式故意制造 Bank Conflict：
    1. 交换索引：用 [tx][ty] 代替 [ty][tx]
    2. 交换全局坐标：用 x 代替 y，用 y 代替 x
    3. 导致同一 Warp 内的线程访问共享内存中不同行的同一列（跨步访问）
    
    跨步访问 = stride 是 Bank 数量的倍数 → 多个线程落在同一个 Bank → Bank Conflict
*/
__global__ void MatmulSharedStaticConflictKernel(float *M_device, float *N_device, float *P_device, int width)
{
    // =========================================================================
    // 1. 声明共享内存（静态版本）
    // =========================================================================
    // 静态共享内存：大小在编译时通过宏 BLOCKSIZE 确定
    // 共享内存位于 GPU 的 SM（流多处理器）内部，访问速度比全局内存快约 100 倍
    __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];

    // =========================================================================
    // 2. 计算当前线程在结果矩阵 P 中的全局坐标
    // =========================================================================
    //
    // 注意：这里的索引计算是**故意制造错误**的！
    // 正确的版本应该是：
    //   x = blockIdx.x * blockDim.x + threadIdx.x   (列索引)
    //   y = blockIdx.y * blockDim.y + threadIdx.y   (行索引)
    //
    // 但这里为了制造 Bank Conflict，我们交换了 x 和 y 的含义：
    //   x 变成了行索引，y 变成了列索引（与正确版本相反）
    //
    int x = blockIdx.x * blockDim.x + threadIdx.x; // ❌ 这应该是列索引，但被当作行索引使用
    int y = blockIdx.y * blockDim.y + threadIdx.y; // ❌ 这应该是行索引，但被当作列索引使用

    // 累加器，用于存储当前线程计算的部分和
    float P_element = 0.0;

    // 保存线程在 Block 中的局部坐标
    int ty = threadIdx.y; // 行局部索引
    int tx = threadIdx.x; // 列局部索引

    // =========================================================================
    // 3. 分块矩阵乘法的核心循环
    // =========================================================================
    // 循环次数 = width / BLOCKSIZE
    // 例如 width=4096, BLOCKSIZE=16 时，需要循环 256 次
    for (int m = 0; m < width / BLOCKSIZE; m++)
    {
        // ---------------------------------------------------------------------
        // 3.1 将数据从全局内存加载到共享内存（协作加载）
        // ---------------------------------------------------------------------
        //
        // 正确版本的加载：
        //   M_deviceShared[ty][tx] = M_device[y * width + (m * BLOCKSIZE + tx)];
        //   N_deviceShared[ty][tx] = N_device[(m * BLOCKSIZE + ty) * width + x];
        //
        // 这里为了制造 Bank Conflict，做了以下修改：
        //   1. 交换了共享内存的索引：[tx][ty] 代替 [ty][tx]
        //   2. 交换了全局坐标：用 x 代替 y，用 y 代替 x
        //   3. N 的数据来源错误：用了 M_device 而不是 N_device
        //
        // ❌ 错误1：M 的加载，索引交换 + 坐标交换
        M_deviceShared[tx][ty] = M_device[x * width + (m * BLOCKSIZE + ty)];

        // ❌ 错误2：N 的加载，数据来源错误（应该是 N_device，却用了 M_device）
        N_deviceShared[tx][ty] = N_device[(m * BLOCKSIZE + tx) * width + y];

        // __syncthreads() 是线程块内的同步点
        // 确保所有线程都完成加载后，才能开始计算
        __syncthreads();

        // ---------------------------------------------------------------------
        // 3.2 使用共享内存中的数据计算部分和
        // ---------------------------------------------------------------------
        //
        // 正确版本的计算：
        //   P_element += M_deviceShared[ty][k] * N_deviceShared[k][tx];
        //
        // 这里为了制造 Bank Conflict，做了以下修改：
        //   1. M 的访问：[tx][k] 代替 [ty][k]
        //   2. N 的访问：[k][ty] 代替 [k][tx]
        //
        for (int k = 0; k < BLOCKSIZE; k++)
        {
            // ❌ 访问模式改变，可能导致 Bank Conflict
            P_element += M_deviceShared[tx][k] * N_deviceShared[k][ty];
        }

        // 再次同步，确保所有线程完成当前 Tile 的计算
        __syncthreads();
    }

    // =========================================================================
    // 4. 将结果写回全局内存
    // =========================================================================
    //
    // 正确版本：
    //   P_device[y * width + x] = P_element;
    //
    // 这里为了配合前面的索引交换，写回时也交换了 x 和 y
    // 这样最终结果是 P[x][y] 而不是 P[y][x]，导致结果矩阵被转置
    //
    P_device[x * width + y] = P_element;
}


/*
    这是一个**故意制造 Bank Conflict** 的矩阵乘法 kernel（动态共享内存版本）
    
    动态共享内存 vs 静态共享内存：
    - 静态：大小在编译时确定，可以声明二维数组，代码直观
    - 动态：大小在运行时确定，只能声明一维数组，需要手动管理内存布局
    
    动态共享内存的内存布局：
    ┌────────────────────────────────┬──────────────────────────────┐
    │   M Tile (stride 个元素)      │   N Tile (stride 个元素)      │
    │   索引: 0 ~ stride-1          │   索引: stride ~ 2*stride-1   │
    └────────────────────────────────┴──────────────────────────────┘
*/
__global__ void MatmulSharedDynamicConflictKernel(float *M_device, float *N_device, float *P_device, int width,
                                                  int blockSize)
{
    // =========================================================================
    // 1. 声明动态共享内存
    // =========================================================================
    // 注意：动态共享内存必须用 extern 关键字，且只能是一维数组
    // 大小由 kernel 启动时的第三个参数指定（sMemSize）
    extern __shared__ float deviceShared[];

    // 计算每个 Tile 的大小（元素个数）
    // 例如 blockSize=16 时，stride = 16 × 16 = 256
    int stride = blockSize * blockSize;

    // =========================================================================
    // 2. 计算当前线程在结果矩阵中的全局坐标（与静态版本相同的错误模式）
    // =========================================================================
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // =========================================================================
    // 3. 分块矩阵乘法的核心循环
    // =========================================================================
    for (int m = 0; m < width / blockSize; m++)
    {
        // ---------------------------------------------------------------------
        // 3.1 将数据加载到共享内存（协作加载）
        // ---------------------------------------------------------------------
        //
        // 正确版本：
        //   deviceShared[ty * blockSize + tx] = M_device[y * width + (m * blockSize + tx)];
        //   deviceShared[stride + (ty * blockSize + tx)] = N_device[(m * blockSize + ty) * width + x];
        //
        // 这里为了制造 Bank Conflict，做了以下修改：
        //   1. 交换了索引：用 tx * blockSize + ty 代替 ty * blockSize + tx
        //   2. 交换了全局坐标：用 x 代替 y，用 y 代替 x
        //   3. N 的数据来源错误（用了 M_device 而不是 N_device）
        //
        // ❌ M 的加载（索引交换 + 坐标交换）
        deviceShared[tx * blockSize + ty] = M_device[x * width + (m * blockSize + ty)];

        // ❌ N 的加载（数据来源错误）
        deviceShared[stride + (tx * blockSize + ty)] = N_device[(m * blockSize + tx) * width + y];

        // 等待所有线程完成加载
        __syncthreads();

        // ---------------------------------------------------------------------
        // 3.2 计算部分和
        // ---------------------------------------------------------------------
        for (int k = 0; k < blockSize; k++)
        {
            // ❌ 访问模式改变，可能导致 Bank Conflict
            P_element += deviceShared[tx * blockSize + k] * deviceShared[stride + (k * blockSize + ty)];
        }

        // 等待所有线程完成当前 Tile 的计算
        __syncthreads();
    }

    // =========================================================================
    // 4. 将结果写回全局内存（边界检查 + 转置输出）
    // =========================================================================
    if (y < width && x < width)
    {
        // 结果被转置存储：P[x][y] 而不是 P[y][x]
        P_device[x * width + y] = P_element;
    }
}


/*
    主机端函数：在 GPU 上执行分块矩阵乘法（Bank Conflict 版本）
    
    参数：
    - M_host, N_host: 主机端（CPU）的输入矩阵
    - P_host: 主机端（CPU）的输出矩阵（结果会拷贝到这里）
    - width: 矩阵的宽度（假设矩阵是 width × width 的方阵）
    - blockSize: Tile 的大小（同时也是 Block 的线程数，即 blockSize × blockSize）
    - staticMem: true=使用静态共享内存版本，false=使用动态共享内存版本
*/
void MatmulSharedConflictOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize, bool staticMem)
{
    // 计算整个矩阵占用的内存大小（字节）
    // width × width 个 float，每个 float 占 4 字节
    int size = width * width * sizeof(float);

    /*
        计算动态共享内存所需的大小
        需要两块 blockSize×blockSize 的 Tile
        每块占 blockSize * blockSize * sizeof(float) 字节
        总共需要 2 倍的大小
    */
    long int sMemSize = blockSize * blockSize * sizeof(float) * 2;

    // =========================================================================
    // 1. 在 GPU 上分配 M 和 N 的显存空间
    // =========================================================================
    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc((void **)&M_device, size));
    CUDA_CHECK(cudaMalloc((void **)&N_device, size));

    // =========================================================================
    // 2. 将 M 和 N 的数据从主机内存拷贝到 GPU 显存
    // =========================================================================
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    // =========================================================================
    // 3. 在 GPU 上分配结果矩阵 P 的显存空间
    // =========================================================================
    float *P_device;
    CUDA_CHECK(cudaMalloc((void **)&P_device, size));

    // =========================================================================
    // 4. 配置 kernel 启动参数
    // =========================================================================
    // dimBlock：每个 Block 的线程数，配置为 blockSize × blockSize 的二维线程块
    // dimGrid：Grid 中的 Block 数，总 Block 数 = (width / blockSize) × (width / blockSize)
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);

    // =========================================================================
    // 5. 启动 kernel
    // =========================================================================
    if (staticMem)
    {
        // 静态共享内存版本：大小在编译时确定，不需要额外的启动参数
        MatmulSharedStaticConflictKernel<<<dimGrid, dimBlock>>>(M_device, N_device, P_device, width);
    }
    else
    {
        // 动态共享内存版本：通过第三个参数 sMemSize 在运行时指定共享内存大小
        // 第四个参数是 CUDA 流（这里用 nullptr，表示默认流）
        MatmulSharedDynamicConflictKernel<<<dimGrid, dimBlock, sMemSize, nullptr>>>(M_device, N_device, P_device, width,
                                                                                    blockSize);
    }

    // =========================================================================
    // 6. 将计算结果从 GPU 显存拷贝回主机内存
    // =========================================================================
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));

    // 同步 GPU 设备，确保所有操作完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // 检查 kernel 执行过程中是否有错误
    LAST_KERNEL_CHECK();

    // =========================================================================
    // 7. 释放 GPU 显存
    // =========================================================================
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}
