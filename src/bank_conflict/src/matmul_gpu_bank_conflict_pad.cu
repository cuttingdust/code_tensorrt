#include "utils.hpp"
#include <cuda_runtime.h>

// 定义 Tile（分块）的大小为 32x32
// Tile 是矩阵乘法中一次加载到共享内存中的子块大小
// 注意：BLOCKSIZE=32 时，每个 Block 有 32×32=1024 个线程（达到硬件上限）
#define BLOCKSIZE     32
#define BLOCKSIZE_PAD (BLOCKSIZE + 1) // Padding 后的列数，用于缓解 Bank Conflict

/*
    ============================================================================
    静态共享内存版本（带 Padding）- 用于演示 Padding 如何缓解 Bank Conflict
    ============================================================================
    
    什么是 Bank Conflict？
    - 共享内存被分成 32 个 Bank（体）
    - 每个 Bank 一次只能响应一个线程的访问
    - 当一个 Warp（32个线程）中的多个线程访问同一个 Bank 的不同地址时，就发生了 Bank Conflict
    - Bank Conflict 会导致访问被串行化，严重影响性能
    
    什么是 Padding？
    - 在声明共享内存时，多申请一列（BLOCKSIZE_PAD = BLOCKSIZE + 1）
    - 原本列优先访问时，步长 = BLOCKSIZE（32），是 Bank 数量的倍数，会产生 Bank Conflict
    - Padding 后，步长 = BLOCKSIZE_PAD（33），不再是 32 的倍数，可以缓解 Bank Conflict
    
    本 kernel 的特点：
    - 加载时使用行优先（合并访问，效率高）
    - 计算时使用列优先（会产生 Bank Conflict，但 Padding 可以缓解）
    - 结果被转置存储（用于演示目的）
    
    注意：这是一个教学用的 kernel，故意制造了索引交换和转置输出
*/
__global__ void MatmulSharedStaticConflictPadKernel(float *M_device, float *N_device, float *P_device, int width)
{
    // =========================================================================
    // 1. 声明共享内存（静态版本，带 Padding）
    // =========================================================================
    // 静态共享内存：大小在编译时通过宏 BLOCKSIZE 和 BLOCKSIZE_PAD 确定
    // 共享内存位于 GPU 的 SM（流多处理器）内部，访问速度比全局内存快约 100 倍
    // Padding：每行多申请一列（32 → 33），用于缓解列优先访问时的 Bank Conflict
    __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE_PAD];
    __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE_PAD];

    // =========================================================================
    // 2. 计算当前线程的全局坐标（注意：这里故意交换了 x 和 y 的含义）
    // =========================================================================
    //
    // 正常版本应该是：
    //   x = blockIdx.x * blockDim.x + threadIdx.x   (列索引)
    //   y = blockIdx.y * blockDim.y + threadIdx.y   (行索引)
    //
    // 这里为了演示效果，交换了 x 和 y 的含义：
    //   x 变成了行索引，y 变成了列索引
    //
    int x = blockIdx.x * blockDim.x + threadIdx.x; // ❌ 行索引（本应是列）
    int y = blockIdx.y * blockDim.y + threadIdx.y; // ❌ 列索引（本应是行）

    // 累加器，用于存储当前线程计算的部分和
    float P_element = 0.0;

    // 保存线程在 Block 中的局部坐标
    int ty = threadIdx.y; // 行局部索引（0 ~ 31）
    int tx = threadIdx.x; // 列局部索引（0 ~ 31）

    // =========================================================================
    // 3. 分块矩阵乘法的核心循环
    // =========================================================================
    // 循环次数 = width / BLOCKSIZE
    // width=4096, BLOCKSIZE=32 时，需要循环 128 次
    for (int m = 0; m < width / BLOCKSIZE; m++)
    {
        // ---------------------------------------------------------------------
        // 3.1 将数据从全局内存加载到共享内存（协作加载）
        // ---------------------------------------------------------------------
        //
        // 注意：这里使用了列优先的加载方式 [tx][ty]
        // 这会导致全局内存访问变成非合并访问（不同行的同一列）
        // 但这是为了演示目的，故意制造非合并访问
        //
        // M 的加载：从全局内存读取第 x 行，第 (m*BLOCKSIZE + ty) 列
        M_deviceShared[tx][ty] = M_device[x * width + (m * BLOCKSIZE + ty)];

        // N 的加载：从全局内存读取第 (m*BLOCKSIZE + tx) 行，第 y 列
        N_deviceShared[tx][ty] = N_device[(m * BLOCKSIZE + tx) * width + y];

        // 如果需要合并访问（性能更好的版本），应该使用：
        // M_deviceShared[ty][tx] = M_device[y * width + (m * BLOCKSIZE + tx)];
        // N_deviceShared[ty][tx] = N_device[(m * BLOCKSIZE + ty) * width + x];

        // __syncthreads() 是线程块内的同步点
        // 确保所有线程都完成加载后，才能开始计算
        __syncthreads();

        // ---------------------------------------------------------------------
        // 3.2 使用共享内存中的数据计算部分和
        // ---------------------------------------------------------------------
        //
        // 这里使用了列优先的访问模式：
        //   M_deviceShared[tx][k]：tx 固定，k 变化 → 同一列的不同行
        //   N_deviceShared[k][ty]：ty 固定，k 变化 → 同一行的不同列
        //
        // 由于声明共享内存时添加了 Padding（列数 33）
        // 列优先访问的步长从 32 变为 33，不再是 32 的倍数
        // 因此可以缓解 Bank Conflict
        //
        for (int k = 0; k < BLOCKSIZE; k++)
        {
            P_element += M_deviceShared[tx][k] * N_deviceShared[k][ty];
        }

        // 再次同步，确保所有线程完成当前 Tile 的计算
        __syncthreads();
    }

    // =========================================================================
    // 4. 将结果写回全局内存（转置存储）
    // =========================================================================
    //
    // 正常版本应该是：P_device[y * width + x] = P_element;
    // 这里为了配合前面的索引交换，写回时也交换了 x 和 y
    // 结果矩阵被转置存储：P[x][y] 而不是 P[y][x]
    //
    P_device[x * width + y] = P_element;
}


/*
    ============================================================================
    动态共享内存版本（带 Padding）- 更灵活的实现方式
    ============================================================================
    
    动态共享内存 vs 静态共享内存：
    - 静态：大小在编译时确定，可以声明二维数组，代码直观
    - 动态：大小在运行时确定，只能声明一维数组，需要手动管理内存布局
    
    动态共享内存的内存布局：
    ┌────────────────────────────────┬──────────────────────────────┐
    │   M Tile (stride 个元素)      │   N Tile (stride 个元素)      │
    │   索引: 0 ~ stride-1          │   索引: stride ~ 2*stride-1   │
    └────────────────────────────────┴──────────────────────────────┘
    
    stride = (blockSize + 1) * blockSize，因为 Padding 增加了一列
*/
__global__ void MatmulSharedDynamicConflictPadKernel(float *M_device, float *N_device, float *P_device, int width,
                                                     int blockSize)
{
    // =========================================================================
    // 1. 声明动态共享内存
    // =========================================================================
    // 注意：动态共享内存必须用 extern 关键字，且只能是一维数组
    // 大小由 kernel 启动时的第三个参数指定（sMemSize）
    extern __shared__ float deviceShared[];

    // 计算每个 Tile 的步长（带 Padding）
    // stride = (blockSize + 1) × blockSize
    // 例如 blockSize=32 时，stride = 33×32 = 1056 个元素
    int stride = (blockSize + 1) * blockSize;

    // =========================================================================
    // 2. 计算当前线程的全局坐标
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
        // 共享内存布局（一维数组）：
        //   M Tile 占据 [0, stride-1]
        //   N Tile 占据 [stride, 2*stride-1]
        //
        // 计算 M Tile 中的一维索引：tx * (blockSize+1) + ty
        // 计算 N Tile 中的一维索引：stride + (tx * (blockSize+1) + ty)
        //
        // M 的加载
        deviceShared[tx * (blockSize + 1) + ty] = M_device[x * width + (m * blockSize + ty)];

        // N 的加载
        deviceShared[stride + (tx * (blockSize + 1) + ty)] = N_device[(m * blockSize + tx) * width + y];

        // 等待所有线程完成加载
        __syncthreads();

        // ---------------------------------------------------------------------
        // 3.2 计算部分和
        // ---------------------------------------------------------------------
        for (int k = 0; k < blockSize; k++)
        {
            // M 元素：deviceShared[tx * (blockSize+1) + k]
            // N 元素：deviceShared[stride + (k * (blockSize+1) + ty)]
            P_element += deviceShared[tx * (blockSize + 1) + k] * deviceShared[stride + (k * (blockSize + 1) + ty)];
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
    ============================================================================
    主机端函数：在 GPU 上执行分块矩阵乘法（Bank Conflict 演示版本）
    ============================================================================
    
    功能：在 GPU 上执行分块矩阵乘法，用于演示 Padding 对 Bank Conflict 的影响
    
    参数：
    - M_host, N_host: 主机端（CPU）的输入矩阵
    - P_host: 主机端（CPU）的输出矩阵（结果会拷贝到这里）
    - width: 矩阵的宽度（假设矩阵是 width × width 的方阵）
    - blockSize: Tile 的大小（同时也是 Block 的线程数，即 blockSize × blockSize）
    - staticMem: true=使用静态共享内存版本，false=使用动态共享内存版本
*/
void MatmulSharedConflictPadOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize,
                                     bool staticMem)
{
    // =========================================================================
    // 1. 计算内存大小
    // =========================================================================
    // width × width 个 float，每个 float 占 4 字节
    int size = width * width * sizeof(float);

    // =========================================================================
    // 2. 计算动态共享内存所需的大小
    // =========================================================================
    // 需要两块 blockSize×(blockSize+1) 的 Tile（带 Padding）
    // 每块占 (blockSize+1) × blockSize × sizeof(float) 字节
    // 总共需要 2 倍的大小
    long int sMemSize = (blockSize + 1) * blockSize * sizeof(float) * 2;

    // =========================================================================
    // 3. 在 GPU 上分配 M 和 N 的显存空间
    // =========================================================================
    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc((void **)&M_device, size));
    CUDA_CHECK(cudaMalloc((void **)&N_device, size));

    // =========================================================================
    // 4. 将 M 和 N 的数据从主机内存拷贝到 GPU 显存
    // =========================================================================
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    // =========================================================================
    // 5. 在 GPU 上分配结果矩阵 P 的显存空间
    // =========================================================================
    float *P_device;
    CUDA_CHECK(cudaMalloc((void **)&P_device, size));

    // =========================================================================
    // 6. 配置 kernel 启动参数
    // =========================================================================
    // dimBlock：每个 Block 的线程数，配置为 blockSize × blockSize 的二维线程块
    // dimGrid：Grid 中的 Block 数，总 Block 数 = (width / blockSize) × (width / blockSize)
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);

    // =========================================================================
    // 7. 启动 kernel
    // =========================================================================
    if (staticMem)
    {
        // 静态共享内存版本：大小在编译时确定，不需要额外的启动参数
        MatmulSharedStaticConflictPadKernel<<<dimGrid, dimBlock>>>(M_device, N_device, P_device, width);
    }
    else
    {
        // 动态共享内存版本：通过第三个参数 sMemSize 在运行时指定共享内存大小
        // 第四个参数是 CUDA 流（这里用 nullptr，表示默认流）
        MatmulSharedDynamicConflictPadKernel<<<dimGrid, dimBlock, sMemSize, nullptr>>>(M_device, N_device, P_device,
                                                                                       width, blockSize);
    }

    // =========================================================================
    // 8. 将计算结果从 GPU 显存拷贝回主机内存
    // =========================================================================
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));

    // 同步 GPU 设备，确保所有操作完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // 检查 kernel 执行过程中是否有错误
    LAST_KERNEL_CHECK();

    // =========================================================================
    // 9. 释放 GPU 显存
    // =========================================================================
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}
