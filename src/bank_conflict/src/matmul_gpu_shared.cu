#include "utils.hpp"

#include <cuda_runtime.h>

// 定义 Tile（分块）的大小为 16x16
// Tile 是矩阵乘法中一次加载到共享内存中的子块大小
#define BLOCKSIZE 16

/*
    使用共享内存（Shared Memory）优化矩阵乘法的静态版本
    静态指的是共享内存的大小在编译时通过宏 BLOCKSIZE 确定
    
    核心优化思想：将矩阵分块（Tiling），利用共享内存的高速访问特性，
    减少对全局内存的重复读取
    
    每个 Block 负责计算结果矩阵中的一个 Tile（BLOCKSIZE x BLOCKSIZE 的区域）
*/
__global__ void MatmulSharedStaticKernel(float *M_device, float *N_device, float *P_device, int width)
{
    // 声明共享内存数组，大小在编译时固定
    // 这些数组位于 GPU 的共享内存中，访问速度比全局内存快约 100 倍
    // 同一个 Block 内的所有线程共享这两块内存
    __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];

    /*
        计算当前线程在结果矩阵 P 中的全局坐标
        
        坐标计算公式：
        - x = blockIdx.x * blockDim.x + threadIdx.x   (列索引)
        - y = blockIdx.y * blockDim.y + threadIdx.y   (行索引)
        
        其中：
        - blockIdx.x/y：当前 Block 在 Grid 中的索引
        - blockDim.x/y：每个 Block 包含的线程数（这里等于 BLOCKSIZE）
        - threadIdx.x/y：当前线程在 Block 中的局部索引
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 累加器，用于存储当前线程计算的部分和
    // 每个线程负责计算 P[y][x] 这一个元素
    float P_element = 0.0;

    // 保存线程在 Block 中的局部坐标，方便后续使用
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    /*
        分块矩阵乘法的核心循环
        将 M 和 N 按 BLOCKSIZE 大小分成若干 Tile，逐个计算
        
        循环次数 = width / BLOCKSIZE
        例如 width=1024, BLOCKSIZE=16 时，需要循环 64 次
        
        算法原理：
        P[y][x] = Σ(m=0 to 63) [M[y][m*16 + 0..15] × N[m*16 + 0..15][x]]
        
        每次循环处理一个 Tile：
        1. 将 M 的一个 Tile 加载到共享内存
        2. 将 N 的一个 Tile 加载到共享内存
        3. 使用共享内存中的数据计算部分和
    */
    for (int m = 0; m < width / BLOCKSIZE; m++)
    {
        /*
            协作加载：整个 Block 的所有线程一起将数据从全局内存加载到共享内存
            
            M_device[y * width + (m * BLOCKSIZE + tx)]
            - y * width：定位到第 y 行
            - m * BLOCKSIZE + tx：定位到该行的第 (m*16 + tx) 列
            
            每个线程负责加载一个元素，整个 Block 的 BLOCKSIZE×BLOCKSIZE 个线程
            一次循环就能把整个 Tile 加载完毕
        */
        M_deviceShared[ty][tx] = M_device[y * width + (m * BLOCKSIZE + tx)];
        N_deviceShared[ty][tx] = N_device[(m * BLOCKSIZE + ty) * width + x];

        /*
            __syncthreads() 是线程块内的同步函数
            确保所有线程都完成上面的加载操作后，才继续往下执行
            这是关键！否则有些线程可能读到未加载完成的旧数据
        */
        __syncthreads();

        /*
            使用共享内存中的数据计算部分和
            从共享内存读取比从全局内存快约 100 倍
            
            计算当前线程的贡献：
            P_element += Σ(k=0 to BLOCKSIZE-1) M_tile[ty][k] × N_tile[k][tx]
            
            注意：这里访问的是共享内存，速度很快
        */
        for (int k = 0; k < BLOCKSIZE; k++)
        {
            P_element += M_deviceShared[ty][k] * N_deviceShared[k][tx];
        }

        /*
            再次同步，确保所有线程都完成当前 Tile 的计算
            然后再进行下一轮循环，加载新的 Tile
            这样可以避免新数据覆盖正在被使用的旧数据
        */
        __syncthreads();
    }

    // 将最终结果写回全局内存
    P_device[y * width + x] = P_element;
}


/*
    使用共享内存优化矩阵乘法的动态版本
    动态指的是共享内存的大小在运行时通过 kernel 启动参数确定
    
    优点：同一个 kernel 可以处理不同大小的 Block，更灵活
    缺点：只能声明一维数组，需要手动管理内存布局
*/
__global__ void MatmulSharedDynamicKernel(float *M_device, float *N_device, float *P_device, int width, int blockSize)
{
    /*
        声明动态共享变量
        
        关键点：
        1. 必须加 extern 关键字
        2. 必须是一维数组
        3. 不能写成多维数组
        4. 大小由 kernel 启动时的第三个参数指定（sMemSize）
        
        重要陷阱：
        如果写成 __shared__ float M_arr[]; __shared__ float N_arr[];
        这两个变量实际上指向同一个地址！因为动态共享内存只有一个连续的区域
        
        正确做法：只用一维数组，通过偏移量来区分 M 和 N 的区域
    */
    extern __shared__ float deviceShared[];

    // 计算每个 Tile 的大小（元素个数）
    // 例如 blockSize=16 时，stride = 256
    int stride = blockSize * blockSize;

    /*
        计算当前线程在结果矩阵中的全局坐标
        注意：这里使用 blockSize 参数而不是 blockDim.x
        因为动态版本中 blockSize 是从参数传入的
    */
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    /*
        分块矩阵乘法的核心循环
        与静态版本逻辑相同，但使用一维数组和偏移量来访问共享内存
    */
    for (int m = 0; m < width / blockSize; m++)
    {
        /*
            将数据加载到共享内存
            
            共享内存布局：
            ┌────────────────────────────────┬──────────────────────────────┐
            │   M Tile (stride 个元素)      │   N Tile (stride 个元素)      │
            │   索引: 0 ~ stride-1          │   索引: stride ~ 2*stride-1   │
            └────────────────────────────────┴──────────────────────────────┘
            
            ty * blockSize + tx：计算当前线程在 M Tile 中的一维索引
            stride + (ty * blockSize + tx)：加上偏移量，定位到 N Tile
        */
        deviceShared[ty * blockSize + tx]            = M_device[y * width + (m * blockSize + tx)];
        deviceShared[stride + (ty * blockSize + tx)] = N_device[(m * blockSize + ty) * width + x];

        // 等待所有线程完成加载
        __syncthreads();

        /*
            计算部分和
            从共享内存读取数据：deviceShared[...]
            
            M 元素位置：deviceShared[ty * blockSize + k]
            N 元素位置：deviceShared[stride + (k * blockSize + tx)]
            
            注意：k 从 0 到 blockSize-1，遍历 Tile 的整个维度
        */
        for (int k = 0; k < blockSize; k++)
        {
            P_element += deviceShared[ty * blockSize + k] * deviceShared[stride + (k * blockSize + tx)];
        }

        // 等待所有线程完成当前 Tile 的计算
        __syncthreads();
    }

    /*
        边界检查
        当 width 不是 blockSize 的整数倍时，需要防止越界访问
        例如 width=1024, blockSize=16 时正好整除，此检查是多余的但不影响性能
    */
    if (y < width && x < width)
    {
        P_device[y * width + x] = P_element;
    }
}


/*
    主机端函数：在 GPU 上执行分块矩阵乘法
    
    参数：
    - M_host, N_host: 主机端（CPU）的输入矩阵
    - P_host: 主机端（CPU）的输出矩阵（结果会拷贝到这里）
    - width: 矩阵的宽度（假设矩阵是 width × width 的方阵）
    - blockSize: Tile 的大小（同时也是 Block 的线程数，即 blockSize × blockSize）
    - staticMem: 是否使用静态共享内存版本（true=静态，false=动态）
*/
void MatmulSharedOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize, bool staticMem)
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

    // 在 GPU 上分配 M 和 N 的显存空间
    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc((void **)&M_device, size));
    CUDA_CHECK(cudaMalloc((void **)&N_device, size));

    // 将 M 和 N 的数据从主机内存拷贝到 GPU 显存
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    // 在 GPU 上分配结果矩阵 P 的显存空间
    float *P_device;
    CUDA_CHECK(cudaMalloc((void **)&P_device, size));

    /*
        配置 kernel 启动参数
        
        dimBlock：每个 Block 的线程数，这里配置为 blockSize × blockSize 的二维线程块
        dimGrid：Grid 中的 Block 数，每个 Block 负责一个 Tile
                 总 Block 数 = (width / blockSize) × (width / blockSize)
    */
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);

    // 根据 staticMem 参数选择使用静态版本还是动态版本
    if (staticMem)
    {
        /*
            静态共享内存版本
            共享内存大小在编译时确定（BLOCKSIZE × BLOCKSIZE × 2 × sizeof(float)）
            不需要额外的启动参数
        */
        MatmulSharedStaticKernel<<<dimGrid, dimBlock>>>(M_device, N_device, P_device, width);
    }
    else
    {
        /*
            动态共享内存版本
            通过第三个参数 sMemSize 在运行时指定共享内存大小
            第四个参数是 CUDA 流（这里用 nullptr，表示默认流）
        */
        MatmulSharedDynamicKernel<<<dimGrid, dimBlock, sMemSize, nullptr>>>(M_device, N_device, P_device, width,
                                                                            blockSize);
    }

    // 将计算结果从 GPU 显存拷贝回主机内存
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));

    // 同步 GPU 设备，确保所有操作完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // 检查 kernel 执行过程中是否有错误
    LAST_KERNEL_CHECK();

    // 释放 GPU 显存，防止内存泄漏
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}
