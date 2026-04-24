#include <cuda_runtime.h>
#include <utils.hpp>
#include <vector>

// ============================================================================
// 配置宏：控制 SleepKernel 的循环次数，用于模拟不同的计算负载
// ============================================================================
// #define MAX_ITER 10000 // memcpy == kernel / 10    (kernel执行太快，看不出 overlapping)
// #define MAX_ITER 100000 // memcpy == kernel / 100   (开始能看出 kernel 的 overlapping)
#define MAX_ITER 5000000 // memcpy == kernel / 10000  (可以非常清楚地看到 kernel 的 overlapping)

// 数据大小：32x32 矩阵（足够小，便于测试）
#define SIZE 32

#define NUM_KERNELS 3 // 每个 Stream 中 Kernel 的数量


#define BIG_KERNEL_ITER   10000000 // 大 Kernel：执行时间长
#define SMALL_KERNEL_ITER 5000000  // 小 Kernel：执行时间短


/*
 * ============================================================================
 * SleepKernel - 一个空转的核函数，用于模拟计算负载
 * ============================================================================
 * 
 * 作用：通过 clock64() 进行忙等待（busy-wait），模拟一个耗时的计算任务。
 * 
 * 为什么需要这个？
 * - 如果核函数执行太快，无法观察到多 stream 之间的并发效果
 * - 通过调整 MAX_ITER，可以控制核函数的执行时间
 * - 让大家可以根据自己的 GPU 性能调整 sleep 时间
 * 
 * 参数：
 *   num_cycles - 需要空转的时钟周期数
 * 
 * 注意：
 *   clock64() 返回 GPU 时钟周期计数，可用于精确计时和忙等待
 */
__global__ void SleepKernel(int64_t num_cycles)
{
    // 记录当前已循环的周期数
    int64_t cycles = 0;
    // 记录开始时的时钟周期
    int64_t start = clock64();

    // 忙等待，直到达到指定的时钟周期数
    // 注意：这是 CPU 风格的空转，在实际生产代码中不推荐，
    // 仅用于教学演示 Stream 并发效果
    while (cycles < num_cycles)
    {
        cycles = clock64() - start;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * ============================================================================
 * MultiStreamSeparate - 多 Stream 版本（分离传输与计算）
 * 
 * 设计：
 *   - Stream 0：专门负责所有 H2D 传输
 *   - Stream 1 ~ N-1：每个负责一个数据块的 Kernel + D2H
 * 
 * 流水线：
 *   1. H2D Stream 异步传输第 i 块数据到 GPU
 *   2. 计算 Stream i 等待传输完成后，执行 Kernel，然后 D2H
 *   3. 不同块的传输、计算、回传可以重叠
 * ============================================================================
 */
void MultiStreamSeparate(float *src_host, float *tar_host, int width, int blockSize, int streamsCount)
{
    // streamsCount 包括 H2D Stream + 计算 Stream 数量
    // 实际可用的计算 Stream 数量 = streamsCount - 1
    int computeStreams = streamsCount - 1;
    if (computeStreams <= 0)
    {
        printf("ERROR: streamsCount must be at least 2\n");
        return;
    }

    int size       = width * width * sizeof(float);
    int chunkSize  = size / computeStreams; // 每个计算 Stream 负责的数据块大小
    int chunkWidth = width / computeStreams;

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(chunkWidth / blockSize, chunkWidth / blockSize);

    // 为每个计算 Stream 分配独立的 GPU 缓冲区
    std::vector<float *> src_device(computeStreams);
    std::vector<float *> tar_device(computeStreams);
    for (int i = 0; i < computeStreams; i++)
    {
        CUDA_CHECK(cudaMalloc(&src_device[i], chunkSize));
        CUDA_CHECK(cudaMalloc(&tar_device[i], chunkSize));
    }

    // 创建 Stream
    cudaStream_t h2dStream;
    cudaStreamCreate(&h2dStream);
    std::vector<cudaStream_t> computeStream(computeStreams);
    for (int i = 0; i < computeStreams; i++)
    {
        cudaStreamCreate(&computeStream[i]);
    }

    // 创建事件用于同步（确保 H2D 完成后再启动 Kernel）
    std::vector<cudaEvent_t> h2dDone(computeStreams);
    for (int i = 0; i < computeStreams; i++)
    {
        cudaEventCreate(&h2dDone[i]);
    }

    // =========================================================================
    // 流水线执行
    // =========================================================================
    for (int i = 0; i < computeStreams; i++)
    {
        int offset = i * chunkSize / sizeof(float);

        // 1. H2D Stream 异步拷贝数据到 GPU
        CUDA_CHECK(cudaMemcpyAsync(src_device[i], src_host + offset, chunkSize, cudaMemcpyHostToDevice, h2dStream));

        // 记录 H2D 完成事件
        CUDA_CHECK(cudaEventRecord(h2dDone[i], h2dStream));

        // 2. 计算 Stream 等待 H2D 完成
        CUDA_CHECK(cudaStreamWaitEvent(computeStream[i], h2dDone[i], 0));

        // 3. 计算 Stream 执行 Kernel
        SleepKernel<<<dimGrid, dimBlock, 0, computeStream[i]>>>(MAX_ITER * NUM_KERNELS);

        // 4. 计算 Stream 执行 D2H 回传
        CUDA_CHECK(
                cudaMemcpyAsync(tar_host + offset, tar_device[i], chunkSize, cudaMemcpyDeviceToHost, computeStream[i]));
    }

    // 等待所有操作完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // 释放资源
    cudaStreamDestroy(h2dStream);
    for (int i = 0; i < computeStreams; i++)
    {
        cudaStreamDestroy(computeStream[i]);
        cudaEventDestroy(h2dDone[i]);
        cudaFree(src_device[i]);
        cudaFree(tar_device[i]);
    }
}


/*
 * ============================================================================
 * SingleStreamSmallBig - 单 Stream 版本（基准测试）
 * 
 * 每个任务：H2D → 小Kernel → H2D → 大Kernel → D2H
 * ============================================================================
 */
void SingleStreamSmallBig(float *src_host, float *tar_host, int width, int blockSize, int totalTasks)
{
    int size = width * width * sizeof(float);

    /// 确保参数有效
    if (width % blockSize != 0)
    {
        printf("ERROR: width(%d) must be divisible by blockSize(%d)\n", width, blockSize);
        return;
    }

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);

    /// 为每个任务分配独立的缓冲区（大小都是完整的 matrix size）
    float **src_devices = new float *[totalTasks];
    float **tar_devices = new float *[totalTasks];

    for (int i = 0; i < totalTasks; i++)
    {
        CUDA_CHECK(cudaMalloc(&src_devices[i], size));
        CUDA_CHECK(cudaMalloc(&tar_devices[i], size));
    }

    for (int i = 0; i < totalTasks; i++)
    {
        // 1. 第一次 H2D：拷贝完整数据到 GPU
        CUDA_CHECK(cudaMemcpy(src_devices[i], src_host, size, cudaMemcpyHostToDevice));

        // 2. 小 Kernel（执行时间短）
        SleepKernel<<<dimGrid, dimBlock>>>(SMALL_KERNEL_ITER);

        // 3. 第二次 H2D：拷贝另一份完整数据（模拟处理不同的输入）
        //    注意：这里使用 src_host 相同的数据，或者可以准备两份不同的源数据
        CUDA_CHECK(cudaMemcpy(src_devices[i], src_host, size, cudaMemcpyHostToDevice));

        // 4. 大 Kernel（执行时间长）
        SleepKernel<<<dimGrid, dimBlock>>>(BIG_KERNEL_ITER);

        // 5. D2H：拷贝结果回 CPU
        CUDA_CHECK(cudaMemcpy(tar_host, tar_devices[i], size, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < totalTasks; i++)
    {
        cudaFree(src_devices[i]);
        cudaFree(tar_devices[i]);
    }
    delete[] src_devices;
    delete[] tar_devices;
}


/*
 * ============================================================================
 * SingleStreamInterleaved - 单 Stream 版本（基准测试）
 * 
 * 每个任务：H2D → 大Kernel → H2D → 小Kernel → D2H
 * ============================================================================
 */
void SingleStreamInterleaved(float *src_host, float *tar_host, int width, int blockSize, int totalTasks)
{
    int size = width * width * sizeof(float);

    /// 确保参数有效
    if (width % blockSize != 0)
    {
        printf("ERROR: width(%d) must be divisible by blockSize(%d)\n", width, blockSize);
        return;
    }

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);

    /// 为每个任务分配独立的缓冲区（大小都是完整的 matrix size）
    float **src_devices = new float *[totalTasks];
    float **tar_devices = new float *[totalTasks];

    for (int i = 0; i < totalTasks; i++)
    {
        CUDA_CHECK(cudaMalloc(&src_devices[i], size));
        CUDA_CHECK(cudaMalloc(&tar_devices[i], size));
    }

    for (int i = 0; i < totalTasks; i++)
    {
        // 1. 第一次 H2D：拷贝完整数据到 GPU
        CUDA_CHECK(cudaMemcpy(src_devices[i], src_host, size, cudaMemcpyHostToDevice));

        // 2. 小 Kernel（执行时间短）
        SleepKernel<<<dimGrid, dimBlock>>>(SMALL_KERNEL_ITER);

        // 3. 第二次 H2D：拷贝另一份完整数据（模拟处理不同的输入）
        //    注意：这里使用 src_host 相同的数据，或者可以准备两份不同的源数据
        CUDA_CHECK(cudaMemcpy(src_devices[i], src_host, size, cudaMemcpyHostToDevice));

        // 4. 大 Kernel（执行时间长）
        SleepKernel<<<dimGrid, dimBlock>>>(BIG_KERNEL_ITER);

        // 5. D2H：拷贝结果回 CPU
        CUDA_CHECK(cudaMemcpy(tar_host, tar_devices[i], size, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < totalTasks; i++)
    {
        cudaFree(src_devices[i]);
        cudaFree(tar_devices[i]);
    }
    delete[] src_devices;
    delete[] tar_devices;
}


/*
 * ============================================================================
 * SingleStreamMixedKernel - 单 Stream 版本（基准测试）
 * 
 * 每个任务：H2D → 大Kernel → 小Kernel → D2H
 * ============================================================================
 */
void SingleStreamMixedKernel(float *src_host, float *tar_host, int width, int blockSize, int totalTasks)
{
    int size       = width * width * sizeof(float);
    int chunkSize  = size / totalTasks;  /// 每个任务的数据块大小
    int chunkWidth = width / totalTasks; /// 每个任务的矩阵宽度

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(chunkWidth / blockSize, chunkWidth / blockSize);

    /// 分配 GPU 缓冲区
    float *src_device, *tar_device;
    CUDA_CHECK(cudaMalloc(&src_device, chunkSize));
    CUDA_CHECK(cudaMalloc(&tar_device, chunkSize));

    for (int i = 0; i < totalTasks; i++)
    {
        int offset = i * chunkSize / sizeof(float);

        /// 1. H2D：拷贝数据到 GPU
        CUDA_CHECK(cudaMemcpy(src_device, src_host + offset, chunkSize, cudaMemcpyHostToDevice));

        /// 2. 大 Kernel（执行时间长）
        SleepKernel<<<dimGrid, dimBlock>>>(BIG_KERNEL_ITER);

        /// 3. 小 Kernel（执行时间短）
        SleepKernel<<<dimGrid, dimBlock>>>(SMALL_KERNEL_ITER);

        /// 4. D2H：拷贝结果回 CPU
        CUDA_CHECK(cudaMemcpy(tar_host + offset, tar_device, chunkSize, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(src_device);
    cudaFree(tar_device);
}


void SingleStreamMultiKernel(float *src_host, float *tar_host, int width, int blockSize, int totalTasks)
{
    // 单 Stream 版本：只应使用 1 个 Stream，串行处理所有任务
    int size      = width * width * sizeof(float);
    int chunkSize = size / totalTasks;

    float *src_device, *tar_device;
    CUDA_CHECK(cudaMalloc(&src_device, chunkSize));
    CUDA_CHECK(cudaMalloc(&tar_device, chunkSize));

    for (int i = 0; i < totalTasks; i++)
    {
        int offset = i * chunkSize / sizeof(float);
        cudaMemcpy(src_device, src_host + offset, chunkSize, cudaMemcpyHostToDevice);
        for (int k = 0; k < NUM_KERNELS; k++)
        {
            dim3 dimBlock(blockSize, blockSize); // 每个 Block 有 blockSize×blockSize 个线程
            dim3 dimGrid(width / blockSize,
                         width / blockSize);              // Grid 中有 (width/blockSize)×(width/blockSize) 个 Block
            SleepKernel<<<dimGrid, dimBlock>>>(MAX_ITER); /// 不指定 Stream，用 Default
        }
        cudaMemcpy(tar_host + offset, tar_device, chunkSize, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    cudaFree(src_device);
    cudaFree(tar_device);
}

/*
 * ============================================================================
 * SleepSingleStream - 单 Stream 版本（基准测试）
 * ============================================================================
 * 
 * 功能：使用单个默认 Stream 执行数据传输和核函数
 * 
 * 流程：
 *   1. 分配 GPU 显存
 *   2. 循环 count 次：
 *      a. 将数据从 Host 拷贝到 Device（同步）
 *      b. 执行 SleepKernel（同步）
 *      c. 将结果从 Device 拷贝回 Host（同步）
 *   3. 释放资源
 * 
 * 特点：
 *   - 使用同步 API（cudaMemcpy）
 *   - 所有操作在默认 Stream 中串行执行
 *   - 传输和计算无法重叠
 * 
 * 为什么作为基准测试？
 *   - 单 Stream 是最简单的执行模型
 *   - 通过对比多 Stream 版本，可以评估 Stream 并发的收益
 * 
 * 参数：
 *   src_host  - 源数据（Host 内存）
 *   tar_host  - 目标数据（Host 内存，本例未使用）
 *   width     - 矩阵宽度
 *   blockSize - Block 大小
 *   count     - 循环次数
 */
void SleepSingleStream(float *src_host, float *tar_host, int width, int blockSize, int count)
{
    // 计算矩阵总字节数
    int size = width * width * sizeof(float);

    // 在 GPU 上分配输入和输出缓冲区
    float *src_device;
    float *tar_device;
    CUDA_CHECK(cudaMalloc((void **)&src_device, size));
    CUDA_CHECK(cudaMalloc((void **)&tar_device, size));

    // 循环执行 count 次（模拟多个任务）
    for (int i = 0; i < count; i++)
    {
        // 1. 同步拷贝：Host → Device
        //    注意：这里的内层循环实际上只执行 1 次，可以简化
        for (int j = 0; j < 1; j++)
        {
            CUDA_CHECK(cudaMemcpy(src_device, src_host, size, cudaMemcpyHostToDevice));
        }

        // 2. 配置核函数启动参数
        dim3 dimBlock(blockSize, blockSize);                // 每个 Block 有 blockSize×blockSize 个线程
        dim3 dimGrid(width / blockSize, width / blockSize); // Grid 中有 (width/blockSize)×(width/blockSize) 个 Block
        SleepKernel<<<dimGrid, dimBlock>>>(MAX_ITER * NUM_KERNELS); // 启动核函数（同步，因为默认 Stream）

        // 3. 同步拷贝：Device → Host
        CUDA_CHECK(cudaMemcpy(src_host, src_device, size, cudaMemcpyDeviceToHost));
    }

    // 等待所有 GPU 操作完成（确保数据已拷贝回 Host）
    CUDA_CHECK(cudaDeviceSynchronize());

    // 释放 GPU 显存
    cudaFree(tar_device);
    cudaFree(src_device);
}


/*
 * ============================================================================
 * SleepMultiStream - 多 Stream 版本（展示并发效果）
 * ============================================================================
 * 
 * 功能：使用多个 Stream 并发执行数据传输和核函数
 * 
 * 核心思想：数据分块 + 独立缓冲区 + 异步操作
 * 
 * 流程：
 *   1. 将总数据分成 count 块，为每个 Stream 分配独立的 GPU 缓冲区
 *   2. 创建 count 个 Stream
 *   3. 为每个 Stream 提交任务：
 *      a. 异步拷贝自己的数据块到 GPU
 *      b. 异步执行核函数处理自己的数据块
 *      c. 异步拷贝结果回 Host
 *   4. 等待所有 Stream 完成
 *   5. 释放资源
 * 
 * 为什么这样设计？
 *   - 独立缓冲区：避免 Stream 之间相互等待内存资源
 *   - 数据分块：让每个 Stream 处理不同的数据，可以真正并行
 *   - 异步 API：允许 CPU 继续提交任务，不等待 GPU 完成
 * 
 * 预期效果：
 *   - 不同 Stream 的 HtoD 传输、Kernel 执行、DtoH 传输可以重叠
 *   - 总执行时间 ≈ 单块时间 + 少量调度开销
 * 
 * 关键点：
 *   - cudaMemcpyAsync 必须配合固定内存（Pinned Memory）使用
 *   - 不同 Stream 的操作可以并发执行
 *   - 必须调用 cudaDeviceSynchronize() 等待所有 Stream 完成
 *   - 使用完后需要销毁 Stream，释放资源
 * 
 * 参数：
 *   src_host  - 源数据（Host 内存，必须是固定内存，通过 cudaMallocHost 分配）
 *   tar_host  - 目标数据（Host 内存，必须是固定内存）
 *   width     - 矩阵宽度
 *   blockSize - Block 大小
 *   count     - Stream 数量（同时也是分块数量）
 */
void SleepMultiStream(float *src_host, float *tar_host, int width, int blockSize, int count)
{
    // =========================================================================
    // 第一步：计算分块参数
    // =========================================================================
    int size       = width * width * sizeof(float); // 总数据字节数
    int chunkSize  = size / count;                  // 每个 Stream 处理的数据块大小（字节）
    int chunkWidth = width / count;                 // 每个 Stream 处理的矩阵宽度（假设整除）

    // =========================================================================
    // 第二步：配置核函数启动参数
    // =========================================================================
    dim3 dimBlock(blockSize, blockSize);                          // 每个 Block 的线程数
    dim3 dimGrid(chunkWidth / blockSize, chunkWidth / blockSize); // 每个 Stream 的 Grid 大小
    // 注意：每个 Stream 的 Grid 大小相同，因为处理的数据块形状相同（都是 chunkWidth × chunkWidth）

    // =========================================================================
    // 第三步：为每个 Stream 分配独立的 GPU 缓冲区
    // =========================================================================
    float **src_device = new float *[count]; // 输入缓冲区数组（GPU 显存）
    float **tar_device = new float *[count]; // 输出缓冲区数组（GPU 显存）
    for (int i = 0; i < count; i++)
    {
        CUDA_CHECK(cudaMalloc((void **)&src_device[i], chunkSize));
        CUDA_CHECK(cudaMalloc((void **)&tar_device[i], chunkSize));
    }

    // =========================================================================
    // 第四步：创建 count 个 Stream
    // =========================================================================
    cudaStream_t *stream = new cudaStream_t[count];
    for (int i = 0; i < count; i++)
    {
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
    }

    // =========================================================================
    // 第五步：为每个 Stream 提交异步任务
    // =========================================================================
    // 关键设计：
    //   - 每个 Stream 独立处理自己的数据块（通过 offset 偏移）
    //   - 使用异步 API，CPU 可以立即提交下一个 Stream 的任务
    //   - 不同 Stream 的操作在 GPU 上可以重叠执行
    for (int i = 0; i < count; i++)
    {
        // 计算当前 Stream 在 Host 内存中的偏移（单位：float 个数）
        int offset = i * chunkSize / sizeof(float);

        // 5.1 异步拷贝：Host → Device
        //    将第 i 块数据从 src_host 拷贝到 src_device[i]
        CUDA_CHECK(cudaMemcpyAsync(src_device[i], src_host + offset, chunkSize, cudaMemcpyHostToDevice, stream[i]));

        // 5.2 异步启动核函数
        //    处理 src_device[i] 中的数据，结果存入 tar_device[i]
        SleepKernel<<<dimGrid, dimBlock, 0, stream[i]>>>(MAX_ITER * NUM_KERNELS);

        // 5.3 异步拷贝：Device → Host
        //    将第 i 块结果从 tar_device[i] 拷贝回 tar_host + offset
        CUDA_CHECK(cudaMemcpyAsync(tar_host + offset, tar_device[i], chunkSize, cudaMemcpyDeviceToHost, stream[i]));
    }

    // =========================================================================
    // 第六步：等待所有 Stream 完成
    // =========================================================================
    // 注意：必须使用 cudaDeviceSynchronize() 而不是 cudaStreamSynchronize()
    // 因为需要等待所有 Stream 中的所有操作都完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================================================================
    // 第七步：释放 GPU 显存
    // =========================================================================
    for (int i = 0; i < count; i++)
    {
        CUDA_CHECK(cudaFree(src_device[i]));
        CUDA_CHECK(cudaFree(tar_device[i]));
    }
    delete[] src_device;
    delete[] tar_device;

    // =========================================================================
    // 第八步：销毁所有 Stream
    // =========================================================================
    for (int i = 0; i < count; i++)
    {
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }
    delete[] stream;
}
