#include "timer.hpp"
#include "matmul.hpp"
#include "utils.hpp"
#include "stream.hpp"

int  seed;
void sleep_test()
{
    Timer timer;
    int   width = 1 << 11; ///  2048 (16MB) 或 4096 (64MB)
    int   size  = width * width;

    float low  = -1.0;
    float high = 1.0;

    int  blockSize = 32;
    int  taskCnt   = 16;
    bool statMem   = true;
    char str[100];

    /// 初始化 固定内存 (页锁定内存) 以加速主机与设备之间的数据传输
    float* src_host;
    float* tar_host;
    cudaMallocHost(&src_host, size * sizeof(float));
    cudaMallocHost(&tar_host, size * sizeof(float));

    seed += 1;
    initMatrixSigned(src_host, size, low, high, seed);
    LOG("Input size is %d", size);

    /// GPU warmup
    timer.start_gpu();
    SleepSingleStream(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    timer.duration_gpu("matmul in gpu(warmup)");

    std::cout << "===============================================================================" << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// 1 stream，处理一次memcpy，以及n个kernel
    timer.start_gpu();
    SleepSingleStream(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "SleepSingleStream <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel",
                 width / blockSize, width / blockSize, blockSize, blockSize, 1, 1, taskCnt);
    timer.duration_gpu(str);

    //////////////////////////////////////////////////////////////////
    timer.start_gpu();
    SingleStreamMultiKernel(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "SingleStreamMultiKernel <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel",
                 width / blockSize, width / blockSize, blockSize, blockSize, 1, 1, taskCnt);
    timer.duration_gpu(str);

    //////////////////////////////////////////////////////////////////
    timer.start_gpu();
    SingleStreamMixedKernel(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "SingleStreamMixedKernel <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel",
                 width / blockSize, width / blockSize, blockSize, blockSize, 1, 1, taskCnt);
    timer.duration_gpu(str);
    //////////////////////////////////////////////////////////////////

    timer.start_gpu();
    SingleStreamInterleaved(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "SingleStreamInterleaved <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel",
                 width / blockSize, width / blockSize, blockSize, blockSize, 1, 1, taskCnt);
    timer.duration_gpu(str);

    //////////////////////////////////////////////////////////////////

    timer.start_gpu();
    SingleStreamSmallBig(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "SingleStreamSmallBig <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel",
                 width / blockSize, width / blockSize, blockSize, blockSize, 1, 1, taskCnt);
    timer.duration_gpu(str);

    //////////////////////////////////////////////////////////////////

    timer.start_gpu();
    MultiStreamSeparate(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "MultiStreamSeparate <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel",
                 width / blockSize, width / blockSize, blockSize, blockSize, 1, 1, taskCnt);
    timer.duration_gpu(str);

    /// n stream，处理一次memcpy，以及n个kernel
    timer.start_gpu();
    SleepMultiStream(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "sleep <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel", width / blockSize,
                 width / blockSize, blockSize, blockSize, taskCnt, 1, taskCnt);
    timer.duration_gpu(str);

    ////////////////////////////////////////////////////////////////////////////////////////////////////
}

void gelu_test()
{
    /*
     * 大家试着在这里对gelu计算做一个多流的计算看看整体延迟的改变
     * 可以观测到相比于memcpy的计算, kernel的延迟会很小
    */
}

int main(int argc, char* argv[])
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    /// 需要先确认自己的GPU是否支持overlap计算
    if (!prop.deviceOverlap)
    {
        LOG("device does not support overlap");
    }
    else
    {
        LOG("device supports overlap");
    }

    sleep_test();
    gelu_test();

    // 这里供大家自由发挥。建议花一些在这里做调度的练习。根据ppt里面的方案实际编写几个测试函数。举几个例子在这里
    // e.g. 一个stream处理: H2D, 多个kernel，D2H。之后多个stream进行overlap
    // e.g. 一个stream处理: H2D, 大kernel，小kernel, D2H。之后多个stream进行overlap
    // e.g. 一个stream处理: H2D, 大kernel, H2D, 小kernel, D2H。之后多个stream进行overlap
    // e.g. 一个stream处理: H2D, 小kernel, H2D, 大kernel, D2H。之后多个stream进行overlap
    // e.g. 一个stream处理: H2D, 另外几个流分别只处理kernel, 和D2H。之后所有stream进行overlap
    // e.g. 一个stream处理: H2D(局部), kernel(局部), D2H(局部)。之后所有stream进行overlap

    return 0;
}
