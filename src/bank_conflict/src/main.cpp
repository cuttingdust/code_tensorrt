#include "timer.hpp"
#include "matmul.hpp"
#include "utils.hpp"

int seed;
int main(int argc, char* argv[])
{
    Timer timer;
    int   width     = 1 << 12; /// 4,096
    int   min       = 0;
    int   max       = 1;
    int   size      = width * width;
    int   blockSize = 32;
    bool  statMem   = true;
    char  str[100];


    float* h_matM = (float*)malloc(size * sizeof(float));
    float* h_matN = (float*)malloc(size * sizeof(float));
    float* h_matP = (float*)malloc(size * sizeof(float));
    float* d_matP = (float*)malloc(size * sizeof(float));

    /// seed = (unsigned)time(NULL);
    seed = 1;
    initMatrix(h_matM, size, min, max, seed);
    seed += 1;
    initMatrix(h_matN, size, min, max, seed);

    LOG("Input size is %d x %d", width, width);

    // /// CPU matrix multiplication
    // timer.start_cpu();
    // MatmulOnHost(h_matM, h_matN, h_matP, width);
    // timer.stop_cpu();
    // timer.duration_cpu<Timer::ms>("matmul in cpu");

    /// GPU warmup
    timer.start_gpu();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop_gpu();
    timer.duration_gpu("matmul in gpu(warmup)");

    /// GPU general implementation <<<128, 32>>>
    timer.start_gpu();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);

    /// GPU general implementation <<<128, 32>>>
    timer.start_gpu();
    MatmulSharedOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(shared memory(static))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);

    /// GPU general implementation <<<128, 32>>>
    timer.start_gpu();
    MatmulSharedConflictOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(shared memory(static, bank conf))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);

    /// GPU general implementation <<<128, 32>>>
    timer.start_gpu();
    MatmulSharedConflictPadOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(shared memory(static, pad resolve bank conf))<<<%d, %d>>>", width / blockSize,
                 blockSize);
    timer.duration_gpu(str);

    std::cout << "==========================================" << std::endl;


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// GPU general implementation <<<128, 32>>>
    statMem = false;
    timer.start_gpu();
    MatmulSharedOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(shared memory(dynamic))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);

    /// GPU general implementation <<<128, 32>>>
    statMem = false;
    timer.start_gpu();
    MatmulSharedConflictOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(shared memory(dynamic, bank conf)))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);

    /// GPU general implementation <<<128, 32>>>
    statMem = false;
    timer.start_gpu();
    MatmulSharedConflictPadOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(shared memory(dynamic, pad resolve bank conf))");
    timer.duration_gpu(str);

    return 0;
}
