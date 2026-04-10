#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    int         deviceCount = 0;
    cudaError_t error       = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-capable device found!\n");
        return 1;
    }

    printf("Found %d CUDA device(s):\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  Device %d: %s\n", i, prop.name);
        printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("    Total Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    printf("CUDA works!\n");
    // printf("CUDA_ENABLED = %d", CUDA_ENABLED);


    return 0;
}
