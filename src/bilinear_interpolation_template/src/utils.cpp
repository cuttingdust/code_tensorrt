#include "utils.hpp"
#include <math.h>
#include <random>


void initMatrix(float* data, int size, int min, int max, int seed)
{
    srand(seed);
    for (int i = 0; i < size; i++)
    {
        data[i] = min + float(rand()) * float(max - min) / RAND_MAX;
    }
}

void initMatrixSigned(float* data, int size, float low, float high, int seed)
{
    std::default_random_engine            generator(seed);
    std::uniform_real_distribution<float> distribution(low, high);
    for (int i = 0; i < size; i++)
    {
        data[i] = distribution(generator);
    }
}


void printMat(float* data, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%.8lf", data[i]);
        if (i != size - 1)
        {
            printf(", ");
        }
        else
        {
            printf("\n");
        }
    }
}


void compareMat(float* h_data, float* d_data, int size)
{
    double precision = 1.0E-4;
    bool   error     = false;
    /* 
     * 这里注意，浮点数运算时CPU和GPU之间的计算结果是有误差的
     * 一般来说误差保持在1.0E-4之内是可以接受的
    */
    for (int i = 0; i < size; i++)
    {
        if (abs(h_data[i] - d_data[i]) > precision)
        {
            error = true;
            printf("res is different in %d, cpu: %.8lf, gpu: %.8lf\n", i, h_data[i], d_data[i]);
            break;
        }
    }
    if (error)
        printf("Matmul result is different\n");
    else
        printf("Matmul result is same, precision is 1.0E-4\n");
}

std::string getPrefix(std::string file_path)
{
    std::string prefix;
    int         pos1 = file_path.find('/');
    int         pos2 = file_path.rfind('.');
    prefix           = file_path.substr(pos1, pos2 - pos1);
    return prefix;
}
