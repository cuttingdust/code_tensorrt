#include "matmul.hpp"


void MatmulOnHost(float *M, float *N, float *P, int width)
{
    /// i = 行索引（结果矩阵 P 的行）
    for (int i = 0; i < width; i++)
    {
        /// j = 列索引（结果矩阵 P 的列）
        for (int j = 0; j < width; j++)
        {
            float sum = 0; /// 临时累加器

            /// k = 求和索引（M 的列，N 的行）
            for (int k = 0; k < width; k++)
            {
                /// 从 M 中取元素：第 i 行，第 k 列
                float a = M[i * width + k];

                /// 从 N 中取元素：第 k 行，第 j 列
                float b = N[k * width + j];

                /// 累加乘积
                sum += a * b;
            }

            /// 将结果存入 P 的第 i 行，第 j 列
            P[i * width + j] = sum;
        }
    }
}
