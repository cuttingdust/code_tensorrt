#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__


extern void MatmulOnHost(float *M_host, float *N_host, float *P_host, int width);
void        MatmulOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize);
void        MatmulSharedOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize, bool staticMem);
void        MatmulSharedConflictOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize,
                                         bool staticMem);
void        MatmulSharedConflictPadOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize,
                                            bool staticMem);

#endif //__MATMUL_HPP__
