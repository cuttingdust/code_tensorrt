#ifndef __STREAM_HPP__
#define __STREAM_HPP__

void SleepSingleStream(float* src_host, float* tar_host, int width, int blockSize, int count);
void SleepMultiStream(float* src_host, float* tar_host, int width, int blockSize, int count);


void SingleStreamMultiKernel(float* src_host, float* tar_host, int width, int blockSize, int count);
void SingleStreamMixedKernel(float* src_host, float* tar_host, int width, int blockSize, int count);
void SingleStreamInterleaved(float* src_host, float* tar_host, int width, int blockSize, int count);
void SingleStreamSmallBig(float* src_host, float* tar_host, int width, int blockSize, int count);
void MultiStreamSeparate(float* src_host, float* tar_host, int width, int blockSize, int streamsCount);
#endif // __STREAM_HPP__
