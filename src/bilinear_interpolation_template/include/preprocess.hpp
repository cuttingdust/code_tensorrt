#pragma once
#define NOMINMAX
#include <opencv2/opencv.hpp>

#include "timer.hpp"

cv::Mat preprocess_cpu(cv::Mat &src, const int &tar_h, const int &tar_w, Timer timer, int tactis);

template <typename T>
cv::Mat preprocess_gpu(cv::Mat &h_src, const int &tar_h, const int &tar_w, Timer timer);

template <typename T>
void resize_bilinear_gpu(T *d_tar, uint8_t *d_src, int tarW, int tarH, int srcW, int srcH);
