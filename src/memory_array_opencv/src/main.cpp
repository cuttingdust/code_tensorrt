#include <iostream>
#include <opencv2/opencv.hpp>

constexpr auto test_img_file = "assert/lena_hed.jpg";

int main(int argc, char* argv[])
{
    {
        std::cout << "cppds.com" << std::endl;
        /// 400 x 400 的灰度图 做反色 黑色=》白色
        /// 一个字节unsigned char 表示灰度
        /// 255-灰度 做反色
        /// 1 2 3 0   //字节对齐 4字节
        /// 4 5 6 0
        /// 7 8 9 0
        /// 1 2 3 0 4 5 6 0 7 8 9 0

        auto img    = cv::imread(test_img_file, cv::IMREAD_GRAYSCALE);
        int  height = img.rows;
        int  width  = img.cols;

        std::cout << "img.elemSize() = " << img.elemSize() << std::endl;
        cv::imshow("test1", img);

        /// 通过指针访问连续的二维数组
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                auto c                  = img.data[i * width + j];
                img.data[i * width + j] = 255 - c;
            }
        }
        imshow("test2", img);

        ///  等待 5000 毫秒（5 秒）或者直到按下任意键
        int key = cv::waitKey(5000);
        /// 可以根据按下的键执行某些操作
        if (key != -1)
        {
            std::cout << "Key pressed: " << key << std::endl;
        }
    }

    // getchar();
    return 0;
}
