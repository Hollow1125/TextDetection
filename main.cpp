#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>


int main(int argc, char* argv[])
{
    //? cv::dnn::Net net = cv::dnn::readNet("frozen_east_text_detection.pb");
    std::string imagePath = "photo.png";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::imshow("Window", image);
    cv::waitKey(0);
    return 0;
}