#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    //? cv::dnn::Net net = cv::dnn::readNet("frozen_east_text_detection.pb");
    std::string imagePath = "E:/Programming/Images/Screenshot_1.png";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    //! DB50
    cv::dnn::TextDetectionModel_DB model("E:/Programming/Cpp_Libraries/DB_TD500_resnet50.onnx");

    float binThresh = 0.3;
    float polyThresh = 0.1;
    uint maxCandidates = 10000;
    double unclipRatio = 1.5;
    model.setBinaryThreshold(binThresh)
        .setPolygonThreshold(polyThresh)
        .setMaxCandidates(maxCandidates)
        .setUnclipRatio(unclipRatio)
    ;

    double scale = 1.0 / 255.0;
    cv::Scalar mean = cv::Scalar(122.67891434, 116.66876762, 104.00698793);
    
    cv::Size inputSize = cv::Size(736, 736);
    
    model.setInputParams(scale, inputSize, mean);
    
    //! EAST
    // cv::dnn::TextDetectionModel_EAST model("E:/Programming/Cpp_Libraries/frozen_east_text_detection.pb");
    
    // float confThreshold = 0.5;
    // float nmsThreshold = 0.4;
    // model.setConfidenceThreshold(confThreshold)
    //     .setNMSThreshold(nmsThreshold)
    // ;
    
    // double detScale = 1.0;
    // cv::Size detInputSize = cv::Size(320, 320);
    // cv::Scalar detMean = cv::Scalar(123.68, 116.78, 103.94);
    // bool swapRB = true;
    // model.setInputParams(detScale, detInputSize, detMean, swapRB);

    std::vector<std::vector<cv::Point>> detResults;
    model.detect(image, detResults);

    cv::Mat visualization = image.clone();
    bool rectangleIsClosed = true;
    int lineThickness = 2;
    for (const auto& polygon : detResults) {
        cv::polylines(visualization, polygon, rectangleIsClosed, cv::Scalar(0, 255, 0), lineThickness);
    }
    
    cv::imshow("Text Detection", visualization);
    cv::waitKey(0);
    return 0;
}