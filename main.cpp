#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>


const cv::String keys =
    "{help h        | | Print this message}"
    "{inputImage i  | | Path to input image}"
    "{eastModel e    | | Path to EAST model}"
    "{dbModel d      | | Path to DB model}";

void DB50TextDetection(cv::Mat image, cv::String &modelPath);
void EASTTextDetection(cv::Mat image, cv::String &modelPath);

int main(int argc, char** argv)
{
    setlocale(LC_ALL, "Rus");

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Use this programm to detect paper sheets on an image");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    

    cv::String imagePath = parser.get<cv::String>("inputImage");
    if (imagePath.empty())
    {
        std::cout << "Error: No input image specified" << std::endl;
        return 1;
    }

    cv::Mat image = cv::imread(imagePath);
    if(image.empty())
    {
        std::cout << L"Ошибка загрузки изображения" << std::endl;
        return 1;
    }

    //! Probabilistic HoughTransofrm
    cv::Mat HoughLinesPImage = image.clone();
    //Mat resized, upscaled;
    //pyrDown(HoughLinesPImage, resized, Size(image.cols/2, image.rows/2));
    //pyrUp(resized, upscaled, image.size());
    cv::Mat gray;
    cvtColor(HoughLinesPImage, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150, 3);

    //* Black and white edges
    imwrite("edges.jpg", edges);

    std::vector<cv::Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, 50, 20, 10);

    for(size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];
        cv::line(HoughLinesPImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 2);
    }
    imwrite("houghlinesP.jpg", HoughLinesPImage);

    //! Regular HoughTransofrm
    cv::Mat HoughLinesImage = image.clone();
    std::vector<cv::Vec2f> linesP;
    cv::HoughLines(edges, linesP, 1, CV_PI / 180, 200);  

    // Draw the lines
    for (int i = 0; i < linesP.size(); i++) {
        float rho = linesP[i][0];
        float theta = linesP[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta);
        double b = sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;

        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        line(HoughLinesImage, pt1, pt2, cv::Scalar(0, 0, 255), 1);
    }

    imwrite("HoughLines.jpg", HoughLinesImage);

    
    cv::String eastModelPath = parser.get<cv::String>("eastModel");
    cv::String dbModelPath = parser.get<cv::String>("dbModel");
    if (!eastModelPath.empty())
    {
        EASTTextDetection(image, eastModelPath);
    }
    else
    {
        std::cout << "No path for EAST model have been provided" << std::endl;
    }
    if (!dbModelPath.empty())
    {
        DB50TextDetection(image, dbModelPath);
    }
    else
    {
        std::cout << "No path for DB50 model have been provided" << std::endl;
    }

    return 0;
}

void EASTTextDetection(cv::Mat image, cv::String &modelPath)
{
    try
    {
        cv::dnn::TextDetectionModel_EAST model(modelPath);
        
        float confThreshold = 0.5;
        float nmsThreshold = 0.4;
        model.setConfidenceThreshold(confThreshold)
            .setNMSThreshold(nmsThreshold)
        ;
        
        double detScale = 1.0;
        cv::Size detInputSize = cv::Size(320, 320);
        cv::Scalar detMean = cv::Scalar(123.68, 116.78, 103.94);
        bool swapRB = true;
        model.setInputParams(detScale, detInputSize, detMean, swapRB);

        std::vector<std::vector<cv::Point>> detResults;
        model.detect(image, detResults);

        cv::Mat visualization = image.clone();
        bool rectangleIsClosed = true;
        int lineThickness = 2;
        for (const auto& polygon : detResults) {
            cv::polylines(visualization, polygon, rectangleIsClosed, cv::Scalar(0, 255, 0), lineThickness);
        }
        imwrite("east_detection.jpg", visualization);
    }
    catch (const cv::Exception& e) 
    {
        std::cerr << "EAST Text Detection Error: " << e.what() << std::endl;
    }
}

void DB50TextDetection(cv::Mat image, cv::String& modelPath) 
{
    try {
        cv::dnn::TextDetectionModel_DB model(modelPath);

        float binThresh = 0.3;
        float polyThresh = 0.1;
        uint maxCandidates = 10000;
        double unclipRatio = 1.5;
        model.setBinaryThreshold(binThresh)
            .setPolygonThreshold(polyThresh)
            .setMaxCandidates(maxCandidates)
            .setUnclipRatio(unclipRatio);

        double scale = 1.0 / 255.0;
        cv::Scalar mean = cv::Scalar(122.67891434, 116.66876762, 104.00698793);
        cv::Size inputSize = cv::Size(736, 736);
        model.setInputParams(scale, inputSize, mean);

        std::vector<std::vector<cv::Point>> detResults;
        model.detect(image, detResults);

        cv::Mat visualization = image.clone();
        for (const auto& polygon : detResults) {
            cv::polylines(visualization, polygon, true, cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite("db_detection.jpg", visualization);
    } 
    catch (const cv::Exception& e) 
    {
        std::cerr << "DB Text Detection Error: " << e.what() << std::endl;
    }
}