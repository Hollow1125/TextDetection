#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

const cv::String keys =
    "{help h        | | Print this message}"
    "{inputImage i  | | Path to input image}"
    "{eastModel e    | | Path to EAST model}"
    "{dbModel d      | | Path to DB model}";

void DB50TextDetection(cv::Mat image, cv::String &modelPath, std::string& outputPath);
void EASTTextDetection(cv::Mat image, cv::String &modelPath, std::string& outputPath);
void ProbabilisticHoughTransform(cv::Mat image, cv::String &modelPath);
void HoughTransform(cv::Mat image, cv::String &modelPath);
void CheckDirectoryExists(std::filesystem::path &directory);

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
    cv::String eastModelPath = parser.get<cv::String>("eastModel");
    cv::String dbModelPath = parser.get<cv::String>("dbModel");

    if (imagePath.empty())
    {
        std::wcout << L"Ошибка чтения изображения" << std::endl;
        return 1;
    }

    std::filesystem::path pathToImages(imagePath);
    std::filesystem::path outputDirEAST = pathToImages / "EAST";
    std::filesystem::path outputDirDB = pathToImages / "DB50";
    
    try 
    {
        std::filesystem::create_directory(outputDirEAST);
        std::filesystem::create_directory(outputDirDB);
    } 
    catch (const std::filesystem::filesystem_error& e) 
    {
        std::wcerr << L"Ошибка создания каталога" << e.what() << std::endl;
        return 1;
    }

    try
    {
        for (const auto& i : std::filesystem::recursive_directory_iterator(pathToImages))
        {
            if (i.is_regular_file())
            {
                std::string extension = i.path().extension().string();
                if (extension == ".png" || extension == ".PNG" || extension == ".jpeg" || extension == ".JPEG" || extension == ".jpg" || extension == ".JPG")
                {
                    cv::Mat image = cv::imread(i.path().string());
                    if(image.empty())
                    {
                        std::cout << L"Ошибка загрузки изображения" << std::endl;
                        continue;
                    }
                    
                    std::filesystem::path relativePath = std::filesystem::relative(i.path(), pathToImages);
                    std::filesystem::path parentDir = relativePath.parent_path();

                    std::filesystem::path eastOutputPath = outputDirEAST / parentDir / i.path().filename();
                    std::filesystem::path dbOutputPath = outputDirDB / parentDir / i.path().filename();

                    CheckDirectoryExists(eastOutputPath.parent_path());
                    CheckDirectoryExists(dbOutputPath.parent_path());

                    //ProbabilisticHoughTransform(image);
                    //HoughTransform(image);

                    if (!eastModelPath.empty())
                    {
                        EASTTextDetection(image.clone(), eastModelPath, eastOutputPath.string());
                    }
                    else
                    {
                        std::cout << "No path for EAST model have been provided" << std::endl;
                    }
                    if (!dbModelPath.empty())
                    {
                        DB50TextDetection(image.clone(), dbModelPath, dbOutputPath.string());
                    }
                    else
                    {
                        std::cout << "No path for DB50 model have been provided" << std::endl;
                    }
                }
            }
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }


    return 0;
}

void CheckDirectoryExists(std::filesystem::path &directory)
{
    if (!std::filesystem::exists(directory)) 
    {
        std::filesystem::create_directories(directory);
    }
}

void ProbabilisticHoughTransform(cv::Mat image)
{
    cv::Mat HoughLinesPImage = image.clone();
    //Mat resized, upscaled;
    //pyrDown(HoughLinesPImage, resized, Size(image.cols/2, image.rows/2));
    //pyrUp(resized, upscaled, image.size());
    cv::Mat gray;
    cv::Mat edges;
    cvtColor(HoughLinesPImage, gray, cv::COLOR_BGR2GRAY);
    //imwrite("edges.jpg", edges);
    cv::Canny(gray, edges, 50, 150, 3);

    std::vector<cv::Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, 50, 20, 10);
    for(size_t i = 0; i < lines.size(); i++) 
    {
        cv::Vec4i l = lines[i];
        cv::line(HoughLinesPImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 2);
    }

    imwrite("houghlinesP.jpg", HoughLinesPImage);
}

void HoughTransform(cv::Mat image)
{
    cv::Mat HoughLinesImage = image.clone();
    std::vector<cv::Vec2f> linesP;

    cv::Mat edges;
    cv::HoughLines(edges, linesP, 1, CV_PI / 180, 200);  

    // Draw the lines
    for (int i = 0; i < linesP.size(); i++)
    {
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
}

void EASTTextDetection(cv::Mat image, cv::String &modelPath, std::string& outputPath)
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

        bool rectangleIsClosed = true;
        int lineThickness = 2;
        for (const auto& polygon : detResults) {
            cv::polylines(image, polygon, rectangleIsClosed, cv::Scalar(0, 255, 0), lineThickness);
        }
        imwrite(outputPath, image);
    }
    catch (const cv::Exception& e) 
    {
        std::wcerr << L"Ошибка: " << e.what() << std::endl;
    }
}

void DB50TextDetection(cv::Mat image, cv::String& modelPath, std::string& outputPath) 
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

        for (const auto& polygon : detResults) {
            cv::polylines(image, polygon, true, cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite(outputPath, image);
    } 
    catch (const cv::Exception& e) 
    {
        std::wcerr << L"Ошибка: " << e.what() << std::endl;
    }
}