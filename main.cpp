#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <windows.h>
#include <vector>
#include <filesystem>
#include <future>

const cv::String keys =
    "{help h         | | Вывод справки}"
    "{inputImage i   | | Путь к папке с изображениями}"
    "{eastModel e    | | Путь к модели EAST}"
    "{dbModel d      | | Путь к модели DB50}";

//! Функция определения текста с помощью модели EAST
void EASTTextDetection(cv::Mat &image, 
    const cv::String &modelPath, 
    const std::filesystem::path& outputDirEAST, 
    const std::filesystem::path& parentDir, 
    const std::filesystem::directory_entry &i);

//! Функция определения текста с помощью модели DB50
void DB50TextDetection(cv::Mat &image, 
    const cv::String &modelPath, 
    const std::filesystem::path& outputDirDB, 
    const std::filesystem::path& parentDir, 
    const std::filesystem::directory_entry &i);

//! Функции определения краев на фото, не используются пока в программе
void ProbabilisticHoughTransform(cv::Mat &image, const cv::String &modelPath);
void HoughTransform(cv::Mat &image, const cv::String &modelPath);

//! Функция проверки существования и создания папки при итерации по каталогу
void CheckDirectoryExists(const std::filesystem::path &directory);

int main(int argc, char** argv)
{
    SetConsoleOutputCP(CP_UTF8);

    //! Ограничение вывода информации в консоль при компиляции программы
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    //! Класс для передачи аргументов в программу через командную строку
    //! Нужен из-за особенностей передачи параметров в библиотеки OpenCV
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Данная программа определяет наличие текста на изображениях");

    //! Проверка количества аргументов
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

    //! Передаваемые программе аргументы
    cv::String imagePath = parser.get<cv::String>("inputImage");  
    cv::String eastModelPath = parser.get<cv::String>("eastModel");
    cv::String dbModelPath = parser.get<cv::String>("dbModel");

    if (imagePath.empty())
    {
        std::cout << "Ошибка чтения изображения" << std::endl;
        return 1;
    }

    //! Создание папок для хранения обработанных фото в родительском каталоге заданного пути
    std::filesystem::path pathToImages(imagePath);
    std::filesystem::path outputDirEAST = pathToImages.parent_path() / "ImagesProcessedWithEAST";
    std::filesystem::path outputDirDB = pathToImages.parent_path() / "ImagesProcessedWithDB50";

    //! Условия в случае, если путь до моделей не был предоставлен пользователем
    try 
    {
        if (!eastModelPath.empty())
        {
            //! Если путь был передан создает соответствующую папку
            std::filesystem::create_directory(outputDirEAST);
        }
        else
        {
            std::cout << "Модель EAST не найдена" << std::endl;
        }
        if (!dbModelPath.empty())
        {
            //! Если путь был передан создает соответствующую папку
            std::filesystem::create_directory(outputDirDB);
        }
        else
        {
            std::cout << "Модель DB50 не найдена" << std::endl;
        }
    } 
    catch (const std::filesystem::filesystem_error& e) 
    {
        std::cerr << "Ошибка создания каталога: " << e.what() << std::endl;
        return 1;
    }

    try
    {
        //! Цикл для обхода всех папок и файлов внутри заданной пользователем
        for (const auto& i : std::filesystem::recursive_directory_iterator(pathToImages))
        {
            //! Если текущий элемент не каталог, то обрабатывает его
            if (i.is_regular_file())
            {
                //! Преобразование расширения файла в string
                std::string extension = i.path().extension().string();
                //! Проверка того, что текущий элемент - изображение
                if (extension == ".png" || extension == ".PNG" || extension == ".jpeg" || extension == ".JPEG" || extension == ".jpg" || extension == ".JPG")
                {
                    //! Создание матрицы на основе текущего изображения
                    cv::Mat image = cv::imread(i.path().string());
                    if(image.empty())
                    {
                        std::cout << "Ошибка загрузки изображения" << std::endl;
                        continue;
                    }
                    
                    //! Получение родительского пути для создания вложенных каталогов для обработанных изображений
                    //! для сохранения изначальной архитектуры
                    std::filesystem::path relativePath = std::filesystem::relative(i.path(), pathToImages);
                    std::filesystem::path parentDir = relativePath.parent_path();

                    //! Вызов функций определения краев на фото, не адаптирован под текущую архитектуру программы
                    //ProbabilisticHoughTransform(image);
                    //HoughTransform(image);

                    //! Создание future'ов для распараллеливания обработки разными моделями
                    std::future<void> eastFuture;
                    std::future<void> dbFuture;

                    if (!eastModelPath.empty())
                    {
                        //! Вызов функции обработки фото в асинхронном режиме, если путь до модели был передан как аргумент
                        eastFuture = std::async(std::launch::async, 
                            EASTTextDetection, 
                            image.clone(), 
                            eastModelPath, 
                            outputDirEAST, 
                            parentDir, 
                            i);
                    }
                    if (!dbModelPath.empty())
                    {
                        //! Вызов функции обработки фото в асинхронном режиме, если путь до модели был передан как аргумент
                        dbFuture = std::async(std::launch::async, 
                            DB50TextDetection, 
                            image.clone(), 
                            dbModelPath, 
                            outputDirDB, 
                            parentDir, 
                            i);
                    }

                    //! Получение future'ов функций
                    if (!eastModelPath.empty())
                    {
                        eastFuture.get();
                    }
                    if (!dbModelPath.empty()) 
                    {
                        dbFuture.get();
                    }
                }
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        std::cerr << "Ошибка файловой системы: " << e.what() << std::endl;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "Ошибка OpenCV: " << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Ошибка: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Неизвестная ошибка " << std::endl;
    }

    return 0;
}

//! Функция проверки существования и создания папки при итерации по каталогу
void CheckDirectoryExists(const std::filesystem::path &directory)
{
    if (!std::filesystem::exists(directory)) 
    {
        std::filesystem::create_directories(directory);
    }
}

//! Функция для поиска линий на изображении
void ProbabilisticHoughTransform(cv::Mat &image)
{
    //! Создание матрицы для хранения копии изображения
    cv::Mat HoughLinesPImage = image.clone();
    //Mat resized, upscaled;
    //pyrDown(HoughLinesPImage, resized, Size(image.cols/2, image.rows/2));
    //pyrUp(resized, upscaled, image.size());
    //! Создание вспомогательных матриц
    cv::Mat gray;
    cv::Mat edges;

    //! Удаление цветов с изображения для более четкого результата
    cvtColor(HoughLinesPImage, gray, cv::COLOR_BGR2GRAY);
    //imwrite("edges.jpg", edges);
    //! Функция для определения краев на изображении
    double thresholdLow = 50;
    double thresholdHigh = 150;
    int apertureSize = 3;
    cv::Canny(gray, edges, thresholdLow, thresholdHigh, apertureSize);

    //! Вектор для хранения найденных линий
    std::vector<cv::Vec4i> lines;

    //! Функция для определения того, являются ли найденные края линиями
    double rho = 1;
    double theta = CV_PI/180;
    int threshold = 50;
    double minLineLength = 20;
    double maxLineGap = 10;
    HoughLinesP(edges, lines, rho, theta, threshold, minLineLength, maxLineGap);

    //! Цикл для выделения линий на фото
    for(size_t i = 0; i < lines.size(); i++) 
    {
        cv::Vec4i l = lines[i];
        cv::line(HoughLinesPImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 2);
    }

    //! Запись результата в отдельный файл 
    imwrite("houghlinesP.jpg", HoughLinesPImage);
}

//! Менее точный аналог предыдущей функции
//TODO: надо проверить будет ли эта функция достоверно определять линии если изменить перспективу изображения
//TODO: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
void HoughTransform(cv::Mat &image)
{
    cv::Mat HoughLinesImage = image.clone();
    std::vector<cv::Vec2f> linesP;

    cv::Mat edges;
    cv::HoughLines(edges, linesP, 1, CV_PI / 180, 200);  

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

//! Функция определения наличия текста моделью EAST
void EASTTextDetection(cv::Mat &image, 
    const cv::String &modelPath, 
    const std::filesystem::path& outputDirEAST, 
    const std::filesystem::path& parentDir, 
    const std::filesystem::directory_entry &i)
{
    try
    {
        //! Загрузка модели
        cv::dnn::TextDetectionModel_EAST model(modelPath);
        
        //! Обход файлов и создание каталогов
        std::filesystem::path  eastOutputPath = outputDirEAST / parentDir / i.path().filename();
        CheckDirectoryExists(eastOutputPath.parent_path());

        //! Задание порогов чувтсивтельности и достоверности определения текста
        float confThreshold = 0.5;
        float nmsThreshold = 0.4f;
        model.setConfidenceThreshold(confThreshold)
            .setNMSThreshold(nmsThreshold)
        ;
        
        //! Масштабирование
        double detScale = 1.0;
        //! Изменение размера изображения в пикселях
        cv::Size detInputSize = cv::Size(736, 736);
        //! Средние значения палитры BGR
        cv::Scalar detMean = cv::Scalar(122.67891434, 116.66876762, 104.00698793);
        //! Смена местами красного и синего каналов, так как OpenCV работает в формате BGR
        bool swapRB = true;
        model.setInputParams(detScale, detInputSize, detMean, swapRB);

        //! Вектор для хранения результатов
        std::vector<std::vector<cv::Point>> detResults;
        model.detect(image, detResults);

        //! Выделение определенного текста прямоугольниками
        bool rectangleIsClosed = true;
        int lineThickness = 2;
        for (const auto& polygon : detResults) {
            cv::polylines(image, polygon, rectangleIsClosed, cv::Scalar(0, 255, 0), lineThickness);
        }
        imwrite(eastOutputPath.string(), image);
    }
    catch (const cv::Exception& e) 
    {
        std::cerr << "Ошибка: " << e.what() << std::endl;
    }
}

void DB50TextDetection(cv::Mat &image, 
    const cv::String &modelPath, 
    const std::filesystem::path& outputDirDB, 
    const std::filesystem::path& parentDir, 
    const std::filesystem::directory_entry &i)
{
    try {
        cv::dnn::TextDetectionModel_DB model(modelPath);

        std::filesystem::path  dbOutputPath = outputDirDB / parentDir / i.path().filename();
        CheckDirectoryExists(dbOutputPath.parent_path());

        float binThresh = 0.3f;
        float polyThresh = 0.1f;
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
        int lineThickness = 2;

        for (const auto& polygon : detResults) {
            cv::polylines(image, polygon, true, cv::Scalar(0, 255, 0), lineThickness);
        }
        cv::imwrite(dbOutputPath.string(), image);
    } 
    catch (const cv::Exception& e) 
    {
        std::cerr << "Ошибка: " << e.what() << std::endl;
    }
}