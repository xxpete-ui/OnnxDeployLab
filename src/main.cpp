#include "Classification.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/imgcodecs.hpp>

std::vector<std::string> loadLabels(const std::filesystem::path& labelPath)
{
    std::ifstream file(labelPath);
    if (!file.is_open())
        throw std::runtime_error("Failed to open labels file");

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) throw std::runtime_error("Empty label found");
        labels.push_back(line);
    }
    if (file.bad()) throw std::runtime_error("Failed to read labels file");
    return labels;
}

int main(int argc, char* argv[])
{
    try
    {
        // Preserve the optional model/image arguments from the previous checkpoint.
        if (argc != 1 && argc != 3)
        {
            std::cerr << "Usage: model_inspect [<model.onnx> <224x224 image>]\n";
            return 1;
        }

        std::cout << "ONNX Runtime version: "
                  << OrtGetApiBase()->GetVersionString() << '\n';
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxDeployLab");
        Ort::SessionOptions options;
        const std::filesystem::path modelPath = argc == 3
            ? std::filesystem::path(argv[1])
            : std::filesystem::path(
                L"D:/Project/OpticalSelectionMachine/OnnxDeployLab/models/cls/hezhuang5.onnx");

        const auto loadStart = std::chrono::steady_clock::now();
        Ort::Session session(env, modelPath.c_str(), options);
        const auto loadEnd = std::chrono::steady_clock::now();
        std::cout << "Model load time: "
                  << std::chrono::duration<double, std::milli>(
                         loadEnd - loadStart).count() << " ms\n";

        const auto inputCount = session.GetInputCount();
        const auto outputCount = session.GetOutputCount();
        std::cout << "Model loaded successfully!\n"
                  << "Input count: " << inputCount << '\n'
                  << "Output count: " << outputCount << '\n';
        if (inputCount != 1 || outputCount != 1)
            throw std::runtime_error("Expected one input and one output");

        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto inputTypeInfo = session.GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        std::cout << "Input name: " << inputName.get() << "\nInput shape: ";
        for (auto dim : inputTensorInfo.GetShape()) std::cout << dim << ' ';
        std::cout << '\n';
        if (inputTensorInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Expected float32 input");

        auto outputName = session.GetOutputNameAllocated(0, allocator);
        auto outputTypeInfo = session.GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        std::cout << "Output name: " << outputName.get() << "\nOutput shape: ";
        for (auto dim : outputTensorInfo.GetShape()) std::cout << dim << ' ';
        std::cout << '\n';

        const auto labels = loadLabels(modelPath.parent_path() / "labels.txt");
        if (labels.size() != 28) throw std::runtime_error("Expected 28 labels");
        std::cout << "Label count: " << labels.size() << '\n';

        const std::vector<std::string> imagePaths = argc == 3
            ? std::vector<std::string>{argv[2]}
            : std::vector<std::string>{
                "D:/Project/OpticalSelectionMachine/OnnxDeployLab/data/cls_test.jpg",
                "D:/Project/OpticalSelectionMachine/OnnxDeployLab/data/cls_test2.jpg"};

        for (const auto& imagePath : imagePaths)
        {
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (image.empty())
                throw std::runtime_error("Failed to read image: " + imagePath);

            ClassificationTiming timing;
            const auto start = std::chrono::steady_clock::now();
            ClassificationResult result = classify(session, image, timing);
            const auto end = std::chrono::steady_clock::now();
            const double elapsedMs =
                std::chrono::duration<double, std::milli>(end - start).count();

            std::cout << "Image: "
                      << std::filesystem::path(imagePath).filename().string() << '\n'
                      << "Best class ID: " << result.classId << '\n'
                      << "Best label: " << labels.at(result.classId) << '\n'
                      << "Best score: " << result.score << '\n'
                      << "Preprocess: " << timing.preprocessMs << " ms\n"
                      << "Run: " << timing.runMs << " ms\n"
                      << "Classification: " << elapsedMs << " ms\n\n";
        }
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ORT error: " << e.what() << '\n';
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
