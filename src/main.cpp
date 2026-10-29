#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

std::vector<float> preprocessClassification(const cv::Mat& image)
{
    if (image.empty() || image.rows != 224 || image.cols != 224 ||
        image.type() != CV_8UC3)
    {
        throw std::invalid_argument("Expected a 224x224 CV_8UC3 image");
    }

    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
    cv::Mat floatImage;
    rgbImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    const int height = floatImage.rows;
    const int width = floatImage.cols;
    std::vector<float> data(static_cast<std::size_t>(3) * height * width);

    for (int c = 0; c < 3; ++c)
    {
        for (int y = 0; y < height; ++y)
        {
            const cv::Vec3f* row = floatImage.ptr<cv::Vec3f>(y);
            for (int x = 0; x < width; ++x)
            {
                const std::size_t index =
                    static_cast<std::size_t>(c) * height * width +
                    static_cast<std::size_t>(y) * width + x;
                data[index] = row[x][c];
            }
        }
    }
    return data;
}

int main(int argc, char* argv[])
{
    try
    {
        if (argc != 1 && argc != 3)
        {
            std::cerr << "Usage: model_inspect <model.onnx> <image>\n";
            return 1;
        }
        const std::filesystem::path modelPath = argc == 3
            ? std::filesystem::path(argv[1])
            : std::filesystem::path(
                L"D:/Project/OpticalSelectionMachine/OnnxDeployLab/models/cls/hezhuang5.onnx");
        const std::string imagePath = argc == 3 ? argv[2]
            : "D:/Project/OpticalSelectionMachine/OnnxDeployLab/data/cls_test.jpg";

        std::cout << "ONNX Runtime version: "
                  << OrtGetApiBase()->GetVersionString() << '\n';

        // Env is declared first so it outlives Session.
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxDeployLab");
        Ort::SessionOptions options;
        Ort::Session session(env, modelPath.c_str(), options);

        std::cout << "Model loaded successfully!\n"
                  << "Input count: " << session.GetInputCount() << '\n'
                  << "Output count: " << session.GetOutputCount() << '\n';
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
        {
            throw std::runtime_error("Expected one input and one output");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto outputName = session.GetOutputNameAllocated(0, allocator);
        auto inputTypeInfo = session.GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto outputTypeInfo = session.GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

        std::cout << "Input name: " << inputName.get() << "\nInput shape: ";
        for (auto dim : inputTensorInfo.GetShape()) std::cout << dim << ' ';
        std::cout << "\nOutput name: " << outputName.get() << "\nOutput shape: ";
        for (auto dim : outputTensorInfo.GetShape()) std::cout << dim << ' ';
        std::cout << '\n';
        if (inputTensorInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
            throw std::runtime_error("Expected float32 input");
        }

        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Failed to read image: " + imagePath);
        std::cout << "Image width: " << image.cols << '\n'
                  << "Image height: " << image.rows << '\n'
                  << "Image channels: " << image.channels() << '\n';

        std::vector<int64_t> runShape = {1, 3, 224, 224};
        auto inputData = preprocessClassification(image);
        std::cout << "Input elements: " << inputData.size() << '\n'
                  << "Input bytes: " << inputData.size() * sizeof(float) << '\n';

        auto memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        // Tensor borrows inputData; keep the vector alive and unchanged during Run.
        auto inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputData.data(), inputData.size(),
            runShape.data(), runShape.size());
        std::cout << "Input tensor created\n";

        const char* inputNames[] = {inputName.get()};
        const char* outputNames[] = {outputName.get()};
        auto outputs = session.Run(
            Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
            outputNames, 1);
        std::cout << "Inference finished\n";

        auto resultInfo = outputs[0].GetTensorTypeAndShapeInfo();
        const auto resultShape = resultInfo.GetShape();
        std::cout << "Actual output shape: ";
        for (auto dim : resultShape) std::cout << dim << ' ';
        std::cout << '\n';

        if (resultInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            resultShape != std::vector<int64_t>{1, 28})
        {
            throw std::runtime_error("Expected float32 output with shape [1,28]");
        }

        // scores borrows data owned by outputs[0].
        const float* scores = outputs[0].GetTensorData<float>();
        const std::size_t scoreCount = resultInfo.GetElementCount();
        std::cout << "Output elements: " << scoreCount << '\n';
        for (std::size_t i = 0; i < scoreCount; ++i)
            std::cout << "Class " << i << ": " << scores[i] << '\n';

        std::size_t bestClassId = 0;
        float bestScore = scores[0];
        for (std::size_t i = 1; i < scoreCount; ++i)
        {
            if (scores[i] > bestScore)
            {
                bestScore = scores[i];
                bestClassId = i;
            }
        }
        std::cout << "Best class ID: " << bestClassId << '\n'
                  << "Best score: " << bestScore << '\n';
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
