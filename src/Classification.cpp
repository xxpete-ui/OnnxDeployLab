#include "Classification.h"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <opencv2/imgproc.hpp>

namespace
{
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

ClassificationResult parseTop1(const float* scores, std::size_t scoreCount)
{
    if (scores == nullptr || scoreCount == 0)
        throw std::invalid_argument("Classification scores are empty");

    ClassificationResult result{0, scores[0]};
    for (std::size_t i = 1; i < scoreCount; ++i)
    {
        if (scores[i] > result.score)
        {
            result.classId = i;
            result.score = scores[i];
        }
    }
    return result;
}
} // namespace

ClassificationResult classify(
    Ort::Session& session,
    const cv::Mat& image,
    ClassificationTiming& timing)
{
    const auto preprocessStart = std::chrono::steady_clock::now();
    auto inputData = preprocessClassification(image);
    const auto preprocessEnd = std::chrono::steady_clock::now();

    const std::vector<int64_t> runShape = {1, 3, 224, 224};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    // inputData owns the buffer and outlives the synchronous Run.
    auto inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(),
        runShape.data(), runShape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);
    const char* inputNames[] = {inputName.get()};
    const char* outputNames[] = {outputName.get()};

    const auto runStart = std::chrono::steady_clock::now();
    auto outputs = session.Run(
        Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
    const auto runEnd = std::chrono::steady_clock::now();

    auto resultInfo = outputs[0].GetTensorTypeAndShapeInfo();
    if (resultInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        resultInfo.GetShape() != std::vector<int64_t>{1, 28})
        throw std::runtime_error("Expected float32 output with shape [1,28]");

    const float* scores = outputs[0].GetTensorData<float>();
    timing.preprocessMs = std::chrono::duration<double, std::milli>(
        preprocessEnd - preprocessStart).count();
    timing.runMs = std::chrono::duration<double, std::milli>(
        runEnd - runStart).count();

    // Return copied values before outputs is destroyed.
    return parseTop1(scores, resultInfo.GetElementCount());
}
