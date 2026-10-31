#pragma once

#include <cstddef>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

struct ClassificationResult
{
    std::size_t classId{};
    float score{};
};

struct ClassificationTiming
{
    double preprocessMs{};
    double runMs{};
};

ClassificationResult classify(
    Ort::Session& session,
    const cv::Mat& image,
    ClassificationTiming& timing);
