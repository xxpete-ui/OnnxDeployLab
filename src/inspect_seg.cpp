#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

int main()
{
    try
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InspectSeg");
        Ort::SessionOptions options;
        const wchar_t* modelPath =
            L"D:/Project/OpticalSelectionMachine/"
            L"OnnxDeployLab/models/seg/hei_huizong_250428.onnx";
        Ort::Session session(env, modelPath, options);
        Ort::AllocatorWithDefaultOptions allocator;

        std::cout << "Input count: " << session.GetInputCount() << '\n';
        for (std::size_t i = 0; i < session.GetInputCount(); ++i)
        {
            auto name = session.GetInputNameAllocated(i, allocator);
            auto typeInfo = session.GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            std::cout << "Input " << i << ": " << name.get() << '\n'
                      << "Element type: "
                      << static_cast<int>(tensorInfo.GetElementType())
                      << "\nShape: ";
            for (auto dim : tensorInfo.GetShape()) std::cout << dim << ' ';
            std::cout << '\n';
        }

        std::cout << "Output count: " << session.GetOutputCount() << '\n';
        for (std::size_t i = 0; i < session.GetOutputCount(); ++i)
        {
            auto name = session.GetOutputNameAllocated(i, allocator);
            auto typeInfo = session.GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            std::cout << "Output " << i << ": " << name.get() << '\n'
                      << "Element type: "
                      << static_cast<int>(tensorInfo.GetElementType())
                      << "\nShape: ";
            for (auto dim : tensorInfo.GetShape()) std::cout << dim << ' ';
            std::cout << '\n';
        }
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 2)
            throw std::runtime_error("Expected one input and two outputs");
        auto inputTypeInfo = session.GetInputTypeInfo(0);
        if (inputTypeInfo.GetTensorTypeAndShapeInfo().GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Expected float32 input");

        const std::string imagePath =
            "D:/Project/OpticalSelectionMachine/"
            "OnnxDeployLab/data/seg_test.jpg";
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty())
            throw std::runtime_error("Failed to read image: " + imagePath);
        std::cout << "Image width: " << image.cols << '\n'
                  << "Image height: " << image.rows << '\n'
                  << "Image channels: " << image.channels() << '\n';

        // LetterBox geometry is kept for the future original-image mask mapping.
        const int inputWidth = 640;
        const int inputHeight = 640;
        const double scale = std::min(
            static_cast<double>(inputWidth) / image.cols,
            static_cast<double>(inputHeight) / image.rows);
        const int resizedWidth = static_cast<int>(std::round(image.cols * scale));
        const int resizedHeight = static_cast<int>(std::round(image.rows * scale));
        if (resizedWidth <= 0 || resizedHeight <= 0)
            throw std::runtime_error("Image is too narrow for this LetterBox input");
        const int padLeft = (inputWidth - resizedWidth) / 2;
        const int padRight = inputWidth - resizedWidth - padLeft;
        const int padTop = (inputHeight - resizedHeight) / 2;
        const int padBottom = inputHeight - resizedHeight - padTop;

        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(resizedWidth, resizedHeight),
                   0.0, 0.0, cv::INTER_LINEAR);
        cv::Mat letterboxImage;
        cv::copyMakeBorder(resizedImage, letterboxImage,
                           padTop, padBottom, padLeft, padRight,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        const std::string letterboxPath =
            "D:/Project/OpticalSelectionMachine/"
            "OnnxDeployLab/data/seg_letterbox.jpg";
        if (!cv::imwrite(letterboxPath, letterboxImage))
            throw std::runtime_error("Failed to save LetterBox image");
        std::cout << "Scale: " << scale << '\n'
                  << "Resized: " << resizedWidth << " x " << resizedHeight << '\n'
                  << "Padding L/R/T/B: " << padLeft << ' ' << padRight << ' '
                  << padTop << ' ' << padBottom << '\n';

        cv::Mat rgbImage;
        cv::cvtColor(letterboxImage, rgbImage, cv::COLOR_BGR2RGB);
        cv::Mat floatImage;
        rgbImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);
        const std::vector<int64_t> runShape = {1, 3, inputHeight, inputWidth};
        std::vector<float> inputData(
            static_cast<std::size_t>(3) * inputHeight * inputWidth);
        for (int c = 0; c < 3; ++c)
        {
            for (int y = 0; y < inputHeight; ++y)
            {
                const cv::Vec3f* row = floatImage.ptr<cv::Vec3f>(y);
                for (int x = 0; x < inputWidth; ++x)
                {
                    const std::size_t index =
                        static_cast<std::size_t>(c) * inputHeight * inputWidth +
                        static_cast<std::size_t>(y) * inputWidth + x;
                    inputData[index] = row[x][c];
                }
            }
        }

        auto memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        auto inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputData.data(), inputData.size(),
            runShape.data(), runShape.size());
        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto outputName0 = session.GetOutputNameAllocated(0, allocator);
        auto outputName1 = session.GetOutputNameAllocated(1, allocator);
        const char* inputNames[] = {inputName.get()};
        const char* outputNames[] = {outputName0.get(), outputName1.get()};
        auto outputs = session.Run(
            Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 2);

        for (std::size_t i = 0; i < outputs.size(); ++i)
        {
            auto info = outputs[i].GetTensorTypeAndShapeInfo();
            std::cout << "Actual output " << i << " shape: ";
            for (auto dim : info.GetShape()) std::cout << dim << ' ';
            std::cout << "\nElements: " << info.GetElementCount() << '\n';
        }

        auto predictionInfo = outputs[0].GetTensorTypeAndShapeInfo();
        const auto predictionShape = predictionInfo.GetShape();
        if (predictionInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            predictionShape.size() != 3 || predictionShape[0] != 1 ||
            predictionShape[1] != 37 || predictionShape[2] <= 0)
            throw std::runtime_error("Unexpected prediction output");

        // [1,37,N]: shape[2] is N, not shape[1].
        const std::size_t candidateCount =
            static_cast<std::size_t>(predictionShape[2]);
        const float* predictions = outputs[0].GetTensorData<float>();

        struct SegCandidate
        {
            std::size_t predictionIndex{};
            float score{};
            cv::Rect2d networkBox;
        };

        const float confidenceThreshold = 0.25f;
        std::vector<SegCandidate> candidates;
        for (std::size_t i = 0; i < candidateCount; ++i)
        {
            const float score = predictions[4 * candidateCount + i];
            if (score < confidenceThreshold) continue;
            const float cx = predictions[0 * candidateCount + i];
            const float cy = predictions[1 * candidateCount + i];
            const float w = predictions[2 * candidateCount + i];
            const float h = predictions[3 * candidateCount + i];
            if (w <= 0.0f || h <= 0.0f) continue;

            SegCandidate candidate;
            candidate.predictionIndex = i;
            candidate.score = score;
            candidate.networkBox = cv::Rect2d(
                cx - w / 2.0, cy - h / 2.0, w, h);
            candidates.push_back(candidate);
        }

        std::cout << "Candidates before filtering: " << candidateCount << '\n'
                  << "Candidates above threshold: " << candidates.size() << '\n';
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        boxes.reserve(candidates.size());
        scores.reserve(candidates.size());
        for (const auto& candidate : candidates)
        {
            boxes.push_back(candidate.networkBox);
            scores.push_back(candidate.score);
        }

        const float nmsThreshold = 0.45f;
        std::vector<int> keptIndices;
        if (!boxes.empty())
            cv::dnn::NMSBoxes(boxes, scores, confidenceThreshold,
                              nmsThreshold, keptIndices);
        std::cout << "Candidates after NMS: " << keptIndices.size() << '\n';

        std::vector<SegCandidate> detections;
        detections.reserve(keptIndices.size());
        for (int index : keptIndices)
            detections.push_back(candidates.at(static_cast<std::size_t>(index)));
        for (const auto& detection : detections)
            std::cout << "Kept prediction index: " << detection.predictionIndex
                      << ", score: " << detection.score << '\n';

        auto protoInfo = outputs[1].GetTensorTypeAndShapeInfo();
        const auto protoShape = protoInfo.GetShape();
        if (protoInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            protoShape.size() != 4 || protoShape[0] != 1 || protoShape[1] != 32 ||
            protoShape[2] <= 0 || protoShape[3] <= 0)
            throw std::runtime_error("Unexpected proto output");

        const std::size_t maskChannels = static_cast<std::size_t>(protoShape[1]);
        for (const auto& detection : detections)
        {
            std::vector<float> maskCoefficients(maskChannels);
            for (std::size_t k = 0; k < maskChannels; ++k)
            {
                const std::size_t channel = 5 + k;
                maskCoefficients[k] = predictions[
                    channel * candidateCount + detection.predictionIndex];
            }

            const int protoHeight = static_cast<int>(protoShape[2]);
            const int protoWidth = static_cast<int>(protoShape[3]);
            const std::size_t planeSize =
                static_cast<std::size_t>(protoHeight) * protoWidth;
            // outputs stays alive throughout this loop.
            const float* proto = outputs[1].GetTensorData<float>();
            cv::Mat maskProbability(protoHeight, protoWidth, CV_32FC1);

            for (int y = 0; y < protoHeight; ++y)
            {
                float* probabilityRow = maskProbability.ptr<float>(y);
                for (int x = 0; x < protoWidth; ++x)
                {
                    const std::size_t pixelIndex =
                        static_cast<std::size_t>(y) * protoWidth + x;
                    float logit = 0.0f;
                    for (std::size_t k = 0; k < maskChannels; ++k)
                        logit += maskCoefficients[k] * proto[k * planeSize + pixelIndex];

                    if (logit >= 0.0f)
                        probabilityRow[x] = 1.0f / (1.0f + std::exp(-logit));
                    else
                    {
                        const float expValue = std::exp(logit);
                        probabilityRow[x] = expValue / (1.0f + expValue);
                    }
                }
            }

            // Intermediate preview only: no box crop, depadding, original-size
            // restoration or binary threshold has been implemented yet.
            cv::Mat maskPreview;
            maskProbability.convertTo(maskPreview, CV_8U, 255.0);
            const std::string maskPath =
                "D:/Project/OpticalSelectionMachine/OnnxDeployLab/data/"
                "seg_mask_proto_" + std::to_string(detection.predictionIndex) + ".png";
            if (!cv::imwrite(maskPath, maskPreview))
                throw std::runtime_error("Failed to save proto mask");
            std::cout << "Saved proto mask: " << maskPath << '\n';
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
