/***********************************************************************************************************************************************************************************************************************************
In this modified code, the `session_options.AppendExecutionProvider_CUDA(0)` line is added to enable GPU acceleration using the CUDA execution provider from the ONNX Runtime library. 
The `onnxruntime/core/providers/cpu/cpu_provider_factory.h` header is replaced with `onnxruntime/core/providers/cuda/cuda_provider_factory.h` to include the necessary CUDA provider header.
Note that this code assumes you have a CUDA-capable GPU and the necessary CUDA libraries installed on your system. If you don't have a CUDA-capable GPU or the required libraries, you may encounter errors when running this code.
************************************************************************************************************************************************************************************************************************************/
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>
#include <iostream>
#include <vector>

// Draw bounding box
void drawBoundingBox(cv::Mat& image, float conf, int left, int top, int right, int bottom, const std::string& label)
{
    cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2);
    std::string text = label + ": " + std::to_string(conf);
    cv::putText(image, text, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
}

int main()
{
    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv5");
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_CUDA(0); // Use GPU
    Ort::Session session(env, "yolov5s.onnx", session_options);

    // Get input and output names
    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name = session.GetInputName(0, allocator);
    const char* output_name = session.GetOutputName(0, allocator);

    // Open a video capture
    cv::VideoCapture cap(0); // Open default camera
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Preprocess frame
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);

        // Convert blob to tensor
        std::vector<int64_t> input_shape = { 1, 3, 640, 640 };
        size_t input_tensor_size = 1 * 3 * 640 * 640;
        std::vector<float> input_tensor_values(input_tensor_size);
        std::memcpy(input_tensor_values.data(), blob.ptr<float>(), input_tensor_size * sizeof(float));

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());
        std::vector<const char*> input_names = { input_name };
        std::vector<const char*> output_names = { output_name };

        // Perform inference
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), 1);

        // Process output
        auto* raw_output = output_tensors.front().GetTensorMutableData<float>();

        // Example output processing (assuming output format is compatible with YOLOv5)
        for (int i = 0; i < 25200; ++i) // Change 25200 based on output tensor shape
        {
            float conf = raw_output[i * 85 + 4]; // Confidence score
            if (conf > 0.5) // Confidence threshold
            {
                int class_id = std::max_element(raw_output + i * 85 + 5, raw_output + i * 85 + 85) - (raw_output + i * 85 + 5);
                float x_center = raw_output[i * 85 + 0] * frame.cols;
                float y_center = raw_output[i * 85 + 1] * frame.rows;
                float width = raw_output[i * 85 + 2] * frame.cols;
                float height = raw_output[i * 85 + 3] * frame.rows;
                int left = static_cast<int>(x_center - width / 2);
                int top = static_cast<int>(y_center - height / 2);
                int right = static_cast<int>(x_center + width / 2);
                int bottom = static_cast<int>(y_center + height / 2);

                std::string label = "Object"; // Replace with actual class name
                drawBoundingBox(frame, conf, left, top, right, bottom, label);
            }
        }

        // Display the frame
        cv::imshow("YOLOv5 Object Detection", frame);
        if (cv::waitKey(1) >= 0) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}


