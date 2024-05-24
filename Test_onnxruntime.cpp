#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() 
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_CUDA(0); // Use CUDA

    Ort::Session session(env, "model.onnx", session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name = session.GetInputName(0, allocator);
    const char* output_name = session.GetOutputName(0, allocator);

    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    std::vector<float> input_tensor_values(1 * 3 * 224 * 224);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    std::vector<const char*> input_names = {input_name};
    std::vector<const char*> output_names = {output_name};

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
    auto* output_data = output_tensors.front().GetTensorMutableData<float>();

    std::cout << "Output: " << output_data[0] << std::endl;

    return 0;
}
