#include <iostream>
#include <vector>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include "/home/kylin/luyh/opencl/onnxruntime/include/onnxruntime/core/providers/opencl/opencl_provider_factory.h"


// 定义一个函数来创建并初始化 ONNX Runtime Tensor
Ort::Value CreateTensor(const std::vector<float>& data, const std::vector<int64_t>& shape, Ort::MemoryInfo& memory_info) {
    return Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(data.data()), data.size(), shape.data(), shape.size());
}

int main() {
    // 初始化 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "test");

    // 创建 ONNX Runtime 会话选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);  // 设置 IntraOp 线程数为 1
    session_options.SetInterOpNumThreads(1);  // 设置 InterOp 线程数为 1
    session_options.SetExecutionMode(ORT_SEQUENTIAL);
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_OpenCL(session_options, 0, 0);
    if (status != nullptr) {
      const char* msg = Ort::GetApi().GetErrorMessage(status);
      std::cerr << "Error setting OpenCL Execution Provider: " << msg << std::endl;
      Ort::GetApi().ReleaseStatus(status);
      return -1;
    }
    // 加载 ONNX 模型
    const char* model_path = "gemm_model_fixed.onnx";  // 模型文件路径
    Ort::Session session(env, model_path, session_options);

    // 获取输入输出信息
    // Ort::AllocatorWithDefaultOptions allocator;
    // const char* input_name_A = session.GetInputName(0, allocator);
    // const char* input_name_B = session.GetInputName(1, allocator);
    // const char* output_name = session.GetOutputName(0, allocator);

    // std::cout << "Input name A: " << input_name_A << std::endl;
    // std::cout << "Input name B: " << input_name_B << std::endl;
    // std::cout << "Output name: " << output_name << std::endl;

    // // 创建示例输入数据
    // std::vector<float> input_A = {0.5, -0.2, 1.0, 0.7, -1.2, 2.0, -0.5, 0.3, 1.3, 0.8, -0.1, 1.1, 0.8, 0.9, 1.0, -0.6};  // 3x4 矩阵
    // std::vector<float> input_B = {0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -0.6};  // 4x5 矩阵

    // // 创建内存信息
    // Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // // 创建输入张量
    // Ort::Value input_tensor_A = CreateTensor(input_A, {4, 4}, memory_info);  // 输入 A 的形状为 {3, 4}
    // Ort::Value input_tensor_B = CreateTensor(input_B, {4, 4}, memory_info);  // 输入 B 的形状为 {4, 5}

    // // 执行推理
    // const char* input_names[] = {input_name_A, input_name_B};  // 输入名称数组
    // const char* output_names[] = {output_name};  // 输出名称数组

    // std::vector<Ort::Value> input_tensors;
    // input_tensors.push_back(std::move(input_tensor_A));
    // input_tensors.push_back(std::move(input_tensor_B));

    // std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), input_tensors.size(), output_names, 1);

    // // 获取输出结果
    // float* output_data = output_tensors[0].GetTensorMutableData<float>();
    // std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    // std::cout << "Output shape: ";
    // for (size_t i = 0; i < output_shape.size(); ++i) {
    //     std::cout << output_shape[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Output data: ";
    // //Output data: 0.11 0.25 1.49 0.54 1.32 0.08 -0.76 -1.61 1.45 1.22 1.27 -0.45 -0.35 -0.97 0.06 1.11
    // for (size_t i = 0; i < output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
    //     std::cout << output_data[i] << " ";
    // }
    // std::cout << std::endl;

    return 0;
}
