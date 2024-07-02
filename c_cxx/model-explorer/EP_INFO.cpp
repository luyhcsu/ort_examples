#include <algorithm>  // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "test");

    // 获取 OrtAPI对象
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // 定义变量来存储提供者列表和数量
    char** provider_list = nullptr;
    int provider_length = 0;

    // 调用 GetAvailableProviders 函数
    OrtStatus* status = ort_api->GetAvailableProviders(&provider_list, &provider_length);
    
    if (status != nullptr) {
        const char* msg = ort_api->GetErrorMessage(status);
        std::cerr << "Error: " << msg << std::endl;
        ort_api->ReleaseStatus(status);
        return -1;
    }

    // 打印可用的执行提供者
    std::cout << "Available Execution Providers:" << std::endl;
    for (int i = 0; i < provider_length; ++i) {
        std::cout << provider_list[i] << std::endl;
    }

    // 释放分配的内存
    ort_api->ReleaseAvailableProviders(provider_list, provider_length);

    return 0;
}
