# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_executable(model-explorer model-explorer.cpp)
target_link_libraries(model-explorer PRIVATE onnxruntime)

add_executable(matmul-case matmul-case.cpp)
target_link_libraries(matmul-case PRIVATE onnxruntime)

add_executable(EP_INFO EP_INFO.cpp)
target_link_libraries(EP_INFO PRIVATE onnxruntime)
#TODO: fix the build error
#add_executable(batch-model-explorer batch-model-explorer.cpp)
#target_link_libraries(batch-model-explorer PRIVATE onnxruntime)