import torch
import torch.nn as nn
import torch.onnx

# 定义一个简单的GEMM模型
class GemmModel(nn.Module):
    def __init__(self):
        super(GemmModel, self).__init__()
    
    def forward(self, A, B):
        return torch.matmul(A, B)

# 实例化模型
model = GemmModel()

# 创建示例输入张量
A = torch.randn(4, 4)  # 3x4矩阵，批次大小为3
B = torch.randn(4, 4)  # 4x5矩阵

# 设置模型为评估模式
model.eval()

# 导出模型为ONNX格式
torch.onnx.export(model,               # 要转换的模型
                  (A, B),              # 模型输入张量
                  "gemm_model_fixed.onnx",   # 导出的ONNX文件名
                  export_params=True,  # 导出训练好的参数
                  opset_version=11,    # ONNX的算子集版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input_A', 'input_B'],  # 输入的名字
                  output_names=['output'])  # 输出的名字

print("模型已成功导出为ONNX格式（固定批次大小）")
