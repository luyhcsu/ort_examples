import torch
import torch.nn as nn
import torch.onnx

class GEMMModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(GEMMModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.bias = nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        return torch.add(torch.matmul(x, self.weight.t()), self.bias)

# 模型参数
input_size = 3
output_size = 2

# 初始化模型
model = GEMMModel(input_size, output_size)

# 示例输入
x = torch.randn(1, input_size)

# 前向传播
output = model(x)
print("前向传播结果:", output)

# 导出模型到ONNX格式
onnx_file_path = "gemm_model.onnx"
torch.onnx.export(
    model,
    x,  # 模型的示例输入
    onnx_file_path,
    export_params=True,  # 导出模型参数
    opset_version=12,  # ONNX算子集版本
    do_constant_folding=True,  # 常量折叠优化
    input_names=['input'],  # 输入名
    output_names=['output']  # 输出名
)

print(f"模型已导出为 {onnx_file_path}")
