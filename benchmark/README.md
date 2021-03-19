# 测试模型的实际 inference time

## 0. 前言

+ 需求概述
    + 需要对比的模型有：I3D、TSM、TSN、C3D、CSN、TIN、TPN、X3D
    + 模型inference的代码环境有：
        + Python 环境：PyTorch，PyTorch FP16，ONNXRUNTIME，TensorRT，TensorRT FP16
        + C++ 环境：PyTorch C++，TensorRT，TensorRT FP16
    + 测试内容：模型推理 inference time，加上io的inference time
    + 输出形式：测试结果要绘制成表格

## Python 测试

+ 模块
    + 模型相关：测试模型推理所需的内容，包括 PyTorch、PyTorch FP16、ONNXRUNTIME、TensorRT、TnesorRT FP16
    + 测试过程相关，包括参数以及输入形式：参考trtexec，包括iteration/warmu等
    + 输出模式：支持多种形式的输出，先支持文本
+ TODO
    + 测试内容：
        + [x] 仅模型推理，不包括 IO
        + [ ] 模型推理以及IO
    + 数据输入形式：
        + [x] 随机输入
        + [ ] MMAction2 数据集输入
        + [ ] 单个本地文件/Webcam输入
    + 待测试的模型类型：
        + [x] PyTorch
        + [x] ONNX
        + [ ] TensorRT
    + 具体模型
        + [ ] TSN
        + [ ] TSM
        + [ ] TIN
        + [ ] TAM
        + [ ] TPN
        + [ ] I3D
        + [ ] R(2+1)D
        + [ ] SlowOnly
        + [ ] SlowFast
        + [ ] X3D
    + 输出结果形式
        + [ ] 文本
        + [ ] 表格


## C++ 测试

