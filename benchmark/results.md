
## 8 帧结果

+ 最终分类都是400类
+ 模型推理速度单位是 ms
+ V100 测试结果如下图

|model|fp16|cudnn|fuse_conv_bn|TSN-R50|TSM-MobileNetV2|TSM-R50|TPN-TSM-R50|TIN-R50|comment|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|ONNX|False|False|False|10-11|5|13|/|/|TPN/TIN ONNX模型转换失败|
|PyTorch|False|False|False|12|10-11|13|21-22|26-27|/|
|PyTorch|True|Fasle|False|9|10-11|12-13|17-18|27-28|TIN 需要改代码|
|PyTorch|False|True|False|12-13|10-11|13-14|17-18|26-27|/|
|PyTorch|False|False|True|10-11|7-8|12-13|20-21|24-25|/|