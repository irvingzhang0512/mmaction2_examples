
## 8 帧结果

+ 最终分类都是400类
+ 模型推理速度单位是 ms
+ V100 测试结果如下图

|model|fp16|cudnn|fuse_conv_bn|TSN-R50|TSM-MobileNetV2|TSM-R50|TPN-TSM-R50|TIN-R50|TAM|comment|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|ONNX|False|False|False|10-11|5|13|/|/||TPN/TIN ONNX模型转换失败|
|PyTorch|False|False|False|9|7|10|18|21|14|/|
|PyTorch|True|Fasle|False|6|8|11|20|26|18|TIN 需要改代码|
|TensorRT|False|Fasle|False|6|8|11|20|26|18|这一行是 trtexec 工具输出结果|

|model|fp16|cudnn|fuse_conv_bn|I3D|SlowOnly|SlowFast|TPN SlowOnly|x3d|CSN|comment|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|TensorRT|False|False|False|31|88|35|
|TensorRT|True|False|False|9|15|
