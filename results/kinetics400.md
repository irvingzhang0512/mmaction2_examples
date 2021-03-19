# Kinetics 400 训练结果

## 0. 前言

+ Kinetics400 数据集的版本很多，各种论文中数据都不统一。我目前使用的是 [Academic Torrents](https://academictorrents.com/details/184d11318372f70018cf9a72ef867e2fb9ce1d26)
    + 值得一提的是，GluonCV 中的行为识别模型也用的是上述数据集，因此其结果可作为参考，详情请参考 [这里](https://cv.gluon.ai/model_zoo/action_recognition.html#kinetics400-dataset)
+ 测试时数据预处理
    + 模型在验证集上测试分为 efficient mode 和 accurate mode
    + 前者一般用在模型训练过程中，即 val pipeline
    + 后者一般用在训练完后，即 test pipeline
+ 一些默认参数
    + 训练 100 epochs


## 1. TSN

+ 测试使用的 pipeline

|model|efficient mode|accurate mode|
|:-:|:-:|:-:|
|tsn_r50_1x1x8|CenterCrop 224|25 clips + ThreeCrop 256|

+ 训练结果

|model|configs|acc1/acc5 efficient mode|acc1/acc5 accurate mode|
|:-:|:-|:-:|:-:|
|tsn_r50_1x1x7|GluonCV|66.97 / x||
|tsn_r50_1x1x8|default config|70.74 / 89.37||
|tsn_r50_1x1x8|randaugment|71.07 / 89.40||
|tsn_r50_1x1x8|cutmix alpha=1.0|70.58 / 89.56||
|tsn_r50_1x1x8|mixup alpha=1.0|71.07 / 89.40||

## 2. TSM

+ 测试使用的 pipeline

|model|efficient mode|accurate mode|
|:-:|:-:|:-:|
|tsm_r50_1x1x8|CenterCrop 224|`twice_samples=True` + ThreeCrop 256|
|tsm_mobilenet_v2_1x1x8_dense|10 clips + CenterCrop 224|10 clips + ThreeCrop 256|

+ 训练结果

|model|configs|acc1/acc5 efficient mode|acc1/acc5 accurate mode|
|:-:|:-|:-:|:-:|
|tsm_r50_1x1x8|default config efficient mode|72.28 / 90.30|75.30 / 91.89|
|tsm_r50_1x1x8|randaugment|72.37 / 90.55|74.97 / 91.88|
|tsm_r50_1x1x8|randaugment|72.28 / 90.40||
|tsm_r50_1x1x8|mixup|72.53 / 90.40|75.43 / 92.03|
|tsm_mobilenetv2_dense_1x1x8|default config|69.05 / 88.33||

## 3. I3D

+ 测试使用的 pipeline

|model|efficient mode|accurate mode|
|:-:|:-:|:-:|
|i3d_r50_32x2x1|CenterCrop 224|10 clips + ThreeCrop 256|

+ 训练结果

|model|configs|acc1/acc5 efficient mode|acc1/acc5 accurate mode|
|:-:|:-|:-:|:-:|
|i3d_r50_32x2x1|GluonCV|66.66 / 85.84|74.0 / x|
|i3d_r50_32x2x1|default config|67.09 / 86.70|74.47 / 91.67|
|i3d_r50_32x2x1|default config|66.95 / 86.58||
|i3d_r50_32x2x1|randaugment|66.52 / 86.51||
|i3d_r50_32x2x1|mixup alpha=1.0|65.35 / 85.87|72.41 / 90.65|
|i3d_r50_32x2x1|cutmix alpha=1.0|66.07 / 86.45||
