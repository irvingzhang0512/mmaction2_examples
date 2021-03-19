## Something-Something-v1 的一些结果


## TSM

+ 默认训练 50 ecpohs。
+ 测试使用的 pipeline

|model|efficient mode|accurate mode|
|:-:|:-:|:-:|
|tsm_r50_1x1x8|CenterCrop 224|`twice_samples=True` + ThreeCrop 256|
|tsm_mobilenet_v2_1x1x8_dense|10 clips + CenterCrop 224|10 clips + ThreeCrop 256|

+ 训练结果

|model|configs|top1 acc(efficient/accurate)|top5 acc(efficient/accurate)|delta top1 acc(efficient/accurate)|delta top5 acc(efficient/accurate)|
|:-|:-|:-:|:-:|:-:|:-:|
|tsm-r50 1x1x8|使用默认config自行训练|45.82 / 47.90|74.38 / 76.02|/|/|
|tsm-r50 1x1x8|mmaction2 model zoo|45.58 / 47.70|75.02 / 76.12|||
|tsm-r50 1x1x8|使用model zoo ckpt测试|45.47 / 47.55|74.56 / 75.79|||
|tsm-r50 1x1x8|mixup alpha=1.0|45.68 / 47.73|74.26 / 76.82|||
|tsm-r50 1x1x8|mixup alpha=0.2|46.35 / 48.49|75.07 / 76.88|||
|tsm-r50 1x1x8|cutmix alpha=0.2|45.92 / 48.49|75.23 / 76.88|||
|tsm-r50 1x1x8|randaugment|47.16 / 48.90|76.07 / 77.92|||
