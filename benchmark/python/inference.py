from inference_utils import DoInference
from mmaction2_model_config_builder import build_mmaction2_config


def test_2d_pytorch_models(warm_up=10,
                           iterations=1000,
                           input_shape_2d=(1, 8, 3, 224, 224),
                           cudnn_benchmark=False,
                           fp16=False,
                           enable_fuse_conv_bn=False):

    model_types = [
        'tsn_res',
        'tsm_mobilenet_v2',
        'tsm_res',
        'tpn_tsm_res',
        'tin_res',
        'tam_res',
    ]

    for model_type in model_types:
        model_config = build_mmaction2_config(model_type)
        DoInference(
            model_type="mmaction2",
            model_args=dict(
                config=model_config,
                cudnn_benchmark=cudnn_benchmark,
                fp16=fp16,
                enable_fuse_conv_bn=enable_fuse_conv_bn),
            input_shape=input_shape_2d,
            warmup=warm_up,
            iterations=iterations,
        ).inference()


def test_2d_onnx_models(warm_up=10,
                        iterations=1000,
                        input_shape_2d=(1, 8, 3, 224, 224)):
    onnx_files = [
        '../data/onnx/tsn_r50.onnx',
        '../data/onnx/tsm_mobilenet_v2.onnx',
        '../data/onnx/tsm_r50.onnx',
    ]

    for onnx_file in onnx_files:
        DoInference(
            model_type="onnx",
            model_args=dict(onnx_file_path=onnx_file),
            input_shape=input_shape_2d,
            warmup=warm_up,
            iterations=iterations,
        ).inference()


if __name__ == '__main__':
    # input_shape_3d = (1, 1, 3, 32, 224, 224)
    # DoInference(
    #     model_type="onnx",
    #     model_args=dict(
    #         onnx_file_path=
    #         "/ssd/zhangyiyang/mmaction2/checkpoints/onnx/i3d_r50_32x2x1.onnx"
    #     ),
    #     input_shapes=(1, 1, 3, 32, 224, 224),
    #     warmup=5,
    #     iterations=200,
    # ).inference()
    test_2d_pytorch_models(fp16=True)
