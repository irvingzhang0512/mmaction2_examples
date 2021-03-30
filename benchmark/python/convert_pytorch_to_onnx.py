import os

from model_utils import MMAction2Model
from mmaction2_model_config_builder import build_mmaction2_config


def convert_2d_recognizers(onnx_models_path, input_shape):
    model_types = [
        'tsn_res',
        'tsm_res',
        'tsm_mobilenet_v2',
        # 'tam_res',
        # 'tin_res',
        # 'tpn_tsm_res',
    ]

    for model_type in model_types:
        # build model
        if 'tsn' in model_type:
            model_config = build_mmaction2_config(model_type)
        else:
            model_config = build_mmaction2_config(
                model_type, num_segments=input_shape[1])
        model = MMAction2Model(model_config)
        onnx_file_name = f'{model.model_name}_{input_shape[1]}f.onnx'
        print(onnx_file_name)
        onnx_path = os.path.join(onnx_models_path, onnx_file_name)
        model.save_onnx(onnx_path, input_shape)


def convert_3d_recognizers(onnx_models_path, input_shape):
    model_types = [
        # 'c3d_res',
        'i3d_res',
        'r2plus1d_res',
        'slowonly_res',
        'slowfast_res',
        'tpn_slowonly',
        'csn',
        'x3d',
    ]

    for model_type in model_types:
        # build model
        model_config = build_mmaction2_config(model_type)
        model = MMAction2Model(model_config)
        onnx_file_name = f'{model.model_name}_{input_shape[3]}f.onnx'
        print(onnx_file_name)
        onnx_path = os.path.join(onnx_models_path, onnx_file_name)
        model.save_onnx(onnx_path, input_shape)


def convert(cfg, ckpt, onnx_path, input_shape):
    model = MMAction2Model(cfg, ckpt)
    model.save_onnx(onnx_path, input_shape)


if __name__ == '__main__':
    onnx_models_path = "../data/onnx"
    if not os.path.exists(onnx_models_path):
        os.makedirs(onnx_models_path)

    # input_shape_3d = (1, 1, 3, 32, 224, 224)
    # convert_3d_recognizers(onnx_models_path, input_shape_3d)
    # input_shape_3d = (1, 1, 3, 16, 224, 224)
    # convert_3d_recognizers(onnx_models_path, input_shape_3d)
    # input_shape_3d = (1, 1, 3, 8, 224, 224)
    # convert_3d_recognizers(onnx_models_path, input_shape_3d)

    input_shape_2d = (1, 32, 3, 224, 224)
    convert_2d_recognizers(onnx_models_path, input_shape_2d)
    input_shape_2d = (1, 8, 3, 224, 224)
    convert_2d_recognizers(onnx_models_path, input_shape_2d)
    input_shape_2d = (1, 16, 3, 224, 224)
    convert_2d_recognizers(onnx_models_path, input_shape_2d)

    # cfg = "/ssd01/zhangyiyang/mmaction2_github/configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb.py"  # noqa
    # ckpt = "/ssd01/zhangyiyang/mmaction2_github/checkpoints/tsm_r50_1x1x8_50e_sthv1_rgb_20210203-01dce462.pth"  # noqa
    # save_path = '/ssd01/zhangyiyang/mmaction2_github/examples/benchmark/data/onnx/tsm_r50_sthv1.onnx'  # noqa
    # input_shape = (1, 8, 3, 224, 224)
    # convert(cfg, ckpt, save_path, input_shape)

    # cfg = "/ssd01/zhangyiyang/mmaction2_github/configs/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py"  # noqa
    # ckpt = "/ssd01/zhangyiyang/mmaction2_github/checkpoints/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014-c9cdc656.pth"  # noqa
    # save_path = '/ssd01/zhangyiyang/mmaction2_github/examples/benchmark/data/onnx/i3d_r50_kinetics400.onnx'  # noqa
    # input_shape = (1, 1, 3, 32, 224, 224)
    # convert(cfg, ckpt, save_path, input_shape)
