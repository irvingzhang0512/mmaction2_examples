import os

from model_utils import MMAction2Model
from mmaction2_model_config_builder import build_mmaction2_config


def convert_2d_recognizers(onnx_models_path, input_shape):
    model_types = [
        'tsn_res', 'tsm_res', 'tsm_mobilenet_v2', 'tin_res', 'tam_res',
        'tpn_tsm_res'
    ]

    for model_type in model_types:
        # build model
        model_config = build_mmaction2_config(model_type)
        model = MMAction2Model(model_config)
        onnx_file_name = model_type + ".onnx"
        onnx_path = os.path.join(onnx_models_path, onnx_file_name)
        model.save_onnx(onnx_path, input_shape)


def convert_3d_recognizers(onnx_models_path, input_shape):
    model_types = [
        'c3d_res',
        # 'i3d_res',
        # 'r2plus1d_res',
        # 'slowonly_res',
        # 'slowfast_res',
        # 'tpn_slowonly',
        # 'csn',
        # 'x3d',
    ]

    for model_type in model_types:
        # build model
        print(model_type)
        model_config = build_mmaction2_config(model_type)
        model = MMAction2Model(model_config)
        onnx_file_name = model_type + ".onnx"
        onnx_path = os.path.join(onnx_models_path, onnx_file_name)
        model.save_onnx(onnx_path, input_shape)


if __name__ == '__main__':
    onnx_models_path = "../data/onnx"
    input_shape_2d = (1, 8, 3, 224, 224)
    input_shape_3d = (1, 1, 3, 16, 112, 112)
    # convert_2d_recognizers(onnx_models_path, input_shape_2d)
    convert_3d_recognizers(onnx_models_path, input_shape_3d)
