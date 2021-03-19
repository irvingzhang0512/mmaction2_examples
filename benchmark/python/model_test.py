import numpy as np
import torch

from model_utils import OnnxModel, MMAction2Model


def test_onnx_model(onnx_file_path, input_shapes):
    onnx_model = OnnxModel(onnx_file_path)
    inputs = np.random.randn(*input_shapes).astype(np.float32)
    onnx_model.inference([inputs])
    print("Successfully test onnx model")


def test_mmaction2_model(config, ckpt_path, input_shapes):
    model = MMAction2Model(config, ckpt_path)
    imgs = torch.randn(*input_shapes)
    model.inference(dict(imgs=imgs))
    print("Successfully test mmaction2 model")


if __name__ == '__main__':
    # onnx_file_path = "/ssd/zhangyiyang/mmaction2/checkpoints/onnx/i3d_r50_32x2x1.onnx"  # noqa
    # onnx_input_shapes = (1, 1, 3, 32, 224, 224)
    # test_onnx_model(onnx_file_path, onnx_input_shapes)

    mmaction2_config = "/ssd/zhangyiyang/mmaction2/configs/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py"  # noqa
    mmaction2_ckpt_path = "/ssd/zhangyiyang/mmaction2/checkpoints/kinetics400/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth"  # noqa
    mmaction2_input_shape = (1, 1, 3, 32, 224, 224)
    test_mmaction2_model(mmaction2_config, mmaction2_ckpt_path,
                         mmaction2_input_shape)
