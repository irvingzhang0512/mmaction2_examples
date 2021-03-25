from model_utils import OnnxModel, MMAction2Model, TensorRTModel


def test_onnx_model(onnx_file_path, input_shape):
    onnx_model = OnnxModel(onnx_file_path)
    onnx_model.inference(onnx_model.build_random_inputs(False, input_shape))
    print("Successfully test onnx model")


def test_mmaction2_model(config, ckpt_path, input_shape):
    model = MMAction2Model(config, ckpt_path)
    model.inference(model.build_random_inputs(False, input_shape))
    print("Successfully test mmaction2 model")


def test_tensorrt(trt_path, input_shape):
    model = TensorRTModel(trt_path)
    model.inference(model.build_random_inputs(False, input_shape))
    print("Successfully test trt model")


if __name__ == '__main__':
    # onnx_file_path = "/ssd/zhangyiyang/mmaction2/examples/benchmark/data/onnx/i3d_res.onnx"  # noqa
    # onnx_input_shape = (1, 1, 3, 32, 224, 224)
    # test_onnx_model(onnx_file_path, onnx_input_shape)

    # mmaction2_config = "/ssd/zhangyiyang/mmaction2/configs/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py"  # noqa
    # mmaction2_ckpt_path = "/ssd/zhangyiyang/mmaction2/checkpoints/kinetics400/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth"  # noqa
    # mmaction2_input_shape = (1, 1, 3, 32, 224, 224)
    # test_mmaction2_model(mmaction2_config, mmaction2_ckpt_path,
    #                      mmaction2_input_shape)
    
    trt_path = "/ssd/zhangyiyang/mmaction2/examples/benchmark/data/tensorrt/i3d_res.trt" # noqa
    trt_input_shape = (1, 1, 3, 32, 224, 224)
    test_tensorrt(trt_path, trt_input_shape)
