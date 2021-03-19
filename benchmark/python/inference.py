import os
import time
import numpy as np
import torch

from model_utils import build_model


class DoInference:

    def __init__(self,
                 model_type,
                 input_shapes,
                 model_args=None,
                 fp16=False,
                 iterations=200,
                 warmup=5):
        if model_args is None:
            model_args = {}
        self.modle_type = model_type
        self.model = build_model(model_type, **model_args)

        self.iterations = iterations
        self.warmup = warmup
        self.input_shapes = input_shapes

        self.fp16 = fp16
        self._build_random_inputs(model_type)

    def _build_random_inputs(self, model_type):
        self.random_inputs = None
        if model_type == 'onnx':
            dtype = np.float16 if self.fp16 else np.float32
            self.random_inputs = [
                np.random.randn(*self.input_shapes).astype(dtype)
            ]
        elif model_type == 'mmaction2':
            cast = torch.HalfTensor if self.fp16 else torch.FloatTensor
            imgs = cast(torch.randn(*self.input_shapes))
            self.random_inputs = dict(imgs=imgs)
        else:
            raise ValueError(
                f"Undefine random inputs for model type {model_type}")

    def inference_once(self):
        tic = time.time()
        _ = self.model.inference(self.random_inputs)
        tac = time.time()
        return tac - tic

    def inference(self):
        total_iterations = self.iterations + self.warmup
        batch_size = self.input_shapes[0]

        total_time = .0
        for idx in range(total_iterations):
            tic = time.time()
            _ = self.model.inference(self.random_inputs)
            tac = time.time()
            if idx >= self.warmup:
                total_time += tac - tic

        self.print_result(total_time, batch_size)

    def print_result(self, total_time, batch_size):
        self.model.print_model_info()
        if batch_size != 1:
            t_per_batch = total_time / self.iterations * 1000.
            print(f"Avg inference time per batch {t_per_batch:.2f} ms")
        t_per_sample = total_time / self.iterations / batch_size * 1000
        print(f"Avg inference time per sample {t_per_sample:.2f} ms")


def test_2d_pytorch_models():

    warm_up = 10
    iterations = 1000
    input_shape_2d = (1, 8, 3, 224, 224)
    configs_2d = [
        # './models/tsn_r50.py',
        # './models/tsm_mobilenet_v2.py',
        # './models/tsm_r50.py',
        # './models/tpn_tsm_r50.py',
        './models/tin_r50.py',
    ]

    for config_2d in configs_2d:
        DoInference(
            model_type="mmaction2",
            model_args=dict(
                config=config_2d,
                cudnn_benchmark=False,
                fp16=True,
                enable_fuse_conv_bn=False),
            input_shapes=input_shape_2d,
            warmup=warm_up,
            iterations=iterations,
        ).inference()


def convert_pytorch_to_onnx():
    from model_utils import MMAction2Model
    onnx_models_path = "../data/onnx"
    input_shape_2d = (1, 8, 3, 224, 224)

    configs_2d = [
        './models/tsn_r50.py',
        './models/tsm_mobilenet_v2.py',
        './models/tsm_r50.py',
        './models/tpn_tsm_r50.py',
        './models/tin_r50.py',
    ]

    for config_2d in configs_2d:
        # build model
        model = MMAction2Model(config_2d)
        base_name = os.path.basename(config_2d)
        onnx_file_name = base_name[:base_name.rfind(".")] + ".onnx"
        onnx_path = os.path.join(onnx_models_path, onnx_file_name)
        model.save_onnx(onnx_path, input_shape_2d)


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

    onnx_files = [
        '../data/onnx/tsn_r50.onnx',
        '../data/onnx/tsm_mobilenet_v2.onnx',
        '../data/onnx/tsm_r50.onnx',
    ]
    warm_up = 10
    iterations = 1000
    input_shape_2d = (1, 8, 3, 224, 224)

    for onnx_file in onnx_files:
        DoInference(
            model_type="onnx",
            model_args=dict(onnx_file_path=onnx_file),
            input_shapes=input_shape_2d,
            warmup=warm_up,
            iterations=iterations,
        ).inference()
