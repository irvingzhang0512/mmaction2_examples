import abc
import logging
import os
import warnings

import numpy as np

try:
    import torch
    from mmaction.models import build_model as mmaction2_build_model
    from mmcv import Config
    from mmcv.cnn import fuse_conv_bn
    from mmcv.runner import load_checkpoint
    from mmcv.runner.fp16_utils import wrap_fp16_model
except ImportError:
    warnings.warn("PyTorch models are not available.")

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa
except ImportError:
    warnings.warn("PyCUDA or TensorRT is not available.")

try:
    import onnx
    import onnxruntime as rt
except ImportError:
    warnings.warn("ONNX or ONNXRuntime is not available.")


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def build_model(model_type, **kwargs):
    if model_type == 'onnx':
        return OnnxModel(**kwargs)
    if model_type == 'tensorrt':
        return TensorRTModel(**kwargs)
    elif model_type == 'mmaction2':
        return MMAction2Model(**kwargs)
    raise ValueError(f"Unknown model type {model_type}")


class BaseModel(metaclass=abc.ABCMeta):
    """??????????????????????????????"""

    @abc.abstractmethod
    def inference(self, inputs):
        pass

    @abc.abstractmethod
    def print_model_info(self):
        pass

    @abc.abstractmethod
    def build_random_inputs(self, fp16, input_shape):
        pass

    @abc.abstractproperty
    def model_name(self):
        pass

    @abc.abstractproperty
    def model_type(self):
        pass

    @abc.abstractproperty
    def comments(self):
        pass


class OnnxModel(BaseModel):

    def __init__(self, onnx_file_path, input_names=None, outputs=None):
        super().__init__()
        self.onnx_file_path = onnx_file_path
        self.onnx_model = onnx.load(onnx_file_path)
        logging.info(f"Successfully load onnx model {onnx_file_path}")

        # get model name
        self._model_name = os.path.basename(onnx_file_path)
        self._model_name = self._model_name[:self._model_name.rfind(".")]

        if input_names is None:
            input_all = [node.name for node in self.onnx_model.graph.input]
            input_initializer = [
                node.name for node in self.onnx_model.graph.initializer
            ]
            input_names = list(set(input_all) - set(input_initializer))
        if not isinstance(input_names, (list, tuple)):
            input_names = [input_names]
        logging.info(f"The input tensor names of onnx model are {input_names}")
        self.input_names = input_names

        self.sess = rt.InferenceSession(onnx_file_path)

    def inference(self, inputs):
        param_dict = {k: v for k, v in zip(self.input_names, inputs)}
        return self.sess.run(None, param_dict)

    def print_model_info(self):
        print(f"ONNX model: {self.model_name}")

    def build_random_inputs(self, fp16, input_shape):
        dtype = np.float16 if fp16 else np.float32
        return [np.random.randn(*input_shape).astype(dtype)]

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_type(self):
        return "ONNX"

    @property
    def comments(self):
        return ""


def _mmaction2_config_to_model_name(config):
    assert isinstance(config, dict)
    assert config['type'] in ['Recognizer2D', 'Recognizer3D']

    backbone = config['backbone']['type']
    neck = config['neck']['type'] if config.get('neck') else None

    if config['type'] == 'Recognizer2D':
        model_names = ['TSM', 'TIN', 'TANet']

        name = None
        base_backbone = ""
        for model_name in model_names:
            if model_name in backbone:
                name = model_name
                base_backbone = backbone.replace(model_name, "")
                break
        if name is None:
            name = 'TSN'

        if 'ResNet' in backbone:
            name = f'{name}_r{config["backbone"]["depth"]}'
        elif len(base_backbone) > 0:
            name = f'{name}_{base_backbone}'

    else:
        model_names = ['SlowOnly', 'SlowFast', 'CSN', 'X3D', '2Plus1d']

        name = None
        base_backbone = ""
        for model_name in model_names:
            if model_name in backbone:
                name = model_name
                base_backbone = backbone.replace(model_name, "")
                break
        if name is None:
            name = 'I3D'

        if 'ResNet' in backbone:
            depth = config["backbone"]["depth"] \
                if config["backbone"].get("depth") else \
                config["backbone"]["slow_pathway"]["depth"]  # slowfast
            name = f'{name}_r{depth}'
        elif len(base_backbone) > 0:
            name = f'{name}_{base_backbone}'

    if neck is not None:
        name = f'{name}_{neck}'

    return name


class MMAction2Model(BaseModel):

    def __init__(self,
                 config,
                 ckpt_path=None,
                 cudnn_benchmark=False,
                 fp16=False,
                 enable_fuse_conv_bn=False):
        self.config = config
        self.fp16 = fp16
        self.cudnn_benchmark = cudnn_benchmark
        self.enable_fuse_conv_bn = enable_fuse_conv_bn

        if isinstance(config, str):
            cfg = Config.fromfile(config)
            cfg.model.backbone.pretrained = None
            self._model_name = _mmaction2_config_to_model_name(cfg.model)
            self.model = mmaction2_build_model(
                cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        elif isinstance(config, dict):
            self.model = mmaction2_build_model(config, None, None)
            # get model name
            self._model_name = _mmaction2_config_to_model_name(config)
        else:
            raise ValueError({f"Unknown config {config}"})

        if ckpt_path is not None:
            load_checkpoint(self.model, ckpt_path, map_location='cpu')

        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        if fp16:
            wrap_fp16_model(self.model)
        if enable_fuse_conv_bn:
            self.model = fuse_conv_bn(self.model)
        self.model.cuda().eval()

    def inference(self, inputs):
        with torch.no_grad():
            return self.model(return_loss=False, **inputs)
        # torch.cuda.synchronize()

    def print_model_info(self):
        print(f"PyTorch model: {self.model_name}, fp16 {self.fp16}, "
              f"cudnn {self.cudnn_benchmark}, "
              f"fuse_conv_bn {self.enable_fuse_conv_bn}")

    def save_onnx(self, output_file, input_shape, opset_version=11):
        # self.model.cpu().train()
        self.model = _convert_batchnorm(self.model)
        self.model.cpu().eval()
        # onnx.export does not support kwargs
        if hasattr(self.model, 'forward_dummy'):
            self.model.forward = self.model.forward_dummy
        else:
            raise NotImplementedError(
                'Please implement the forward method for exporting.')

        input_tensor = torch.randn(*input_shape)

        torch.onnx.export(
            self.model,
            input_tensor,
            output_file,
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=opset_version)

    def build_random_inputs(self, fp16, input_shape):
        cast = torch.HalfTensor if fp16 else torch.FloatTensor
        imgs = cast(torch.randn(*input_shape)).cuda()
        return dict(imgs=imgs)

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_type(self):
        return "PyTorch"

    @property
    def comments(self):
        return (f"fp16 {self.fp16} "
                f"cudnn {self.cudnn_benchmark} "
                f"fuse_conv_bn {self.enable_fuse_conv_bn}")


class TensorRTModel:

    def __init__(self, trt_path):
        # get model name
        self._model_name = os.path.basename(trt_path)
        self._model_name = self._model_name[:self._model_name.rfind(".")]

        # create engine
        self.trt_path = trt_path
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        with open(trt_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        # create context and buffer
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        bindings = []
        host_input = device_input = host_output = device_output = None

        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            print(f"binding name {binding}, idx {binding_idx}")
            shape = trt.volume(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                print(shape)
                host_input = np.empty(shape, dtype=np.float32)
                device_input = cuda.mem_alloc(host_input.nbytes)
                bindings.append(int(device_input))
            else:
                host_output = cuda.pagelocked_empty(shape, dtype)
                device_output = cuda.mem_alloc(host_output.nbytes)
                bindings.append(int(device_output))

        assert device_input is not None
        assert device_output is not None
        assert len(bindings) == 2

        self.bindings = bindings
        self.device_input = device_input
        self.host_input = host_input
        self.device_output = device_output
        self.host_output = host_output

    def inference(self, inputs):
        # ????????????????????? inputs
        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_output, self.device_output,
                               self.stream)
        self.stream.synchronize()
        return self.host_output

    def print_model_info(self):
        print(f"TensorRT model: {self.trt_path}")

    def build_random_inputs(self, fp16, input_shape):
        inputs = np.random.rand(*input_shape).astype(np.float32).ravel()
        np.copyto(self.host_input, inputs)
        return inputs

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_type(self):
        return "TensorRT"

    @property
    def comments(self):
        return ""
