import abc
import logging

import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmaction.models import build_model as mmaction2_build_model
import onnx
import onnxruntime as rt


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
    elif model_type == 'mmaction2':
        return MMAction2Model(**kwargs)
    raise ValueError(f"Unknown model type {model_type}")


class BaseModel(metaclass=abc.ABCMeta):
    """所有模型相关类的基类"""

    @abc.abstractmethod
    def inference(self, inputs):
        pass

    @abc.abstractmethod
    def print_model_info(self):
        pass


class OnnxModel(BaseModel):

    def __init__(self, onnx_file_path, input_names=None, outputs=None):
        super().__init__()
        self.onnx_file_path = onnx_file_path
        self.onnx_model = onnx.load(onnx_file_path)
        logging.info(f"Successfully load onnx model {onnx_file_path}")

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
        print(f"ONNX model: {self.onnx_file_path}")


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
            self.model = mmaction2_build_model(
                cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        elif isinstance(config, config):
            self.model = mmaction2_build_model(config, None, None)
        else:
            raise ValueError({f"Unknown config {config}"})

        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        if fp16:
            wrap_fp16_model(self.model)
        if enable_fuse_conv_bn:
            self.model = fuse_conv_bn(self.model)

    def inference(self, inputs):
        self.model = MMDataParallel(self.model, device_ids=[0])
        self.model.eval()
        with torch.no_grad():
            return self.model(return_loss=False, **inputs)
        # torch.cuda.synchronize()

    def print_model_info(self):
        print(f"PyTorch model: {self.config}, fp16 {self.fp16}, "
              f"cudnn {self.cudnn_benchmark}, "
              f"fuse_conv_bn {self.enable_fuse_conv_bn}")

    def save_onnx(self,
                  output_file,
                  input_shape,
                  ckpt_path=None,
                  opset_version=11):
        self.model.cpu().eval()
        self.model = _convert_batchnorm(self.model)
        # onnx.export does not support kwargs
        if hasattr(self.model, 'forward_dummy'):
            self.model.forward = self.model.forward_dummy
        else:
            raise NotImplementedError(
                'Please implement the forward method for exporting.')
        if ckpt_path is not None:
            load_checkpoint(self.model, ckpt_path, map_location='cpu')

        input_tensor = torch.randn(input_shape)

        torch.onnx.export(
            self.model,
            input_tensor,
            output_file,
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=opset_version)
