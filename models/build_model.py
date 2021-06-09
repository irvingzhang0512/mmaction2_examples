from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
import torch


def build_model_by_config(config, ckpt_path=None):
    if isinstance(config, str):
        cfg = Config.fromfile(config)
        cfg.model.backbone.pretrained = None
    elif isinstance(config, dict):
        cfg = config
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    if ckpt_path is not None:
        load_checkpoint(model, ckpt_path, map_location='cpu')
    model.eval()
    return model


def print_layer_name(model, layer_name):
    if len(model._modules) == 0:
        print(layer_name)
    else:
        for key in model._modules:
            name = key if len(layer_name) == 0 else layer_name + '/' + key
            print_layer_name(model._modules[key], name)


if __name__ == '__main__':
    # config = "/ssd01/zhangyiyang/demo-utils/action_recognition/data/mmaction2_tsm_r50_1x1x16_inference.py" # noqa
    # ckpt = "/ssd01/zhangyiyang/demo-utils/action_recognition/data/mmaction2_tsm_1x1x16_v0.4.0.pth" # noqa
    ckpt = "/ssd01/zhangyiyang/mmaction2_github/checkpoints/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth"  # noqa
    config = "/ssd01/zhangyiyang/mmaction2_github/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py"  # noqa
    # config = "/ssd01/zhangyiyang/mmaction2_github/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py" # noqa
    # ckpt = "/ssd01/zhangyiyang/mmaction2_github/checkpoints/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth" # noqa
    input_shape = (1, 8, 3, 224, 224)
    device = "cuda:0"

    inputs = torch.ones(input_shape, dtype=torch.float32).to(device)
    model = build_model_by_config(config, ckpt).to(device)
    print_layer_name(model, "")
    # with torch.no_grad():
    #     results = model(return_loss=False, imgs=inputs)
    #     # print(results)
