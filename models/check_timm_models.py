from mmcv import Config
from mmaction.models import build_recognizer
import timm
import torch
from tqdm import tqdm

config_name = "/ssd01/zhangyiyang/mmaction2_github/configs/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb.py"  # noqa
cfg = Config.fromfile(config_name)
cfg.model.backbone.pretrained = None


def _test_mmaction2(config, model_name, in_channels, input_size):
    imgs = torch.rand((1, 3) + input_size).to("cuda:0")
    gt_labels = torch.LongTensor([2]).to("cuda:0")
    timm_backbone = dict(type=f'timm.{model_name}', pretrained=False)
    # print(config)
    config.model['backbone'] = timm_backbone
    config.model['cls_head']['in_channels'] = in_channels
    recognizer = build_recognizer(config.model).to("cuda:0")
    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, dict)
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    del imgs
    del gt_labels
    del recognizer


writer = open("timm_test_v2.txt", "a")
writer.write(
    "|model name|input size|mean|std|in channels|pretrained|verify|\n")
writer.write("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|\n")

for model_name in tqdm(timm.list_models()[:]):
    try:
        model = timm.create_model(model_name).to("cuda:0")
        pretrained = (len(model.default_cfg.get('url', '')) > 0)
    except:  # noqa
        elements = [model_name, "/", "/", "/", "/", "/", "Model Init Error"]
        writer.write("|")
        writer.write("|".join(elements))
        writer.write("|\n")
        continue

    if hasattr(model, 'forward_features'):
        try:
            img = torch.rand((1, ) +
                             model.default_cfg['input_size']).to("cuda:0")
            with torch.no_grad():
                features = model.forward_features(img)
            in_channels = features.size()[1]
            verify = "True"
            try:
                _test_mmaction2(cfg, model_name, in_channels,
                                model.default_cfg['input_size'])
            except:  # noqa
                verify = "MMAction2 Test Error"
            elements = [
                model_name,
                str(model.default_cfg['input_size']),
                str(model.default_cfg['mean']),
                str(model.default_cfg['std']),
                str(in_channels),
                str(pretrained), verify
            ]
            writer.write("|")
            writer.write("|".join(elements))
            writer.write("|\n")

            del img
        except:  # noqa
            elements = [
                model_name,
                str(model.default_cfg['input_size']),
                str(model.default_cfg['mean']),
                str(model.default_cfg['std']), "/",
                str(pretrained), "Model Inference Error"
            ]
            writer.write("|")
            writer.write("|".join(elements))
            writer.write("|\n")
    else:
        elements = [
            model_name,
            str(model.default_cfg['input_size']),
            str(model.default_cfg['mean']),
            str(model.default_cfg['std']), "/",
            str(pretrained), "Function 'forward_features' doesn't exist"
        ]
        writer.write("|")
        writer.write("|".join(elements))
        writer.write("|\n")

    del model
    writer.flush()

writer.close()
