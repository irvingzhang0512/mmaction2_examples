from mmcv import Config
from mmaction.models import build_model


def _config_dict_to_model_name(config):
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


def create_model(config):
    if isinstance(config, str):
        config = Config.fromfile(config)
        config = config.model
    elif isinstance(config, dict):
        pass
    model = build_model(config)

    return model
