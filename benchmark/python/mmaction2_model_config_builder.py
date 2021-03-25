def build_tsn_res(num_lasses=400, depth=50):
    return dict(
        type='Recognizer2D',
        backbone=dict(
            type='ResNet', pretrained=None, depth=depth, norm_eval=False),
        cls_head=dict(
            type='TSNHead',
            num_classes=num_lasses,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.4,
            init_std=0.01),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips=None))


def build_tsm_res(num_classes=400, depth=50, shift_div=8, num_segments=8):
    return dict(
        type='Recognizer2D',
        backbone=dict(
            type='ResNetTSM',
            pretrained=None,
            depth=depth,
            norm_eval=False,
            num_segments=8,
            shift_div=shift_div),
        cls_head=dict(
            type='TSMHead',
            num_classes=num_classes,
            in_channels=2048,
            num_segments=num_segments,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips='prob'))


def build_tsm_mobilenet_v2(num_classes=400, shift_div=8, num_segments=8):
    return dict(
        type='Recognizer2D',
        backbone=dict(
            type='MobileNetV2TSM',
            shift_div=shift_div,
            num_segments=num_segments,
            is_shift=True,
            pretrained=None),
        cls_head=dict(
            type='TSMHead',
            num_segments=num_segments,
            num_classes=num_classes,
            in_channels=1280,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips='prob'))


def build_tin_res(depth=50, num_classes=400, shift_div=4, num_segments=8):
    return dict(
        type='Recognizer2D',
        backbone=dict(
            type='ResNetTIN',
            pretrained=None,
            depth=50,
            norm_eval=False,
            num_segments=num_segments,
            shift_div=shift_div),
        cls_head=dict(
            type='TSMHead',
            num_classes=400,
            in_channels=2048,
            spatial_type='avg',
            num_segments=num_segments,
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=False),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips=None))


def build_tam_res(num_classes=400, depth=50, num_segments=8):
    return dict(
        type='Recognizer2D',
        backbone=dict(
            type='TANet',
            pretrained=None,
            depth=depth,
            num_segments=num_segments,
            tam_cfg=dict()),
        cls_head=dict(
            type='TSMHead',
            num_classes=400,
            in_channels=2048,
            spatial_type='avg',
            num_segments=num_segments,
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips='prob'))


def build_tpn_tsm_res(num_classes=400, depth=50, num_segments=8, shift_div=8):
    return dict(
        type='Recognizer2D',
        backbone=dict(
            type='ResNetTSM',
            pretrained=None,
            depth=depth,
            num_segments=num_segments,
            out_indices=(2, 3),
            norm_eval=False,
            shift_div=shift_div),
        neck=dict(
            type='TPN',
            in_channels=(1024, 2048),
            out_channels=1024,
            spatial_modulation_cfg=dict(
                in_channels=(1024, 2048), out_channels=2048),
            temporal_modulation_cfg=dict(downsample_scales=(8, 8)),
            upsample_cfg=dict(scale_factor=(1, 1, 1)),
            downsample_cfg=dict(downsample_scale=(1, 1, 1)),
            level_fusion_cfg=dict(
                in_channels=(1024, 1024),
                mid_channels=(1024, 1024),
                out_channels=2048,
                downsample_scales=((1, 1, 1), (1, 1, 1))),
            aux_head_cfg=dict(out_channels=174, loss_weight=0.5)),
        cls_head=dict(
            type='TPNHead',
            num_classes=num_classes,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.01),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips=None))


def build_i3d_res(num_classes=400, depth=50):
    return dict(
        type='Recognizer3D',
        backbone=dict(
            type='ResNet3d',
            pretrained2d=True,
            pretrained=None,
            depth=depth,
            conv_cfg=dict(type='Conv3d'),
            norm_eval=False,
            inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
            zero_init_residual=False),
        cls_head=dict(
            type='I3DHead',
            num_classes=num_classes,
            in_channels=2048,
            spatial_type='avg',
            dropout_ratio=0.5,
            init_std=0.01),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips='prob'))


def build_c3d_res(num_classes=400):
    return dict(
        type='Recognizer3D',
        backbone=dict(
            type='C3D',
            pretrained=None,
            style='pytorch',
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=None,
            act_cfg=dict(type='ReLU'),
            dropout_ratio=0.5,
            init_std=0.005),
        cls_head=dict(
            type='I3DHead',
            num_classes=num_classes,
            in_channels=4096,
            spatial_type=None,
            dropout_ratio=0.5,
            init_std=0.01),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips='score'))


def build_slowonly_res(num_classes=400, depth=50):
    return dict(
        type='Recognizer3D',
        backbone=dict(
            type='ResNet3dSlowOnly',
            depth=depth,
            pretrained=None,
            lateral=False,
            conv1_kernel=(1, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        cls_head=dict(
            type='I3DHead',
            in_channels=2048,
            num_classes=num_classes,
            spatial_type='avg',
            dropout_ratio=0.5),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips='prob'))


def build_tpn_slowonly(num_classes=400, depth=50):
    return dict(
        type='Recognizer3D',
        backbone=dict(
            type='ResNet3dSlowOnly',
            depth=depth,
            pretrained=None,
            lateral=False,
            out_indices=(2, 3),
            conv1_kernel=(1, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        neck=dict(
            type='TPN',
            in_channels=(1024, 2048),
            out_channels=1024,
            spatial_modulation_cfg=dict(
                in_channels=(1024, 2048), out_channels=2048),
            temporal_modulation_cfg=dict(downsample_scales=(8, 8)),
            upsample_cfg=dict(scale_factor=(1, 1, 1)),
            downsample_cfg=dict(downsample_scale=(1, 1, 1)),
            level_fusion_cfg=dict(
                in_channels=(1024, 1024),
                mid_channels=(1024, 1024),
                out_channels=2048,
                downsample_scales=((1, 1, 1), (1, 1, 1))),
            aux_head_cfg=dict(out_channels=400, loss_weight=0.5)),
        cls_head=dict(
            type='TPNHead',
            num_classes=num_classes,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.01),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips='prob'))


def build_r2plus1d_res(num_classes=400, depth=34):
    return dict(
        type='Recognizer3D',
        backbone=dict(
            type='ResNet2Plus1d',
            depth=depth,
            pretrained=None,
            pretrained2d=False,
            norm_eval=False,
            conv_cfg=dict(type='Conv2plus1d'),
            norm_cfg=dict(type='SyncBN', requires_grad=True, eps=1e-3),
            conv1_kernel=(3, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(1, 1, 1, 1),
            spatial_strides=(1, 2, 2, 2),
            temporal_strides=(1, 2, 2, 2),
            zero_init_residual=False),
        cls_head=dict(
            type='I3DHead',
            num_classes=num_classes,
            in_channels=512,
            spatial_type='avg',
            dropout_ratio=0.5,
            init_std=0.01),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips='prob'))


def build_slowfast_res(num_classes=400, depth=50):
    return dict(
        type='Recognizer3D',
        backbone=dict(
            type='ResNet3dSlowFast',
            pretrained=None,
            resample_rate=8,  # tau
            speed_ratio=8,  # alpha
            channel_ratio=8,  # beta_inv
            slow_pathway=dict(
                type='resnet3d',
                depth=depth,
                pretrained=None,
                lateral=True,
                conv1_kernel=(1, 7, 7),
                dilations=(1, 1, 1, 1),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(0, 0, 1, 1),
                norm_eval=False),
            fast_pathway=dict(
                type='resnet3d',
                depth=50,
                pretrained=None,
                lateral=False,
                base_channels=8,
                conv1_kernel=(5, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                norm_eval=False)),
        cls_head=dict(
            type='SlowFastHead',
            in_channels=2304,  # 2048+256
            num_classes=num_classes,
            spatial_type='avg',
            dropout_ratio=0.5),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips='prob'))


def build_csn(num_classes=400, depth=152):
    return dict(
        type='Recognizer3D',
        backbone=dict(
            type='ResNet3dCSN',
            pretrained2d=False,
            pretrained=None,
            depth=depth,
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=False,
            zero_init_residual=False),
        cls_head=dict(
            type='I3DHead',
            num_classes=num_classes,
            in_channels=2048,
            spatial_type='avg',
            dropout_ratio=0.5,
            init_std=0.01),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips='prob'))


def build_x3d(num_classes=400):
    return dict(
        type='Recognizer3D',
        backbone=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2),
        cls_head=dict(
            type='X3DHead',
            in_channels=432,
            num_classes=num_classes,
            spatial_type='avg',
            dropout_ratio=0.5,
            fc1_bias=False),
        # model training and testing settings
        train_cfg=None,
        test_cfg=dict(average_clips='prob'))


def build_mmaction2_config(model_type, **kwargs):
    if model_type == 'tsn_res':
        return build_tsn_res(**kwargs)
    elif model_type == 'tsm_res':
        return build_tsm_res(**kwargs)
    elif model_type == 'tsm_mobilenet_v2':
        return build_tsm_mobilenet_v2(**kwargs)
    elif model_type == 'tin_res':
        return build_tin_res(**kwargs)
    elif model_type == 'tam_res':
        return build_tam_res(**kwargs)
    elif model_type == 'tpn_tsm_res':
        return build_tpn_tsm_res(**kwargs)
    elif model_type == 'i3d_res':
        return build_i3d_res(**kwargs)
    elif model_type == 'c3d_res':
        return build_c3d_res(**kwargs)
    elif model_type == 'slowonly_res':
        return build_slowonly_res(**kwargs)
    elif model_type == 'tpn_slowonly':
        return build_tpn_slowonly(**kwargs)
    elif model_type == 'slowfast_res':
        return build_slowfast_res(**kwargs)
    elif model_type == 'x3d':
        return build_x3d(**kwargs)
    elif model_type == 'csn':
        return build_csn(**kwargs)
    elif model_type == 'r2plus1d_res':
        return build_r2plus1d_res(**kwargs)
    raise ValueError(f"unknown model type {model_type}")
