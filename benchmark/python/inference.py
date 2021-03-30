from inference_utils import DoInference


def _write_to_local(file_path, inference_result, info):
    keys = [
        'model_type', 'model_name', 'num_segments', 'batch_size', 'comments',
        'time_per_batch', 'time_per_sample'
    ]
    splits = [str(inference_result[key]) for key in keys]
    info = '' if info is None else info
    splits.append(info)
    with open(file_path, 'a') as f:
        f.write(",".join(splits) + '\n')


def test_2d_pytorch_models(
        model_types=(
            'tsn_res',
            'tsm_mobilenet_v2',
            'tsm_res',
            'tin_res',
            'tam_res',
            # 'tpn_tsm_res',
        ),
        warm_up=10,
        iterations=1000,
        input_shape_2d=(1, 8, 3, 224, 224),
        cudnn_benchmark=False,
        fp16=False,
        enable_fuse_conv_bn=False,
        result_file_path=None,
        info=None):
    from mmaction2_model_config_builder import build_mmaction2_config

    for model_type in model_types:
        if 'tsn' in model_type:
            model_config = build_mmaction2_config(model_type)
        else:
            model_config = build_mmaction2_config(
                model_type, num_segments=input_shape_2d[1])
        inference_result = DoInference(
            model_type="mmaction2",
            model_args=dict(
                config=model_config,
                cudnn_benchmark=cudnn_benchmark,
                fp16=fp16,
                enable_fuse_conv_bn=enable_fuse_conv_bn),
            input_shape=input_shape_2d,
            warmup=warm_up,
            iterations=iterations,
        ).inference()
        if result_file_path is not None:
            _write_to_local(result_file_path, inference_result, info)


def test_3d_pytorch_models(
        model_types=(
            'i3d_res',
            'slowonly_res',
            'slowfast_res',
            'x3d',
            'csn',
            'r2plus1d_res',
            # 'tpn_slowonly',
        ),
        warm_up=10,
        iterations=1000,
        input_shape_3d=(1, 1, 3, 32, 224, 224),
        cudnn_benchmark=False,
        fp16=False,
        enable_fuse_conv_bn=False,
        result_file_path=None,
        info=None):
    from mmaction2_model_config_builder import build_mmaction2_config

    for model_type in model_types:
        model_config = build_mmaction2_config(model_type)
        inference_result = DoInference(
            model_type="mmaction2",
            model_args=dict(
                config=model_config,
                cudnn_benchmark=cudnn_benchmark,
                fp16=fp16,
                enable_fuse_conv_bn=enable_fuse_conv_bn),
            input_shape=input_shape_3d,
            warmup=warm_up,
            iterations=iterations,
        ).inference()
        _write_to_local(result_file_path, inference_result, info)


def test_2d_onnx_models(
        onnx_files=('../data/onnx/tsn_res.onnx',
                    '../data/onnx/tsm_mobilenet_v2.onnx',
                    '../data/onnx/tsm_res.onnx', '../data/onnx/tam_res.onnx'),
        warm_up=10,
        iterations=1000,
        input_shape_2d=(1, 8, 3, 224, 224),
        result_file_path=None,
        info=None):

    for onnx_file in onnx_files:
        inference_result = DoInference(
            model_type="onnx",
            model_args=dict(onnx_file_path=onnx_file),
            input_shape=input_shape_2d,
            warmup=warm_up,
            iterations=iterations,
        ).inference()
        _write_to_local(result_file_path, inference_result, info)


def test_3d_onnx_models(
        onnx_files=(
            '../data/onnx/i3d_res.onnx',
            '../data/onnx/r2plus1d_res.onnx',
            '../data/onnx/slowonly_res.onnx',
            '../data/onnx/slowfast_res.onnx',
            '../data/onnx/x3d.onnx',
            '../data/onnx/tpn_slowonly.onnx',
        ),
        warm_up=10,
        iterations=1000,
        input_shape_3d=(1, 1, 3, 32, 224, 224),
        result_file_path=None,
        info=None):
    for onnx_file in onnx_files:
        inference_result = DoInference(
            model_type="onnx",
            model_args=dict(onnx_file_path=onnx_file),
            input_shape=input_shape_3d,
            warmup=warm_up,
            iterations=iterations,
        ).inference()
        _write_to_local(result_file_path, inference_result, info)


def test_frames(num_frames, result_file_name, info, warmup, iterations):
    input_shape_3d = (1, 1, 3, num_frames, 224, 224)
    input_shape_2d = (1, num_frames, 3, 224, 224)
    test_2d_pytorch_models(
        warm_up=warmup,
        iterations=iterations,
        fp16=False,
        input_shape_2d=input_shape_2d,
        result_file_path=result_file_name,
        info=info)
    test_3d_pytorch_models(
        warm_up=warmup,
        iterations=iterations,
        input_shape_3d=input_shape_3d,
        fp16=False,
        result_file_path=result_file_name,
        info=info)
    test_3d_onnx_models(
        onnx_files=(
            f'../data/onnx/I3D_r50_{num_frames}f.onnx',
            f'../data/onnx/CSN_r152_{num_frames}f.onnx',
            f'../data/onnx/2Plus1d_r34_{num_frames}f.onnx',
            f'../data/onnx/SlowOnly_r50_{num_frames}f.onnx',
            f'../data/onnx/SlowFast_r50_{num_frames}f.onnx',
            f'../data/onnx/X3D_{num_frames}f.onnx',
            # f'../data/onnx/SlowOnly_r50_TPN_{num_frames}f.onnx',
        ),
        input_shape_3d=input_shape_3d,
        warm_up=warmup,
        iterations=iterations,
        result_file_path=result_file_name,
        info=info)
    test_2d_onnx_models(
        onnx_files=(
            f'../data/onnx/TSN_r50_{num_frames}f.onnx',
            f'../data/onnx/TSM_r50_{num_frames}f.onnx',
            f'../data/onnx/TSM_MobileNetV2_{num_frames}f.onnx',
            f'../data/onnx/TANet_{num_frames}f.onnx',
        ),
        input_shape_2d=input_shape_2d,
        warm_up=warmup,
        iterations=iterations,
        result_file_path=result_file_name,
        info=info)


if __name__ == '__main__':
    info = 'v100'
    result_file_name = "result.txt"
    warmup = 100
    iterations = 2000

    test_frames(32, result_file_name, info, warmup, iterations)
    test_frames(16, result_file_name, info, warmup, iterations)
    test_frames(8, result_file_name, info, warmup, iterations)
