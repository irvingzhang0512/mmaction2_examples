import numpy as np
import onnx
import onnxruntime as rt


def get_numpy_random_inputs(input_shape, dtype=np.float32):
    return np.random.rand(*input_shape).astype(dtype)


def inference_onnx(onnx_file_path, inputs):
    """Get predictions by ONNX.
    For now, multi-gpu mode and dynamic tensor shape are not supported.

    onnx_file_name: Onnx model file path
    inputs: A numpy.ndarray object, the input tensor of onnx model
    """

    # get input tensor name
    onnx_model = onnx.load(onnx_file_path)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1

    # get predictions
    sess = rt.InferenceSession(onnx_file_path)
    return sess.run(None, {net_feed_input[0]: inputs})[0]


if __name__ == '__main__':
    # test onnx inference
    onnx_file_path = "./test.onnx"
    input_shape = (1, 8, 3, 224, 224)
    res = inference_onnx(onnx_file_path, get_numpy_random_inputs(input_shape))
