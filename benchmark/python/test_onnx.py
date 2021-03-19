import numpy as np
import onnx
import onnxruntime as rt

input_shape = (1, 1, 3, 32, 224, 224)
output_file = "/ssd/zhangyiyang/mmaction2/checkpoints/onnx/i3d_r50_32x2x1_softmax.onnx"

input_tensor = np.random.randn(*input_shape).astype(np.float32)

# check by onnx
onnx_model = onnx.load(output_file)
onnx.checker.check_model(onnx_model)

# get onnx output
input_all = [node.name for node in onnx_model.graph.input]
input_initializer = [node.name for node in onnx_model.graph.initializer]
net_feed_input = list(set(input_all) - set(input_initializer))
assert len(net_feed_input) == 1
sess = rt.InferenceSession(output_file)
onnx_result = sess.run(None, {net_feed_input[0]: input_tensor})[0]

print(onnx_result, np.sum(onnx_result))
