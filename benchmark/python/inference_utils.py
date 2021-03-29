import time

from model_utils import build_model


class DoInference:

    def __init__(self,
                 model_type,
                 input_shape,
                 model_args=None,
                 fp16=False,
                 iterations=200,
                 result_file=None,
                 warmup=5):
        if model_args is None:
            model_args = {}
        self.modle_type = model_type
        self.model = build_model(model_type, **model_args)

        self.iterations = iterations
        self.warmup = warmup
        self.input_shape = input_shape
        self.result_file = result_file

        if len(input_shape) == 5:
            # 2d model
            self.num_segments = input_shape[1]
        elif len(input_shape) == 6:
            # 3d model
            self.num_segments = input_shape[3]
        else:
            raise ValueError(f"Invalid input shape{input_shape}")

        self.fp16 = fp16
        self.random_inputs = self.model.build_random_inputs(fp16, input_shape)

    def inference(self):
        total_iterations = self.iterations + self.warmup
        batch_size = self.input_shape[0]

        total_time = .0
        for idx in range(total_iterations):
            tic = time.time()
            _ = self.model.inference(self.random_inputs)
            tac = time.time()
            if idx >= self.warmup:
                total_time += tac - tic
        inference_result = self.get_inference_result(total_time, batch_size)
        self.print_result(inference_result)
        return inference_result

    def print_result(self, res):
        self.model.print_model_info()
        if res['batch_size'] != 1:
            print(
                f"Avg inference time per batch {res['time_per_batch']:.0f} ms")
        print(f"Avg inference time per sample {res['time_per_sample']:.0f} ms")

    def get_inference_result(self, total_time, batch_size):
        """保存模型推理结果.

        要保存的内容包括：
        1. 模型形式：即是PyTorch、ONNX，还是TensorRT等
        2. 模型名称：即是TSM、TSN、I3D等，以及backbone
        3. 测试相关参数：这个就多了，比如 输入帧数量，fp16等
        4. 备注：总有一些其他的需要注意
        """
        t_per_batch = int(round(total_time / self.iterations * 1000., 0))
        t_per_sample = int(
            round(total_time / self.iterations / batch_size * 1000, 0))
        return {
            'model_type': self.model.model_type,  # 模型类型
            'model_name': self.model.model_name,  # 模型名称
            'num_segments': self.num_segments,  # 输入帧数量
            'batch_size': batch_size,  # 测试数据 batch size
            'time_per_batch': t_per_batch,  # batch size 平均推理速度
            'time_per_sample': t_per_sample,  # 每个样本平均推理速度
            'comments': self.model.comments,  # 模型详情
        }
