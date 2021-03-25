import time

from model_utils import build_model


class DoInference:

    def __init__(self,
                 model_type,
                 input_shape,
                 model_args=None,
                 fp16=False,
                 iterations=200,
                 warmup=5):
        if model_args is None:
            model_args = {}
        self.modle_type = model_type
        self.model = build_model(model_type, **model_args)

        self.iterations = iterations
        self.warmup = warmup
        self.input_shape = input_shape

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

        self.print_result(total_time, batch_size)

    def print_result(self, total_time, batch_size):
        self.model.print_model_info()
        if batch_size != 1:
            t_per_batch = total_time / self.iterations * 1000.
            print(f"Avg inference time per batch {t_per_batch:.0f} ms")
        t_per_sample = total_time / self.iterations / batch_size * 1000
        print(f"Avg inference time per sample {t_per_sample:.0f} ms")
