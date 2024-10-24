from transformers.trainer_callback import TrainerCallback

class CustomCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.logging_steps = 1

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.logging_steps == 0:
            if 'loss' in logs:
                print(f"Step: {state.global_step}, Training Loss: {logs['loss']:.4f}")
