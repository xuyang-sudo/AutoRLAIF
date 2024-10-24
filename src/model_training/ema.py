import torch
from transformers import TrainerCallback

class LoRAEMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class LoRAEMACallback(TrainerCallback):
    def __init__(self, model, decay=0.9):
        self.ema = LoRAEMA(model, decay)
        self.ema.register()

    def on_step_end(self, args, state, control, **kwargs):
        self.ema.update()

    def on_evaluate(self, args, state, control, **kwargs):
        self.ema.apply_shadow()
        yield
        self.ema.restore()

    def on_train_end(self, args, state, control, **kwargs):
        self.ema.apply_shadow()

    def on_save(self, args, state, control, **kwargs):
        self.ema.apply_shadow()
        yield
        self.ema.restore()
