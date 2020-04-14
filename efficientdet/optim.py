import math

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class WarmupCosineDecayLRScheduler(LearningRateSchedule):

    def __init__(self, 
                 max_lr: float,
                 warmup_steps: int,
                 decay_steps: int,
                 alpha: float = 0.):
        super(WarmupCosineDecayLRScheduler, self).__init__()

        self.name = 'WarmupCosineDecayLRScheduler'
        self.alpha = alpha

        self.max_lr = max_lr
        self.last_step = 0

        self.warmup_steps = warmup_steps
        self.linear_increase = self.max_lr / float(self.warmup_steps)

        self.decay_steps = decay_steps

    @property
    def current_lr(self):
        if self.last_step < self.warmup_steps:
            return self.linear_increase * self.last_step
        else:
            rate = (self.last_step - self.warmup_steps) / self.decay_steps
            cosine_decayed = 0.5 * (1.0 + tf.cos(
                tf.constant(math.pi) * rate))
            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            return self.max_lr * decayed

    def __call__(self, step):
        self.last_step = int(step)
        return self.current_lr

    def get_config(self):
        return {
            "steps_per_epoch": self.steps_per_epoch,
            "n_epochs": self.n_epochs,
            'alpha': self.alpha,
        }