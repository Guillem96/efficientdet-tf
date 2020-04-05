import math

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class EfficientDetLRScheduler(LearningRateSchedule):

    def __init__(self, 
                 n_epochs: int, 
                 steps_per_epoch: int,
                 alpha: float = 0.):
        super(EfficientDetLRScheduler, self).__init__()

        self.name = 'EfficientDetLRScheduler'
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.alpha = alpha

        self.initial_lr = 0.
        self.max_lr = 0.16
        self.last_step = 0

        self.linear_increase = tf.linspace(
            self.initial_lr, self.max_lr, self.steps_per_epoch)
        
        self.total_steps = n_epochs * steps_per_epoch
        self.warmup_steps = steps_per_epoch
        self.decay_steps = self.total_steps - self.warmup_steps

    @property
    def current_lr(self):
        if self.last_step < self.warmup_steps:
            return self.linear_increase[self.last_step]
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