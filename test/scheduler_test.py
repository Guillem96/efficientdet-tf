import unittest

import tensorflow as tf

import matplotlib.pyplot as plt

from efficientdet import optim


class SchedulerTest(unittest.TestCase):
    
    def test_scheduler(self):
        epochs = 10

        max_lr = 3e-3
        alpha = 1e-2

        steps_per_epoch = 1024 
        scheduler = optim.WarmupCosineDecayLRScheduler(
            max_lr,
            steps_per_epoch, 
            (steps_per_epoch * (epochs - 1)), alpha=alpha)

        lrs = [scheduler(i) for i in range(epochs * steps_per_epoch)]
        epoch_ends_at = [i * steps_per_epoch for i in range(epochs)]

        print('Last lr', lrs[-1])

        plt.plot(range(epochs * steps_per_epoch), lrs)
        plt.vlines(epoch_ends_at, 0, max_lr)
        plt.show(block=True)

if __name__ == "__main__":
    unittest.main()