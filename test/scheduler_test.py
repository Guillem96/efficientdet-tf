import unittest

import tensorflow as tf

import matplotlib.pyplot as plt

from efficientdet import optim


class SchedulerTest(unittest.TestCase):
    
    def test_scheduler(self):
        epochs = 10
        steps_per_epoch = 1024
        scheduler = optim.EfficientDetLRScheduler(epochs, steps_per_epoch)
        lrs = (scheduler(i) for i in range(epochs * steps_per_epoch))
        epoch_ends_at = [i * steps_per_epoch for i in range(epochs)]

        plt.plot(range(epochs * steps_per_epoch), list(lrs))
        plt.vlines(epoch_ends_at, 0, 0.16)
        plt.show(block=True)

if __name__ == "__main__":
    unittest.main()