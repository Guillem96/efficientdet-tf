from typing import Mapping

import tensorflow as tf

from . import coco


class COCOmAPCallback(tf.keras.callbacks.Callback):

    def __init__(self,
                 validation_data: tf.data.Dataset,
                 class2idx: Mapping[str, int], 
                 validate_every: int = 1,
                 print_freq: int = 10) -> None:
        
        self.validation_data = validation_data
        self.gtCOCO = coco.tf_data_to_COCO(validation_data, class2idx)

        self.class2idx = class2idx
        self.validate_every = validate_every
        self.print_freq = print_freq

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if (epoch + 1) % self.validate_every == 0:
            coco.evaluate(self.model, 
                          self.validation_data, 
                          self.gtCOCO,
                          sum(1 for _ in self.validation_data),
                          self.print_freq)


class LogLearningRate(tf.keras.callbacks.Callback):
    
    def __init__(self):
        super(LogLearningRate, self).__init__()
        self.steps = 0

    def on_train_batch_end(self, batch_idx: int, logs: dict = None) -> None:
        if isinstance(self.model.optimizer.learning_rate, 
                      tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = self.model.optimizer.learning_rate(self.steps)
        else:
            lr = self.model.optimizer.learning_rate(self.steps)
        
        if logs is not None:
            logs['learning_rate'] = lr
        
        self.steps += 1


class RemapLogsName(tf.keras.callbacks.Callback):

    def __init__(self, rename_map: Mapping[str, str]) -> None:
        super(RemapLogsName, self).__init__()
        self.mapping = rename_map

    def _new_key(self, key: str) -> str:
        for old_key, new_key in self.mapping.items():
            if old_key in key:
             return key.replace(old_key, new_key)

        return key
    
    def update_logs(self, logs: dict = None) -> None:
        if logs is not None:
            for k in logs:
                logs[self._new_key(k)] = logs.pop(k)
                
    def on_train_batch_end(self, batch_idx: int, logs: dict = None) -> None:
        self.update_logs(logs)
    
    def on_test_batch_end(self, batch_idx: int, logs: dict = None) -> None:
        self.update_logs(logs)
