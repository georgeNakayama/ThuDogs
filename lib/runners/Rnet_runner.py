import jittor as jt
from jittor import nn
import numpy as np
from lib.runners import Runner

class RnetRunner(Runner):
    def __init__(self):
        super().__init__()

    def train_per_batch(self, batch_idx, inputs, targets):
        self.scheduler.step(iters=batch_idx, epochs=self.epoch, total_len=len(self.train_dataset), by_epoch=False)
        labels = targets[0]
        outputs = self.model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        loss = nn.cross_entropy_loss(outputs, labels)
        self.optimizer.step(loss)
        
        num_correct = np.sum([1 for t, p in zip(labels.data, pred) if p == t])
        acc = num_corrects / self.train_dataset.batch_size
        return num_correct, {'total': loss.data[0]}

    def val_per_batch(self, batch_idx, inputs, targets):
        labels = targets[0]
        outputs = self.model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        num_correct = np.sum([1 for p, l in zip(pred, labels.data) if p == t])
        return {'single' : num_correct}