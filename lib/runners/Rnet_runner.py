import jittor as jt
from jittor import nn
import numpy as np
from lib.runners import Runner

class RnetRunner(Runner):
    def __init__(self):
        super().__init__()

    def train_per_batch(self, batch_idx, inputs, targets, log_interval=10):
        self.scheduler.step(iters=batch_idx, epochs=self.epoch - 1, total_len=len(self.train_dataset), by_epoch=False)
        labels = targets[0]
        outputs = self.model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        loss = nn.cross_entropy_loss(outputs, labels)
        self.optimizer.step(loss)
        
        num_correct = np.sum([1 for t, p in zip(labels.data, pred) if p == t])
        acc = num_corrects / self.train_dataset.batch_size
        
        self.logger.add_scalar('Loss', loss.data[0], self.iter)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t accuracy: {:.6f}'.format(
                    self.epoch, batch_idx, len(self.train_dataset),
                    100. * batch_idx / len(self.train_dataset), loss.data[0], acc))
        return num_correct

    def val_per_batch(self, batch_idx, inputs, targets):
        labels = targets[0]
        outputs = self.model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        num_correct = np.sum([1 for p, l in zip(pred, labels.data) if p == t])
        return num_correct