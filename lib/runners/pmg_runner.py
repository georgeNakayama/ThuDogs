import jittor as jt
from jittor import nn
import numpy as np
from lib.runners import Runner
from lib.utils.general import jigsaw_generator

class PMGRunner(Runner):
    def __init__(self):
        super().__init__()
    
    def train_per_batch(self, batch_idx, inputs, targets, log_interval=10):

        self.scheduler.step(iters=batch_idx, epochs=self.epoch - 1, total_len=len(self.train_dataset), by_epoch=False)
        labels = targets[0]
        
        # Step 1
        inputs1 = jigsaw_generator(inputs, 8)
        output1, _, _, _ = self.model(inputs1)
        loss1 = nn.cross_entropy_loss(output1, labels)
        self.optimizer.step(loss1)

        # Step 2
        inputs2 = jigsaw_generator(inputs, 4)
        _, output2, _, _ = self.model(inputs2)
        loss2 = nn.cross_entropy_loss(output2, labels)
        self.optimizer.step(loss2)

        # Step 3
        inputs3 = jigsaw_generator(inputs, 2)
        _, _, output3, _ = self.model(inputs3)
        loss3 = nn.cross_entropy_loss(output3, labels)
        self.optimizer.step(loss3)

        # Step 4
        _, _, _, output4 = self.model(inputs)
        concat_loss = nn.cross_entropy_loss(output4, labels)
        self.optimizer.step(concat_loss)

        #calculating acc
        pred = np.argmax(output4, axis=1)
        num_correct = np.sum(pred==labels.data)
        acc = num_correct / self.train_dataset.batch_size

        #logging
        iter = batch_idx + (self.epoch - 1) * len(self.train_dataset)
        self.logger.add_scalar('Loss/1', loss1.data[0], iter)
        self.logger.add_scalar('Loss/2', loss2.data[0], iter)
        self.logger.add_scalar('Loss/3', loss3.data[0], iter)
        self.logger.add_scalar('Loss/concat', concat_loss.data[0], iter)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]Loss1: {:.3f}|Loss2: {:.3f}| Loss3: {:.3f}|Concat Loss: {:.3f}|accuracy: {:.3f}'.format(
                    self.epoch, batch_idx, len(self.train_dataset),
                    100. * batch_idx / len(self.train_dataset), loss1.data[0], loss2.data[0], loss3.data[0], concat_loss.data[0], acc))
        return num_correct

    def val_per_batch(self, batch_idx, inputs, targets):
        labels = targets[0]
        pred = self.model(inputs)
        num_correct = np.sum(pred==labels.data)
        return num_correct

