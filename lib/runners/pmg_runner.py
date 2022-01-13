import jittor as jt
from jittor import nn
import numpy as np
from lib.runners import Runner
from lib.utils.general import jigsaw_generator

class PMGRunner(Runner):
    def __init__(self):
        super().__init__()
    
    def train_per_batch(self, batch_idx, inputs, targets, log_interval=10):

        self.scheduler.step(iters=batch_idx, epochs=self.epoch, total_len=len(self.train_dataset), by_epoch=False)
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

        pred = np.argmax(output4, axis=1)
        num_correct = np.sum(pred==labels.data)

        total_loss = loss1.data[0] + loss2.data[0] + loss3.data[0] + concat_loss.data[0]
        
        return num_correct, {'1': loss1.data[0], '2': loss2.data[0], '3': loss3.data[0], 'concat': concat_loss.data[0], 'total': total_loss}

    def val_per_batch(self, batch_idx, inputs, targets):
        labels = targets[0]
        out1, out2, out3, out_concat = self.model(inputs)
        combined_out = out1 + out2 + out3 + out_concat
        pred = np.argmax(out_concat, axis=1)
        cobined_pred = np.argmax(combined_out, axis=1)
        num_correct = np.sum(pred==labels.data)
        num_correct_combined = np.sum(cobined_pred==labels.data)
        return {'single': num_correct, 'combined': num_correct_combined}

