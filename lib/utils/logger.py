from tensorboardX import SummaryWriter
import os

class Logger:
    def __init__(self, save_dir):
        self._logger = SummaryWriter(os.path.join(save_dir, 'Tensorboard'), flush_secs=10)

    def log(self, data, steps, label):
        if isinstance(data, dict):
            for k, v in data.items():
                self.log(v, steps, label + '/' + k)
        elif isinstance(data, list):
            for idx, datum in enumerate(data):
                self.log(datum, steps, label + '/' + idx)
        else:
            self._logger.add_scalar(label, data, steps)