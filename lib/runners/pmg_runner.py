import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
from jittor.dataset import ImageFolder
import lib.datasets.transforms as trans
from tensorboardX import SummaryWriter
from datetime import datetime
import pickle
from lib.utils.general import build_file, jigsaw_generator
from lib.configs import init_cfg, get_cfg
from lib.utils import build_from_cfg, MODELS, SCHEDULERS, OPTIMS, DATASETS, TRANSFORMS

class PMGRunner:
    def __init__(self):
        cfg = get_cfg()
        self.cfg = cfg
        assert cfg.exp_name is not None, 'must set experiement id using exp_name in config file'
        self.exp_name = str(cfg.exp_name)
        self.work_dir = os.path.join(cfg.work_dir, self.exp_name)
        self.batch_size = cfg.batch_size if cfg.batch_size else 128
        self.max_epochs = cfg.max_epochs if cfg.max_epochs else 50
        self.save_interval = cfg.save_interval if cfg.save_interval else 5
        self.val_interval = cfg.val_interval if cfg.val_interval else 1
        self.save_dir = os.path.join(cfg.save_dir, self.exp_name) if cfg.save_dir else 'saved_weights/{}'.format(self.exp_name)


        transforms_train = build_from_cfg(cfg.transforms_train, TRANSFORMS)
        transforms_val = build_from_cfg(cfg.transforms_val, TRANSFORMS)
        print('Using train dataset: {}'.format(cfg.train_dataset['type']))
        self.train_dataset = build_from_cfg(cfg.train_dataset, DATASETS, transforms=transforms_train)
        print("total length is {}".format(self.train_dataset.total_len))

        print('Using validation dataset: {}'.format(cfg.val_dataset['type']))
        self.val_dataset = build_from_cfg(cfg.val_dataset, DATASETS, transforms=transforms_val)
        print("total length is {}".format(self.val_dataset.total_len))

        #self.test_dataset = build_from_cfg(cfg.test_dataset, DATASETS, transforms=transforms)

        if self.train_dataset:
            self.max_iter = self.max_epochs * len(self.train_dataset)
        else:
            self.max_iter = 0

        self.epoch = 0
        self.iter = 0
        self.model = build_from_cfg(cfg.model, MODELS, num_classes=self.train_dataset.num_classes)
        print('Using model: {}'.format(cfg.model['type']))
        self.optimizer = build_from_cfg(cfg.optimizer, OPTIMS, params=[
            {'name': 'classifier_concat', 'params': self.model.classifier_concat.parameters(), 'lr': 0.002},
            {'name': 'conv_block1', 'params': self.model.conv_block1.parameters(), 'lr': 0.002},
            {'name': 'classifier1', 'params': self.model.classifier1.parameters(), 'lr': 0.002},
            {'name': 'conv_block2', 'params': self.model.conv_block2.parameters(), 'lr': 0.002},
            {'name': 'classifier2', 'params': self.model.classifier2.parameters(), 'lr': 0.002},
            {'name': 'conv_block3', 'params': self.model.conv_block3.parameters(), 'lr': 0.002},
            {'name': 'classifier3', 'params': self.model.classifier3.parameters(), 'lr': 0.002},
            {'name': 'features', 'params': self.model.features.parameters(), 'lr': 0.0002}]
            )
        print('Using optimizer: {}'.format(cfg.optimizer['type']))
        self.scheduler = build_from_cfg(cfg.scheduler, SCHEDULERS, optimizer=self.optimizer)
        print('Using scheduler: {}'.format(cfg.scheduler['type']))
        self.logger = SummaryWriter(os.path.join(self.work_dir, 'Tensorboard'), flush_secs=10)

    def run(self):
        while not self.finished:
            self.train()
            if self.epoch % self.val_interval == 0:
                self.val()
            if self.epoch % self.save_interval == 0:
                self.save()
        self.save()

    @property
    def finished(self):
        return self.iter >= self.max_iter or self.epoch >= self.max_epochs  
    
    def train(self):
        assert self.train_dataset is not None, 'please set training dataset'
        self.model.train()
        total_corrects = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_dataset):
            
            classes = targets[0]
             # Step 1
            inputs1 = jigsaw_generator(inputs, 8)
            output1, _, _, _ = self.model(inputs1)
            loss1 = nn.cross_entropy_loss(output1, classes)
            self.optimizer.step(loss1)

            # Step 2
            inputs2 = jigsaw_generator(inputs, 4)
            _, output2, _, _ = self.model(inputs2)
            loss2 = nn.cross_entropy_loss(output2, classes)
            self.optimizer.step(loss2)

            # Step 3
            inputs3 = jigsaw_generator(inputs, 2)
            _, _, output3, _ = self.model(inputs3)
            loss3 = nn.cross_entropy_loss(output3, classes)
            self.optimizer.step(loss3)

            # Step 4
            _, _, _, output4 = self.model(inputs)
            concat_loss = nn.cross_entropy_loss(output4, classes)
            self.optimizer.step(concat_loss)

            #calculating acc
            pred = np.argmax(output4, axis=1)
            num_correct = np.sum(pred==classes.data)
            total += self.train_dataset.batch_size
            total_corrects += num_correct
            acc = num_correct / self.train_dataset.batch_size

            #logging
            self.logger.add_scalar('Loss/1', loss1.data[0], self.iter)
            self.logger.add_scalar('Loss/2', loss2.data[0], self.iter)
            self.logger.add_scalar('Loss/3', loss3.data[0], self.iter)
            self.logger.add_scalar('Loss/concat', concat_loss.data[0], self.iter)
            self.scheduler.step(iters=self.iter, epochs=self.epoch, total_len=len(self.train_dataset), by_epoch=False)
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.3f}|Loss2: {:.3f}| Loss3: {:.3f}|Concat Loss: {:.3f}|\t accuracy: {:.6f}'.format(
                        self.epoch, batch_idx, len(self.train_dataset),
                        100. * batch_idx / len(self.train_dataset), loss1.data[0], loss2.data[0], loss3.data[0], concat_loss.data[0], acc))
            self.iter += 1
        total_acc = total_corrects / total
        print('Train Epoch: {}\t Accuracy: {:.6f}'.format(self.epoch, total_acc))
        self.logger.add_scalar('Accuracy/train', total_acc, self.epoch)
        for nlr in range(len(self.optimizer.param_groups)):
            print('Current learning rate for parameter group {} is {:.6f} at epoch {}'.format(self.optimizer.param_groups[nlr]['name'], self.optimizer.param_groups[nlr]['lr'], self.epoch))
            self.logger.add_scalar('Learning rate/{}'.format(self.optimizer.param_groups[nlr]['name']), self.optimizer.param_groups[nlr]['lr'], self.epoch)
        self.epoch += 1

    @jt.no_grad()
    @jt.single_process_scope()
    def val(self):
        if self.val_dataset is None:
            assert False, 'please set validation dataset'
        print('Evaluating---------')
        self.model.eval()
        total_preds = []
        total_targets = []
        for batch_idx, (inputs, targets) in enumerate(self.val_dataset):
            pred = self.model(inputs)
            total_preds.extend(pred)
            total_targets.extend(targets[0].data)

        acc = np.sum([1 for p, t in zip(total_preds, total_targets) if p == t]) / self.val_dataset.total_len
        print('Validation at iteration {}: \tAcc: {:.6f}'.format(self.iter, acc))
        self.logger.add_scalar('Accuracy/validation', acc, self.epoch)

    @jt.single_process_scope()
    def save(self):
        save_data = {
            "meta":{
                "epoch": self.epoch,
                "iter": self.iter,
                "max_iter": self.max_iter,
                "max_epoch": self.max_epochs,
                "config": self.cfg.dump()
            },
            "model":self.model.state_dict(),
            "scheduler": self.scheduler.parameters(),
            "optimizer": self.optimizer.state_dict()
        }

        save_file = build_file(self.work_dir,prefix=f"checkpoints/ckpt_{self.epoch}.pkl")
        jt.save(save_data,save_file)
