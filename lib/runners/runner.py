import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
import pickle
from lib.configs import init_cfg, get_cfg
from lib.utils import build_from_cfg, MODELS, SCHEDULERS, OPTIMS, DATASETS, TRANSFORMS, build_file, Logger
from queue import Queue

class Runner:
    def __init__(self):
        cfg = get_cfg()
        self.cfg = cfg
        assert cfg.exp_name is not None, 'must set experiement id using exp_name in config file'
        self.exp_name = str(cfg.exp_name)
        self.work_dir = os.path.join(cfg.work_dir, self.exp_name)
        self.batch_size = cfg.batch_size if cfg.batch_size else 128
        self.max_epoch = cfg.max_epoch if cfg.max_epoch else 50
        self.save_interval = cfg.save_interval if cfg.save_interval else 5
        self.val_interval = cfg.val_interval if cfg.val_interval else 1
        self.resume_path = cfg.resume_path if cfg.resume_path else None
        self.past_saves = Queue(maxsize=cfg.num_chk_points)

        #building transforms from cfg file
        val_transforms = build_from_cfg(cfg.val_transforms, TRANSFORMS)
        train_transforms = build_from_cfg(cfg.train_transforms, TRANSFORMS)

        #building datasets from cfg file
        print('Using train dataset: {}'.format(cfg.train_dataset['type']))
        self.train_dataset = build_from_cfg(cfg.train_dataset, DATASETS, transforms=train_transforms)
        print("total length is {}".format(self.train_dataset.total_len))

        print('Using validation dataset: {}'.format(cfg.val_dataset['type']))
        self.val_dataset = build_from_cfg(cfg.val_dataset, DATASETS, transforms=val_transforms)
        print("total length is {}".format(self.val_dataset.total_len))

        self.epoch = 0
        self.model = build_from_cfg(cfg.model, MODELS, num_classes=self.train_dataset.num_classes)
        print('Using model: {}'.format(cfg.model['type']))
        self.optimizer = build_from_cfg(cfg.optimizer, OPTIMS, params=self.model.parameters())
        print('Using optimizer: {}'.format(cfg.optimizer['type']))
        self.scheduler = build_from_cfg(cfg.scheduler, SCHEDULERS, optimizer=self.optimizer)
        print('Using scheduler: {}'.format(cfg.scheduler['type']))
        self.logger = Logger(save_dir=self.work_dir)

        if self.resume_path is not None:
            self.resume()

    def run(self):
        while not self.finished:
            self.train()
            if self.epoch % self.val_interval == 0:
                self.val()
            if self.epoch % self.save_interval == 0:
                self.save()
            self.epoch += 1
        self.save()

    @property
    def finished(self):
        return self.epoch >= self.max_epoch 
    
    def train(self):
        if self.train_dataset is None:
            assert False, 'please set training dataset'
        self.model.train()
        total_correct = 0
        total = 0
        train_losses = {}
        for batch_idx, (inputs, targets) in enumerate(self.train_dataset):
            num_correct, loss_dict = self.train_per_batch(batch_idx, inputs, targets)
            total_correct += num_correct
            total += inputs.shape[0]

            iteration = batch_idx + (self.epoch * len(self.train_dataset))
            temp_losses = {}
            for k, v in loss_dict.items():
                if k not in train_losses.keys():
                    train_losses[k] = 0.
                train_losses[k] += v
                temp_losses[k] = train_losses[k] / (batch_idx + 1)

            self.logger.log(temp_losses, iteration, 'Averaged_loss')
            self.logger.log(loss_dict, iteration, 'Loss_per_iter')
            
            if batch_idx % 50 == 0:
                out_string = 'Train Epoch: {} [{}/{} ({:.0f}%)] '.format(self.epoch, batch_idx, len(self.train_dataset), 100. * batch_idx / len(self.train_dataset))
                for k, v in loss_dict.items():
                    out_string += '|Loss/{} : {:.3f} '.format(k, temp_losses[k])
                out_string += '|Accuracy: {:.3f}%({}/{})'.format(100. * float(total_correct) / total, total_correct, total)
                print(out_string)

        acc = total_correct / total
        print('Train Epoch: {}\t Accuracy: {:.6f}'.format(self.epoch, acc))
        self.logger.log(acc, self.epoch, 'Accuracy/train')
        self.logger.log(self.optimizer.lr, self.epoch, 'Learning rate')

    def train_per_batch(self, batch_idx, inputs, targets):
        raise NotImplementedError

    @jt.no_grad()
    @jt.single_process_scope()
    def val(self):
        if self.val_dataset is None:
            assert False, 'please set validation dataset'
        print('---------Evaluating---------')
        self.model.eval()
        total_num_correct_dict = {}
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.val_dataset):
            num_correct_dict = self.val_per_batch(batch_idx, inputs, targets)
            for k, v in num_correct_dict.items():
                if k not in total_num_correct_dict.keys():
                    total_num_correct_dict[k] = 0
                total_num_correct_dict += v 
            total += inputs.shape[0]

        accs = {k: v / total for k, v in total_num_correct_dict.items()}
        out_string = 'Validation at epoch {}: '.format(self.epoch)
        for k, v in accs.items():
            out_string += '| Accuracy/{} : {}% ({:.3f}/{:.3f}) '.format(k, v, total_num_correct_dict[k], total)
        print(out_string)
        self.logger.log(accs, self.epoch, 'Accuracy')

    def val_per_batch(self, batch_idx, inputs, targets):
        raise NotImplementedError

    @jt.single_process_scope()
    def save(self):
        save_data = {
            "meta":{
                "epoch": self.epoch,
                "max_epoch": self.max_epoch,
                "config": self.cfg.dump()
            },
            "model":self.model.state_dict(),
            "scheduler": self.scheduler.parameters(),
            "optimizer": self.optimizer.parameters()
        }

        save_file = build_file(self.work_dir,prefix=f"checkpoints/ckpt_{self.epoch}.pkl")

        if self.past_saves.full():
            os.remove(self.past_saves.get())
        
        jt.save(save_data,save_file)
        self.past_saves.put(save_file)
        print('data saved to file path {} for epoch {}'.format(save_file, self.epoch))

    def load(self, load_path, model_only=False):
        resume_data = jt.load(load_path)

        if (not model_only):
            meta = resume_data.get("meta",dict())
            self.epoch = meta.get("epoch",self.epoch)
            self.max_epoch = meta.get("max_epoch",self.max_epoch)
            self.scheduler.load_parameters(resume_data.get("scheduler",dict()))
            self.optimizer.load_parameters(resume_data.get("optimizer",dict()))
        if ("model" in resume_data):
            self.model.load_parameters(resume_data["model"])
        elif ("state_dict" in resume_data):
            self.model.load_parameters(resume_data["state_dict"])
        else:
            self.model.load_parameters(resume_data)

        print(f"Loading model parameters from {load_path}")

    def resume(self):
        self.load(self.resume_path)
