import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
from jittor.dataset import ImageFolder
from tqdm import tqdm
import lib.datasets.transforms as trans
from tensorboardX import SummaryWriter
import pickle
from lib.utils.general import build_file
from lib.configs import init_cfg, get_cfg
from lib.utils import build_from_cfg, MODELS, SCHEDULERS, OPTIMS, DATASETS, TRANSFORMS

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
        self.save_dir = os.path.join(cfg.save_dir, self.exp_name) if cfg.save_dir else 'saved_weights/{}'.format(self.exp_name)
        self.resume_path = cfg.resume_path if cfg.resume_path else None
        self.save_flag = 0


        val_transforms = build_from_cfg(cfg.val_transforms, TRANSFORMS)
        train_transforms = build_from_cfg(cfg.train_transforms, TRANSFORMS)


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
        self.logger = SummaryWriter(os.path.join(self.work_dir, 'Tensorboard'), flush_secs=10)

        if self.resume_path is not None:
            self.resume()

    def run(self):
        while not self.finished:
            self.epoch += 1
            self.train()
            if (self.epoch - 1) % self.val_interval == 0:
                self.val()
            if (self.epoch - 1) % self.save_interval == 0:
                self.save()
        self.save()

    @property
    def finished(self):
        return self.epoch >= self.max_epoch 
    
    def train(self):
        if self.train_dataset is None:
            assert False, 'please set training dataset'
        self.model.train()
        num_correct = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_dataset):
            num_correct += self.train_per_batch(batch_idx, inputs, targets)

        acc = num_correct / self.train_dataset.total_len
        print('Train Epoch: {}\t Accuracy: {:.6f}'.format(self.epoch, acc))
        self.logger.add_scalar('Accuracy/train', acc, self.epoch)
        self.logger.add_scalar('Learning rate', self.optimizer.lr, self.epoch)

    def train_per_batch(self, batch_idx, inputs, targets):
        raise NotImplementedError

    @jt.no_grad()
    @jt.single_process_scope()
    def val(self):
        if self.val_dataset is None:
            assert False, 'please set validation dataset'
        print('---------Evaluating---------')
        self.model.eval()
        num_correct = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(self.val_dataset)):
            num_correct += self.val_per_batch(batch_idx, inputs, targets)

        acc = num_correct / self.val_dataset.total_len
        print('Validation at epoch {}: \tAcc: {:.6f}'.format(self.epoch, acc))
        self.logger.add_scalar('Accuracy/validation', acc, self.epoch)

    def val_per_batch(self, batch_idx, inputs, targets):
        raise NotImplementedError

    @jt.single_process_scope()
    def save(self, how_many=2):
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

        save_file = build_file(self.work_dir,prefix=f"checkpoints/ckpt_{self.save_flag}.pkl")
        if os.path.exists(save_file):
            os.remove(save_file)
        jt.save(save_data,save_file)
        print('data saved to file path {} for epoch {}'.format(save_file, self.epoch))
        self.save_flag = (self.save_flag + 1) % how_many

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
