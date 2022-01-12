from jittor import lr_scheduler
from lib.utils import SCHEDULERS
import math

class Warmup:
    def __init__(self, optimizer, start_warmup_lr=0., warmup_iter=500, mode=None):
        self.optimizer = optimizer
        self.base_lr = self.optimizer.lr
        self.warmup_iter = warmup_iter
        self.mode = mode
        if mode is None:
            print('No warm up used.')
        self.start_warmup_lr = start_warmup_lr
        self.base_lr_pg = [pg.get("lr", optimizer.lr) for pg in optimizer.param_groups]
        

    def parameters(self):
        parameters = {
            'warmup_iter': self.warmup_iter,
            'mode': self.mode,
            'start_warmup_lr': self.start_warmup_lr,
            'base_lr_pg': self.base_lr_pg
        }
        return parameters 
    def load_parameters(self, dict):
        dict['warmup_iter'] = self.warmup_iter
        dict['mode'] = self.mode 
        dict['start_warmup_lr'] = self.start_warmup_lr
        dict['base_lr_pg'] = self.base_lr_pg

    def get_warmup_lr(self, lr, cur_iter):
        if self.mode == 'exponential':
            ratio = (lr / self.start_warmup_lr) ** (1. / self.warmup_iter)
            return self.start_warmup_lr * (self.ratio ** cur_iter)
        elif self.mode == 'linear':
            ratio = cur_iter / self.warmup_iter
            return self.start_warmup_lr + ratio * (lr - self.start_warmup_lr)
        elif self.mode == 'constant':
            return lr

    def get_lr(self,lr,steps):
        return lr 
    
    def _update_lr(self,steps,get_lr_func):
        self.optimizer.lr = get_lr_func(self.base_lr,steps)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = get_lr_func(self.base_lr_pg[i],steps)

    def step(self,iters,epochs,total_len, by_epoch=True):
        if self.mode is not None:
            if iters>=self.warmup_iter:
                if by_epoch:
                    self._update_lr(epochs, self.get_lr)
                else:
                    self._update_lr(iters-self.warmup_iter,self.get_lr)
            else:
                self._update_lr(iters,self.get_warmup_lr)
        else:
            if by_epoch:
                self._update_lr(epochs, self.get_lr)
            else:
                self._update_lr(iters / total_len, self.get_lr) 

@SCHEDULERS.register_module()
class CosineAnnealingLR(Warmup):
    def __init__(self, optimizer, T_max, T_mult=1, eta_min=0., warmup_iter=500, mode=None, start_warmup_lr=0.):
        self.optimizer = optimizer
        self.eta_min = eta_min
        self.T_max=T_max
        self.T_mult=T_mult
        super(CosineAnnealingLR, self).__init__(optimizer=self.optimizer, start_warmup_lr=start_warmup_lr, warmup_iter=warmup_iter, mode=mode)
    
    def get_lr(self, base_lr,steps):
        target_lr = self.eta_min
        cos_out = math.cos(math.pi * (steps / self.T_max)) + 1
        lr = target_lr + 0.5 * (base_lr - target_lr) * cos_out
        if ((steps / self.T_max) - 1) % 2 == 0:
            print('T_max updated.')
            self.T_max *= self.T_mult
        return lr

    def parameters(self):
        parameters = super().parameters()
        new_parameters = {
            'T_max': self.T_max,
            'eta_min': self.eta_min,
            'T_mult': self.T_mult
        }
        parameters.update(new_parameters)
        return parameters
    def load_parameters(self, parameters):
        assert isinstance(parameters, dict), 'parameters must be a dictionary'
        self.T_max = parameters['T_max']
        self.eta_min = parameters['eta_min']
        self.T_mult = parameters['T_mult']
        parameters.pop('T_max')
        parameters.pop('eta_min')
        parameters.pop('T_mult')
        super().load_parameters(parameters)

@SCHEDULERS.register_module()
class ExponentialLR(lr_scheduler.ExponentialLR):
    def __init__(self, **kwargs):
        super(ExponentialLR, self).__init__(**kwargs)
    def parameters(self):
        parameters = {
            'gamma': self.gamma,
            'last_epoch': self.last_epoch
        }
        return parameters 
    def load_parameters(self, parameters):
        assert isinstance(parameters, dict), 'parameters must be a dictionary'
        self.gamma = parameters['gamma']
        self.last_epoch = parameters['last_epoch']
    

@SCHEDULERS.register_module()
class MultiStepLR(lr_scheduler.MultiStepLR):
    def __init__(self, **kwargs):
        super(MultiStepLR, self).__init__(**kwargs)
    
    def parameters(self):
        parameters = {
            'milestones': self.milestones,
            'gamma': self.gamma,
            'last_epoch': self.last_epoch
        }
        return parameters 
    def load_parameters(self, parameters):
        assert isinstance(parameters, dict), 'parameters must be a dictionary'
        self.gamma = parameters['gamma']
        self.last_epoch = parameters['last_epoch']
        self.milestones = parameters['milestones']
        
@SCHEDULERS.register_module()
class StepLR(lr_scheduler.StepLR):
    def __init__(self, **kwargs):
        super(StepLR, self).__init__(**kwargs)

    def parameters(self):
        parameters = {
            'step_size': self.step_size,
            'gamma': self.gamma,
            'last_epoch': self.last_epoch,
            'cur_epoch': self.cur_epoch
        }
        return parameters 
    def load_parameters(self, parameters):
        assert isinstance(parameters, dict), 'parameters must be a dictionary'
        self.gamma = parameters['gamma']
        self.last_epoch = parameters['last_epoch']
        self.step_size = parameters['step_size']
        self.cur_epoch = parameters['cur_epoch']