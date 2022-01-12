from jittor import optim
from lib.utils import OPTIMS

__all__ = ['SGD', 'RMSprop', 'Adam', 'AdamW']

@OPTIMS.register_module()
class SGD(optim.SGD):
    def __init__(self, **kwargs):
        super(SGD, self).__init__(**kwargs)

    def parameters(self):
        data = {}
        for k,d in self.__dict__.items():
            if k == "param_groups":
                continue
            data[k]=d 
        return data
    
    def load_parameters(self,data):
        if isinstance(data,dict):
            for k,d in data.items():
                if k in self.__dict__:
                    self.__dict__[k]=d 

@OPTIMS.register_module()
class RMSprop(optim.RMSprop):
    def __init__(self, **kwargs):
        super(RMSprop, self).__init__(**kwargs)

@OPTIMS.register_module()
class Adam(optim.Adam):
    def __init__(self, **kwargs):
        super(Adam, self).__init__(**kwargs)

@OPTIMS.register_module()
class AdamW(optim.AdamW):
    def __init__(self, **kwargs):
        super(AdamW, self).__init__(**kwargs)