__all__ = ['Registry', 'OPTIMS', 'SCHEDULERS', 'MODELS', 'DATASETS', 'TRANSFORMS', 'LOSSES', 'build_from_cfg']

class Registry:
    def __init__(self):
        self.modules = {}
    
    def register_module(self, name = None, module=None):
        def _register_module(module):
            key = name
            if key is None:
                key = module.__name__
            assert key not in self.modules, '{} is already registered'.format(key)
            self.modules[key]=module
            return module
        if module is not None:
            return _register_module(module)
        return _register_module
    
    def get_module(self,name):
        assert name in self.modules, '{} is not registered'.format(name)
        return self.modules[name]

def build_from_cfg(cfg, registry, **kwargs):
    if isinstance(cfg, str):
        return registry.get_module(cfg)(**kwargs)
    elif isinstance(cfg, dict):
        args = cfg.copy()
        args.update(kwargs)
        name = args.pop('type')
        module = registry.get_module(name)
        try:
            return module(**args)
        except TypeError as e:
            if "<class" not in str(e):
                e = f"{obj_cls}.{e}"
            raise TypeError(e)
    elif isinstance(cfg,list):
        from jittor import nn 
        return nn.Sequential([build_from_cfg(cf, registry, **kwargs) for cf in cfg])
    elif cfg is None:
        return None
    else:
        assert False, 'type {} not supported'.format(type(cfg))

def build_params_from_cfg(cfg, params, model):
    if isinstance(cfg, str):
        return registry.get_module(cfg)(**kwargs)
    elif isinstance(cfg, dict):
        args = cfg.copy()
        args.update(kwargs)
        name = args.pop('type')
        module = registry.get_module(name)
        try:
            return module(**args)
        except TypeError as e:
            if "<class" not in str(e):
                e = f"{obj_cls}.{e}"
            raise TypeError(e)
    elif isinstance(cfg,list):
        from jittor import nn 
        return nn.Sequential([build_from_cfg(cf, registry, **kwargs) for cf in cfg])
    elif cfg is None:
        return None
    else:
        assert False, 'type {} not supported'.format(type(cfg))

OPTIMS = Registry()
SCHEDULERS = Registry()
MODELS = Registry()
DATASETS = Registry()
TRANSFORMS = Registry()
LOSSES = Registry()

        
    