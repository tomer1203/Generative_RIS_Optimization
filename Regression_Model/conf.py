import yaml

from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(_Singleton('SingletonMeta', (object,), {})): pass
class Config(Singleton):

    def __init__(self):
        with open("configuration.yaml", 'r') as yaml_stream:
            config = yaml.load(yaml_stream,Loader)
        for k, v in config.items():
            setattr(self, k, v)

        # add all the configurations which require some mathematical evaluations
        self.evaluate_expressions()

    def evaluate_expressions(self):
        self.diffusion_inp_size = self.input_size + self.tx_size + self.diffusion_sigma_inp_size

    def set_mode(self, mode):
        mode_func = getattr(self, mode)
        mode_func()

    def mode_a(self):
        conf_mode = {'lr_range': [3,2,1]}
        self.__dict__.update(conf_mode)





