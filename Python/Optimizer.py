from abc import ABC, abstractmethod
import yaml

CONFIG_PATH = r'ConfigFiles/config.yaml'

class Optimizer(ABC):
    pass


class IterativeOptimizer(Optimizer):

    def __init__(self, fp):
        self.flag_preset = fp[0]
        print(self.flag_preset)
        print("Iterative Optimizer")

    def generate_initial_domain(self):
        pass

    def optimize(self):
        pass
