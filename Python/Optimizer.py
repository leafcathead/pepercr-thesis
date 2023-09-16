from abc import ABC, abstractmethod


class Optimizer(ABC):
    pass


class IterativeOptimizer(Optimizer):

    def __init__(self, fp):
        self.flag_preset = fp
        print("Iterative Optimizer")