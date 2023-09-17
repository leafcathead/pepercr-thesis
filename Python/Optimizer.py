from abc import ABC, abstractmethod
import yaml
import numpy as np

#CONFIG_PATH = r'ConfigFiles/config.yaml'

class Optimizer(ABC):
    pass

# self.flags - The entire search area of flags.
# self.flag_presets - List of subsets of self.flags
class IterativeOptimizer(Optimizer):

    def __init__(self, fp, num_of_p):
        self.flags = fp
        self.num_of_presets = num_of_p
        self.flag_presets = self.generate_initial_domain()
        print(self.flag_presets)

        print("Iterative Optimizer")

    def generate_initial_domain(self):
        # Generate a random integer, max size of self.flags. This will determine max size of each preset.
        # Randomly select flags of that size. WITH REPLACEMENT!
        presets = []

        preset_size = np.random.randint(len(self.flags))
        for i in range(self.num_of_presets):
            print(i)
            presets.append(np.random.choice(self.flags, size=preset_size, replace=True))
        return presets

    def optimize(self):
        pass
