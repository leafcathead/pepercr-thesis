from abc import ABC, abstractmethod
import yaml
import numpy as np
import subprocess
import uuid


# CONFIG_PATH = r'ConfigFiles/config.yaml'


class Optimizer(ABC):

    def __init__(self, cfg, test_path):
        self.CFG = cfg
        self.nofib_log_dir = cfg["locations"]["nofib_logs_dir"]
        self.nofib_exec_path = cfg["locations"]["nofib_exec_path"]
        self.test_path = test_path
        self.test_name = test_path.split("/")[1]  # Gets only the test name. Slices the test category.

    @abstractmethod
    def configure_baseline(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def write_results(self):
        pass

    def _setup_preset_task(self, preset):
        extra_flags = ""
        if len(preset) > 0:
            extra_flags = 'EXTRA_HC_OPTS="'
            for flag in preset:
                extra_flags += flag + " "
            extra_flags += '" '
            return extra_flags
        # apply_preset_task_all(preset['presetName'], extra_flags)
        else:
            print("No flags? What's the point?")
            return ""

    def _build_individual_test_command(self, flag_string, log_name):
        # return f'make -C {process_name} {flag_string} NoFibRuns=10 2>&1 | tee logs/{process_name}-nofib-log'
        return f'make -C {self.test_path} {flag_string}  NoFibRuns={self.CFG["settings"]["nofib_runs"]} 2>&1 | tee {log_name}'


# self.flags - The entire search area of flags.
# self.flag_presets - List of subsets of self.flags
class IterativeOptimizer(Optimizer, ABC):

    def __init__(self, cfg, test_path):
        super().__init__(cfg, test_path)
        self.flags = self.CFG["settings"]["flags"]
        self.num_of_presets = self.CFG["iterative_settings"]["num_of_presets"]
        self.flag_presets = self.__generate_initial_domain()
        self.log_dictionary = dict()

        print("Iterative Optimizer")

    def __generate_initial_domain(self):
        # Generate a random integer, max size of self.flags. This will determine max size of each preset.
        # Randomly select flags of that size. WITH REPLACEMENT!
        presets = []

        preset_size = np.random.randint(len(self.flags))
        for i in range(self.num_of_presets):
            print(i)
            presets.append(np.random.choice(self.flags, size=preset_size, replace=True))
        return presets

    def optimize(self):
        command_list = []

        for preset in self.flag_presets:
            run_id = uuid.uuid4()
            log_file_name = f'{self.test_name}-{run_id}-nofib-log'
            command = super()._build_individual_test_command(super()._setup_preset_task(preset), f'{self.CFG["settings"]["log_output_loc"]}/{log_file_name}')
            command_list.append(command)
            self.log_dictionary[log_file_name] = preset

        for c in command_list:
            print(fr'Applying command to {self.test_path}')
            result = subprocess.run(
                c,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.nofib_exec_path,
                text=True)
            print(result)

    def configure_baseline(self):
        pass

    def write_results(self):
        pass