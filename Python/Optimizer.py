from abc import ABC, abstractmethod
import numpy as np
import subprocess
import uuid
import pandas as pd


# CONFIG_PATH = r'ConfigFiles/config.yaml'


class Optimizer(ABC):

    def __init__(self, cfg, test_path):
        self.CFG = cfg
        self.nofib_log_dir = cfg["locations"]["nofib_logs_dir"]
        self.nofib_exec_path = cfg["locations"]["nofib_exec_path"]
        self.analysis_dir = cfg["locations"]["nofib_analysis_dir"]
        self.analysis_exec_path = cfg["locations"]["nofib_analysis_exec"]
        self.test_path = test_path
        self.test_name = test_path.split("/")[1]  # Gets only the test name. Slices the test category.
        self.tables = {"slow": None, "norm": None, "fast": None}

    @abstractmethod
    def configure_baseline(self, mode):
        pass

    @abstractmethod
    def optimize(self, mode):
        pass

    @abstractmethod
    def _analyze(self, mode):
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

    def _build_individual_test_command(self, flag_string, log_name, mode):
        return f'make -C {self.test_path} {flag_string}  NoFibRuns={self.CFG["settings"]["nofib_runs"]} mode={mode} 2>&1 | tee {log_name}'

    def _build_individual_analyze_commands(self, log_list, table, csv_name):
        command_string_list = []
        logs_string = ""
        for log in log_list:
            logs_string += " logs/" + log

        return f'nofib-analyse/nofib-analyse --csv={table} {logs_string} > analysis/{csv_name}'


class IterativeOptimizer(Optimizer, ABC):

    def __init__(self, cfg, test_path):
        super().__init__(cfg, test_path)
        self.flags = self.CFG["settings"]["flags"]
        self.num_of_presets = self.CFG["iterative_settings"]["num_of_presets"]
        self.flag_presets = self.__generate_initial_domain()
        self.log_dictionary = dict()
        self.csv_dictionary = dict()
        self.optimal_preset = None

        print("Iterative Optimizer")

    def __generate_initial_domain(self):
        # Generate a random integer, max size of self.flags. This will determine max size of each preset.
        # Randomly select flags of that size. WITH REPLACEMENT!
        presets = []

        preset_size = np.random.randint(len(self.flags))
        for i in range(self.num_of_presets):
            print(i)
            presets.append([np.random.choice(self.flags, size=preset_size, replace=True), uuid.uuid4()])
        return presets

    def optimize(self, mode):
        command_list = []

        for preset in self.flag_presets:

            # If preset already exist in Dictionary, get the same ID.
            run_id = -1
            for entry in self.log_dictionary:
                if np.array_equiv(self.log_dictionary[entry]["preset"], preset[0]):
                    print("Found a match!")
                    run_id = self.log_dictionary[entry]["id"]
                    break

            if run_id == -1:
                run_id = preset[1]

            log_file_name = f'{self.test_name}-iterative-{mode}-{run_id}-nofib-log'
            command = super()._build_individual_test_command(super()._setup_preset_task(preset[0]),
                                                             f'{self.CFG["settings"]["log_output_loc"]}/{log_file_name}',
                                                             mode)
            command_list.append(command)
            self.log_dictionary[log_file_name] = {"preset": preset[0], "mode": mode, "id": run_id}

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

        self.optimal_preset = self._analyze(mode)

    def _analyze(self, mode):
        # Get a list of all files that we need to analyze
        command_list = []
        logs_list = []

        for entry in self.log_dictionary:
            if self.log_dictionary[entry]["mode"] == mode:
                logs_list.append(entry)
                log_name = entry
                flags = self.log_dictionary[entry]["preset"]
                mode = self.log_dictionary[entry]["mode"]
                run_id = self.log_dictionary[entry]["id"]

        # Put them into a command
        output_id = uuid.uuid4()
        runtime_csv_name = f'{self.test_name}-runtime-{mode}-{output_id}.csv'

        # Runtime
        self.csv_dictionary[runtime_csv_name] = output_id
        command_list.append(super()._build_individual_analyze_commands(logs_list, "Runtime", runtime_csv_name))

        # Compile Time
        # compiletime_csv_name = f'{self.test_name}-compiletime-{mode}-{output_id}.csv'
        # self.csv_dictionary[compiletime_csv_name] = output_id
        # command_list.append(super()._build_individual_analyze_commands(logs_list, compiletime_csv_name))

        # Elapsed Time
        elapsed_csv_name = f'{self.test_name}-elapsed-{mode}-{output_id}.csv'
        self.csv_dictionary[elapsed_csv_name] = output_id
        command_list.append(super()._build_individual_analyze_commands(logs_list, "Elapsed", elapsed_csv_name))

        # Launch the analysis program and export to CSV
        for c in command_list:
            # print("Running Command: " + c)
            result = subprocess.run(
                c,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.nofib_exec_path,
                text=True)

        # Re-Import that CSV and re-configure it the way we want

        # Configure column headers
        headers_ideal = ["ID", "Flags", "Mode", "Runtime", "Elapsed_Time"]  # This is for our combined table later.
        headers_real = ["Program"]
        for log in logs_list:
            headers_real.append(log)

        print(headers_real)

        run_times = pd.read_csv(self.analysis_dir + "/" + runtime_csv_name, header=None, names=headers_real)
        print(run_times.head())
        # compile_times = pd.read_csv(self.analysis_dir + "/" + self.csv_dictionary[compiletime_csv_name])
        elapsed_times = pd.read_csv(self.analysis_dir + "/" + elapsed_csv_name, header=None, names=headers_real)
        print(elapsed_times.head())

        tables_to_merge = [run_times, elapsed_times]

        # Create the Custom Pandas table

        merged_table = pd.DataFrame(columns=["ID", "Flags", "Mode", "Runtime", "Elapsed_Time"])

        # Configure each row.

        # for log in self.log_dictionary:
        #     if mode == self.log_dictionary[log]:
        #         id = self.log_dictionary[log]["id"]
        #         flags= self.log_dictionary[log]["preset"]

        print(self.log_dictionary)

        for t in tables_to_merge:
            for c in t.columns:
                if not (c == "Program"):
                    r_id = self.log_dictionary[c]["id"]
                    flags = self.log_dictionary[c]["preset"]
                    m = self.log_dictionary[c]["mode"]
                    r = t[c].max()
                    e = t[c].max()

                    merged_table.loc[len(merged_table.index)] = [r_id, flags, m, r, e]

        self.tables[mode] = merged_table

        return merged_table

    def configure_baseline(self, mode):
        command_list = []
        baseline_list = [["-O0"], ["-O2"]]
        print("Configuring baseline... -O0, -O2")

        for preset in baseline_list:
            log_file_name = f'{self.test_name}-iterative{preset[0]}-{mode}-nofib-log'
            command_list.append(super()._build_individual_test_command(super()._setup_preset_task(preset),
                                                                       f'{self.CFG["settings"]["log_output_loc"]}/{log_file_name}',
                                                                       mode))
            self.log_dictionary[log_file_name] = {"preset": preset, "mode": mode, "id": preset[0]}

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

    def write_results(self):
        # Take the tables in the dictionary and concatinate them together!

        tables = self.tables.values()
        complete_table = pd.concat(tables)

        print(complete_table)

        complete_table.to_csv(f'{self.analysis_dir}/{self.test_name}-Iterative-COMPLETE.csv')

        pass


class BOCAOptimizer(Optimizer, ABC):
    pass


class GeneticOptimizer(Optimizer, ABC):
    pass
