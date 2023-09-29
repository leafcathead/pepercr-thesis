import os
import random
from abc import ABC, abstractmethod
import numpy as np
import subprocess
import uuid
import pandas as pd

from Genetics import crossover_chromosomes
from Genetics import Chromosome
import copy  # Used for testing


# CONFIG_PATH = r'ConfigFiles/config.yaml'


class Optimizer(ABC):

    def __init__(self, cfg, test_path, t):
        self.CFG = cfg
        self.threaded = t
        self.flags = self.CFG["settings"]["flags"]
        self.num_of_threads = cfg["settings"]["multicore_cores"]
        self.nofib_log_dir = cfg["locations"]["nofib_logs_dir"]
        self.nofib_exec_path = cfg["locations"]["nofib_exec_path"]
        self.analysis_dir = cfg["locations"]["nofib_analysis_dir"]
        self.analysis_exec_path = cfg["locations"]["nofib_analysis_exec"]
        self.test_path = test_path
        self.test_name = test_path.split("/")[1]  # Gets only the test name. Slices the test category.
        self.log_dictionary = dict()
        self.csv_dictionary = dict()
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
        extra_flags = 'EXTRA_HC_OPTS="'
        if len(preset) > 0:

            for flag in preset:
                extra_flags += flag + " "
        if self.threaded:
            print("Application is multi-threaded!")
            extra_flags += '-threaded" '
            extra_flags += f'EXTRA_RUNTEST_OPTS="+RTS -N{self.num_of_threads} -RTS'

        extra_flags += '" '
        return extra_flags

    def _build_individual_test_command(self, flag_string, log_name, mode):
        return f'make -C {self.test_path} {flag_string}  NoFibRuns={self.CFG["settings"]["nofib_runs"]} mode={mode} 2>&1 | tee {log_name}'

    def _build_individual_analyze_commands(self, log_list, table, csv_name):
        command_string_list = []
        logs_string = ""
        for log in log_list:
            logs_string += " logs/" + log

        # Can use the --normalise="none" flag to get raw values instead of percentages from the baseline.
        return f'nofib-analyse/nofib-analyse --normalise="none" --csv={table} {logs_string} > analysis/{csv_name}'
        # return f'nofib-analyse/nofib-analyse --csv={table} {logs_string} > analysis/{csv_name}'

    def _run_analysis_tool(self, mode):
        # Run analysis tool and put things into a table.
        # Get a list of all files that we need to analyze
        command_list = []
        logs_list = []

        for entry in self.log_dictionary:
            if self.log_dictionary[entry]["mode"] == mode:
                logs_list.append(entry)
                mode = self.log_dictionary[entry]["mode"]

        # Put them into a command
        output_id = uuid.uuid4()
        runtime_csv_name = f'{self.test_name}-runtime-{mode}-{output_id}.csv'

        # Runtime
        self.csv_dictionary[runtime_csv_name] = output_id
        command_list.append(self._build_individual_analyze_commands(logs_list, "Runtime", runtime_csv_name))

        # Compile Time
        # compiletime_csv_name = f'{self.test_name}-compiletime-{mode}-{output_id}.csv'
        # self.csv_dictionary[compiletime_csv_name] = output_id
        # command_list.append(super()._build_individual_analyze_commands(logs_list, compiletime_csv_name))

        # Elapsed Time
        elapsed_csv_name = f'{self.test_name}-elapsed-{mode}-{output_id}.csv'
        self.csv_dictionary[elapsed_csv_name] = output_id
        command_list.append(self._build_individual_analyze_commands(logs_list, "Elapsed", elapsed_csv_name))

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
            # print(result)

        # Re-Import that CSV and re-configure it the way we want

        # Configure column headers
        headers_ideal = ["ID", "Flags", "Mode", "Runtime", "Elapsed_Time"]  # This is for our combined table later.
        headers_real = ["Program"]
        for log in logs_list:
            headers_real.append(log)

        # print(headers_real)

        run_times = pd.read_csv(self.analysis_dir + "/" + runtime_csv_name, header=None, names=headers_real)
        print(run_times.head())
        # compile_times = pd.read_csv(self.analysis_dir + "/" + self.csv_dictionary[compiletime_csv_name])
        elapsed_times = pd.read_csv(self.analysis_dir + "/" + elapsed_csv_name, header=None, names=headers_real)
        print(elapsed_times.head())

        tables_to_merge = [run_times, elapsed_times]

        # Create the Custom Pandas table

        merged_table = pd.DataFrame(columns=["ID", "Flags", "Mode", "Runtime", "Elapsed_Time"])

        # Configure each row.

        # print(self.log_dictionary)

        # print typeof

        for c in run_times.columns:
            if not (c == "Program"):
                if isinstance(self, IterativeOptimizer):
                    r_id = self.log_dictionary[c]["id"]
                    flags = self.log_dictionary[c]["preset"]
                    m = self.log_dictionary[c]["mode"]
                    r = run_times[c].max()
                elif isinstance(self, GeneticOptimizer):
                    r_id = self.log_dictionary[c]["id"]
                    flags = self.log_dictionary[c]["chromosome"].get_active_genes()
                    m = self.log_dictionary[c]["mode"]
                    r = run_times[c].max()
                else:
                    raise TypeError("What other type of optimizer is there?")
                merged_table.loc[len(merged_table.index)] = [r_id, flags, m, r, None]

        for c in elapsed_times.columns:
            if not (c == "Program"):
                r_id = self.log_dictionary[c]["id"]
                e = run_times[c].max()
                merged_table.loc[merged_table["ID"] == r_id, 'Elapsed_Time'] = e

        # for t in tables_to_merge:
        #     for c in t.columns:
        #         print (c)
        #         if not (c == "Program"):
        #             r_id = self.log_dictionary[c]["id"]
        #             flags = self.log_dictionary[c]["preset"]
        #             m = self.log_dictionary[c]["mode"]
        #             r = t[c].max()
        #             e = t[c].max()
        #
        #             merged_table.loc[len(merged_table.index)] = [r_id, flags, m, r, e]
        #             print(merged_table)
        #             print("---------------------------------------------------\n")

        self.tables[mode] = merged_table

        return merged_table


class IterativeOptimizer(Optimizer, ABC):

    optimizer_number = 0

    def __init__(self, cfg, test_path, t):
        super().__init__(cfg, test_path, t)
        self.num_of_presets = self.CFG["iterative_settings"]["num_of_presets"]
        self.flag_presets = self.__generate_initial_domain()
        self.optimal_preset = None
        self.optimizer_number = IterativeOptimizer.optimizer_number
        IterativeOptimizer.optimizer_number += 1

        print("Iterative Optimizer")

    def __generate_initial_domain(self):
        # Generate a random integer, max size of self.flags. This will determine max size of each preset.
        # Randomly select flags of that size. WITH REPLACEMENT!
        presets = []

        preset_size = np.random.randint(len(self.flags))
        for i in range(self.num_of_presets):
            presets.append(
                [np.append(["-O0"], np.random.choice(self.flags, size=preset_size, replace=True)), uuid.uuid4()])
        return presets

    def optimize(self, mode):
        command_list = []

        self.configure_baseline(mode)

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
        return super()._run_analysis_tool(mode)

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
        # Take the tables in the dictionary and concat them together!

        tables = self.tables.values()
        complete_table = pd.concat(tables)

        complete_table = complete_table.drop_duplicates(keep='first', subset=["ID", "Mode"])

        print(complete_table)

        if not os.path.exists(f'{self.analysis_dir}/{self.test_name}'):
            os.mkdir(f'{self.analysis_dir}/{self.test_name}')

        complete_table.to_csv(f'{self.analysis_dir}/{self.test_name}/{self.test_name}-Iterative-COMPLETE-{self.optimizer_number}.csv')



class BOCAOptimizer(Optimizer, ABC):
    pass


class GeneticOptimizer(Optimizer, ABC):

    def __init__(self, cfg, test_path, t):
        super().__init__(cfg, test_path, t)
        Chromosome.genes = self.CFG["settings"]["flags"]
        Chromosome.num_of_segments = self.CFG["genetic_settings"]["num_of_segments"]
        self.max_iterations = self.CFG["genetic_settings"]["max_iterations"]
        self.chromosomes = self.__generate_initial_population(self.CFG["genetic_settings"]["population_size"])
        self.mutation_prob = self.CFG["genetic_settings"]["mutation_prob"]
        self.elitism_ratio = self.CFG["genetic_settings"]["elitism_ratio"]
        self.crossover_prob = self.CFG["genetic_settings"]["crossover_prob"]
        self.no_improvement_threshold = self.CFG["genetic_settings"]["max_iter_without_improvement"]
        self.base_log_dictionary = dict()
        self.iterations = 0
        self.iterations_with_no_improvement = 0

    def __generate_initial_population(self, pop_size):
        chromosomes = []
        for i in range(0, pop_size):
            rand_active_genes = np.random.randint(len(self.flags))
            active_genes = np.random.choice(self.flags, size=rand_active_genes, replace=False)
            chromosomes.append(Chromosome(active_genes, uuid.uuid4()))
        for c in chromosomes:
            print(c)
        return chromosomes

    def configure_baseline(self, mode):
        baseline_flags = ["-01", "-02"]

        for f in baseline_flags:
            log_file_name = f'{self.test_name}-genetic-{mode}{f}-nofib-log'
            command = super()._build_individual_test_command(super()._setup_preset_task([f]),
                                                             f'{self.CFG["settings"]["log_output_loc"]}/{log_file_name}',
                                                             mode)
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.nofib_exec_path,
                text=True)
            print(result)

        print("Baseline Configured...")

    def optimize(self, mode):
        command_list = []

        if self.iterations == 0:
            self.configure_baseline(mode)

        # Set up command to run benchmark for each chromosome
        for c in self.chromosomes:
            log_file_name = f'{self.test_name}-genetic-{mode}-{c.genetic_id}-{self.iterations}-nofib-log'
            command = super()._build_individual_test_command(super()._setup_preset_task(c.get_active_genes()),
                                                             f'{self.CFG["settings"]["log_output_loc"]}/{log_file_name}',
                                                             mode)
            command_list.append(command)
            self.log_dictionary[log_file_name] = {"chromosome": c, "mode": mode, "id": c.genetic_id}

        # Run each command
        for c in command_list:
            print(fr'Applying command to {self.test_path}')
            result = subprocess.run(
                c,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.nofib_exec_path,
                text=True)
            # print(result)

        self._analyze(mode)

    def _analyze(self, mode):

        old_chromosomes = copy.deepcopy(
            self.chromosomes)  # Used for testing. Will be removed one I have confirmed that crossing over/mutation is successful.

        merged_table = super()._run_analysis_tool(mode)
        print(merged_table)

        merged_table = merged_table.set_index("ID")

        for c in self.chromosomes:
            row = merged_table.loc[[c.genetic_id]]
            c.fitness = row["Runtime"].iloc[0]  # Store fitness value from table into Chromosome

        # Sort Chromosomes by ascending order (Better values at front of the list)
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=False)

        # Selection  by Linear Selection

        # First, we must store the elite chromosomes. These are not ones we will cross and mutate.

        elite_chromosomes = self.chromosomes[:round(len(self.chromosomes) * self.elitism_ratio)]

        non_elite_chromosomes = list(filter(lambda x: x not in elite_chromosomes, self.chromosomes))

        if list(set(elite_chromosomes) & set(non_elite_chromosomes)):  # Checks intersection.
            raise RuntimeError("Duplicates exist within the chromosome list or the filter did not work...")

        # Get the highest performing values that are not 'Elite'

        selected_list = self.__select_via_linear_ranking(non_elite_chromosomes)

        # Crossover by Segment Based Crossover

        crossover_list = self.crossover(selected_list)

        self.chromosomes = set(crossover_list + elite_chromosomes + non_elite_chromosomes)

        # Mutate them by Gauss By Center

    # Use exploration tilted selection pressure.
    def __select_via_linear_ranking(self, lower_ranked_population):

        n = len(lower_ranked_population)
        n_plus = n / 3
        n_minus = 2 * (n / 3)
        selected_list = []

        for i in range(0, n):
            chromosome = lower_ranked_population[i]
            # Calculate ranked probability
            ranked_probability = (1 / n) * (n_minus + (n_plus - n_minus) * ((i - 1) / (n - 1)))
            print(f'{i}: {ranked_probability}')
            if ranked_probability >= random.random():
                selected_list.append(chromosome)

        return selected_list

    def write_results(self):
        pass

    def crossover(self, selected_list, binary_mask=None):

        new_pop_list = []

        # Create the binary mask (Should always be, but I wanted to be able to test it easier)

        if binary_mask is None:
            binary_mask = []
            for i in range(0, Chromosome.num_of_segments):
                if self.crossover_prob >= random.random():
                    binary_mask.append(1)
                else:
                    binary_mask.append(0)

        print(binary_mask)

        # selected_list = [Chromosome([], 0),Chromosome([], 1),Chromosome([], 2)]

        # Use sequential pairing for crossover.

        if len(selected_list) % 2 == 1:
            new_pop_list.append(
                selected_list[len(selected_list) - 1])  # Odd number list need the last element to just be re-introduced

        print("SEQUENTIAL THING!")
        print(f"Original List:  + {selected_list}")
        iterator = iter(selected_list)

        crossing_pairs = list(zip(iterator, iterator))

        print(f'New List: {crossing_pairs}')

        # Perform the crossover

        for pair in crossing_pairs:
            a = pair[0]  # Fitter chromosome
            b = pair[1]  # Less fit chromosome
            b = crossover_chromosomes(a, b, binary_mask)  # Worse performing chromosome is replaced.

            new_pop_list.append(a)
            new_pop_list.append(b)

        return new_pop_list

    def mutate(self):
        pass
