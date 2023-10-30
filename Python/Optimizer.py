import math
import os
import random
from abc import ABC, abstractmethod
import numpy as np
import subprocess
import uuid
import pandas as pd
from eliot import log_call, to_file, log_message, start_action
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import date

from Genetics import crossover_chromosomes
from Genetics import Chromosome
import copy  # Used for testing


# CONFIG_PATH = r'ConfigFiles/config.yaml'


class Optimizer(ABC):

    def __init__(self, cfg, test_path, t, test_desc="COMPLETE"):
        self.CFG = cfg
        self.threaded = t
        self.flags = self.CFG["settings"]["flags"]
        self.num_of_threads = cfg["settings"]["multicore_cores"]
        self.nofib_log_dir = cfg["locations"]["nofib_logs_dir"]
        self.nofib_exec_path = cfg["locations"]["nofib_exec_path"]
        self.analysis_dir = cfg["locations"]["nofib_analysis_dir"]
        self.analysis_exec_path = cfg["locations"]["nofib_analysis_exec"]
        self.run_allowance = cfg["settings"]["run_allowance"]
        self.test_path = test_path
        self.test_name = test_path.split("/")[1]  # Gets only the test name. Slices the test category.
        self.log_dictionary = dict()
        self.csv_dictionary = dict()
        self.label = test_desc
        self.tables = {"slow": None, "norm": None, "fast": None}
        try:
            to_file(open(f'{cfg["locations"]["log_location"]}/{self.test_name}-{self.label}-{date.today()}.log', "a+"))
            self.ctx = start_action(action_type="LOG_RESULTS")
        except IOError as e:
            print("Unable to open logging file...")
            print("Optimizer results will not be logged...")
            print(e)

    @abstractmethod
    def configure_baseline(self, mode):
        pass

    @abstractmethod
    def optimize(self, mode):
        pass

    @abstractmethod
    def _analyze(self, mode):
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
        return (
            f'echo {logs_string}', f'xargs nofib-analyse/nofib-analyse --normalise="none" --csv={table}', f"{csv_name}")
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
            with open(f"logs_tmp.txt", "w") as f:
                result = subprocess.run(
                    c[0],
                    shell=True,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    cwd=self.nofib_exec_path,
                    text=True)
            with open(f"logs_tmp.txt", "r") as f:
                with open(f"{self.analysis_dir}/{c[2]}", "w") as output_csv:
                    result2 = subprocess.run(
                        c[1],
                        shell=True,
                        stdin=f,
                        stdout=output_csv,
                        stderr=subprocess.PIPE,
                        cwd=self.nofib_exec_path,
                        text=True)

            # p1.stdout.close()
            # output = result #p2.communicate()[0]
            print(f"Result1: {result}")
            print(f"Result2: {result2}")

        # Re-Import that CSV and re-configure it the way we want

        # Configure column headers
        headers_ideal = ["ID", "Flags", "Mode", "Runtime", "Elapsed_Time"]  # This is for our combined table later.
        headers_real = ["Program"]
        for log in logs_list:
            headers_real.append(log)

        # print(headers_real)

        run_times = pd.read_csv(self.analysis_dir + "/" + runtime_csv_name, header=None, names=headers_real)
        # compile_times = pd.read_csv(self.analysis_dir + "/" + self.csv_dictionary[compiletime_csv_name])
        elapsed_times = pd.read_csv(self.analysis_dir + "/" + elapsed_csv_name, header=None, names=headers_real)

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
                elif isinstance(self, BOCAOptimizer):
                    r_id = self.log_dictionary[c]["id"]
                    flags = self.log_dictionary[c]["BOCA"].flags
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

        
        print(f"Merged_table: {merged_table}")

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

        if (self.tables[mode] is None):
            self.tables[mode] = merged_table
        else:
            self.tables[mode] = pd.concat([self.tables[mode], merged_table])
            self.tables[mode].drop_duplicates(subset=["ID"], keep="last", inplace=True)

        return self.tables[mode]

    @abstractmethod
    def write_results(self):
        # Take the tables in the dictionary and concat them together!

        tables = self.tables.values()
        complete_table = pd.concat(tables)

        complete_table = complete_table.drop_duplicates(keep='first', subset=["ID", "Mode"])

        if not os.path.exists(f'{self.analysis_dir}/{self.test_name}'):
            os.mkdir(f'{self.analysis_dir}/{self.test_name}')

        return complete_table


class IterativeOptimizer(Optimizer, ABC):
    optimizer_number = 0

    @log_call(action_type="OPTIMIZER_CREATION", include_args=["test_path", "test_desc"], include_result=False)
    def __init__(self, cfg, test_path, t, test_desc="COMPLETE"):
        super().__init__(cfg, test_path, t, test_desc)
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
            self.run_allowance -= 1
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
        returner = super()._run_analysis_tool(mode)
        # self.log_dictionary.clear()
        return returner

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
        complete_table = super().write_results()

        best_result = (complete_table.sort_values("Runtime", ascending=True).loc[0, "Flags"],
                       complete_table.sort_values("Runtime", ascending=True).loc[0, "Runtime"])

        with start_action(action_type="LOG_RESULTS") as ctx:
            ctx.log(message_type="INFO", optimizer="RIO", no=f"{self.optimizer_number}", best_result=best_result,
                    run_allowance=self.run_allowance, iterations=self.num_of_presets)

        complete_table.to_csv(
            f'{self.analysis_dir}/{self.test_name}/{self.test_name}-Iterative-{self.label}-{self.optimizer_number}.csv')


class BOCAOptimization:
    iteration = 0

    def __init__(self, flags, flag_b):
        self.id = uuid.uuid4()
        if not isinstance(flags, list):
            raise ValueError("BOCA flags must be of type LIST")
        for f in flags:
            if not isinstance(f, str):
                raise ValueError(f"BOCA flag must be of type STRING. Received {f} which is type {type(f)}")
        self.flags = flags
        self.flag_bits = flag_b
        self.runtime = -1
        self.expected_improvement = 0
        self.iteration_created = BOCAOptimization.iteration


class BOCAOptimizer(Optimizer, ABC):
    optimizer_number = 0

    @log_call(action_type="OPTIMIZER_CREATION", include_args=["test_path", "test_desc"], include_result=False)
    def __init__(self, cfg, test_path, t, test_desc="COMPLETE"):
        super().__init__(cfg, test_path, t, test_desc)
        self.__c1 = cfg["boca_settings"]["initial_set"]
        self.training_set = self.__generate_training_set(self.__c1)
        self.num_of_K = cfg["boca_settings"]["num_of_impactful_optimizations"]
        self.baseline_set = dict()
        self.max_iterations = cfg["boca_settings"]["iterations"]
        self.decay = cfg["boca_settings"]["decay"]
        self.offset = cfg["boca_settings"]["offset"]
        self.scale = cfg["boca_settings"]["scale"]
        self.iterations = 0
        self.best_candidate = None
        self.optimizer_number = BOCAOptimizer.optimizer_number

    def __generate_training_set(self, set_size):
        init_set = []
        for i in range(0, set_size):
            rand_active_flags_num = np.random.randint(len(self.flags))
            random_choices = list(np.random.choice(self.flags, size=rand_active_flags_num, replace=False))
            bit_random_choices = list(map(lambda x: 1 if x in random_choices else 0, self.flags))
            init_set.append(
                BOCAOptimization(["-O0"] + random_choices, bit_random_choices))
        return init_set

    def __boca_to_df(self, mode):
        boca_table = pd.DataFrame(columns=["ID", "Mode", "Flags", "Runtime", "Iteration", "Best"])
        for entry in self.baseline_set:
            # chromosome_table.loc[len(chromosome_table.index)] = [entry, mode, [entry], self.base_log_dictionary[entry]["Runtime"]]
            boca_table.loc[len(boca_table.index)] = [entry, mode, entry,
                                                     self.baseline_set[entry]["Runtime"].iloc[0], 0, False]

        for b in self.training_set:
            boca_table.loc[len(boca_table.index)] = [b.id, mode, b.flags, b.runtime, b.iteration_created,
                                                     (lambda x: x is self.best_candidate)(b)]

        return boca_table

    def configure_baseline(self, mode):

        baseline_flags = ["-O0", "-O2"]

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

            str1 = ""
            self.log_dictionary[log_file_name] = {"BOCA": BOCAOptimization([f], str1.zfill(len(self.flags))),
                                                  "mode": mode, "id": f}
            table = self._run_analysis_tool(mode)
            self.log_dictionary.clear()
            self.baseline_set[f] = table

        print("Baseline Configured...")

    def optimize(self, mode):
        print("Beginning Optimization")
        print(f"Iteration: {self.iterations}")

        if self.iterations == 0:
            self.configure_baseline(mode)

        self.csv_dictionary.clear()
        self.log_dictionary.clear()

        command_list = []

        for b in self.training_set:

            log_file_name = f'{self.test_name}-BOCA-{mode}-{b.id}-nofib-log'

            if self.log_dictionary.get(log_file_name) is None and b.runtime == -1:
                # Set up command to run benchmark for each BOCA Optimization


                command = super()._build_individual_test_command(super()._setup_preset_task(b.flags),
                                                                 f'{self.CFG["settings"]["log_output_loc"]}/{log_file_name}',
                                                                 mode)
                command_list.append(command)
                self.log_dictionary[log_file_name] = {"BOCA": b, "mode": mode, "id": b.id}

        # Run each command
        for c in command_list:
            self.run_allowance -= 1
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

        merged_table = super()._run_analysis_tool(mode)
        merged_table = merged_table.set_index("ID")

        print(merged_table)

        for b in self.training_set:
            row = merged_table.loc[[b.id]]
            b.runtime = row["Runtime"].iloc[0]  # Store fitness value from table into Chromosome

        self.best_candidate = min(self.training_set, key=lambda x: x.runtime)
        print("Current Best Candidate: ", self.best_candidate)

        if (self.iterations == self.max_iterations) or (self.run_allowance <= 0):
            print("Max Iterations reached...")
            self.iterations = 0
            self.tables[mode] = self.__boca_to_df(mode)
            return

        rf = RandomForestRegressor()
        X_train = list(map(lambda x: x.flag_bits, self.training_set))
        y_train = list(map(lambda x: x.runtime, self.training_set))

        if len(X_train) != len(y_train):
            raise RuntimeError("Somehow we have more runtimes than we do presets!")

        # print(X_train)
        # print("\n")
        # print(y_train)
        # np.array(X_train)
        # np.array(y_train)

        rf.fit(X_train, y_train)

        # Determine importance
        importance = []
        for index, f in enumerate(self.flags):
            importance.append((rf.feature_importances_[index], self.flags[index]))

        # Determine Importance Opts

        important_optimizations = self.__get_important_optimizations(rf, importance)

        # Determine Unimportant Opts

        unimportant_optimizations = list(set(self.flags) - set(important_optimizations))

        # Do Decay Stuff
        important_settings = [obj for obj in self.training_set if any(item in obj.flags for item in important_optimizations)]
        important_settings = list(map(lambda boca_obj: boca_obj.flags , list(set(important_settings))))
        ##print("Important Settings: ", important_settings)
        all_candidates = []

        for index, optimization in enumerate(important_settings):
            C = self.__normal_decay(self.iterations)
        #    if C > len(unimportant_optimizations):
         #       C = len(unimportant_optimizations)
            new_candidate_flags = optimization + list(np.random.choice(unimportant_optimizations, size=int(C), replace=False))
            new_candidate_flags = list(set(new_candidate_flags))
            all_candidates.append(BOCAOptimization(["-O0"] + new_candidate_flags, list(
                map(lambda x: 1 if x in new_candidate_flags else 0, self.flags))))

        # Predict

        for index, candidate in enumerate(all_candidates):
            # candidate = (lambda A, B, e: B[A.index(e)] if e in A else None)(self.flags,)
            results = []
            trees = rf.estimators_
            for t in trees:
                result = t.predict(np.array(candidate.flag_bits).reshape(1, -1))
                results.append(result)
            # rf.predict(np.array(candidate.flag_bits).reshape(1,-1)) # Breaks with the paper. Paper has mean and std
            # print(f'Result: {result}')
            candidate.expected_improvement = self.__get_expected_improvement(results)

        # Find Best Candidate
        all_candidates = list(
            filter(lambda x: x.flag_bits not in list(map(lambda y: y.flag_bits, self.training_set)), all_candidates))

        if len(all_candidates) > 0:
            best_candidate = max(all_candidates, key=lambda x: x.expected_improvement)
            # Add to training set
            self.training_set.append(best_candidate)

        else:
            print("No unique candidate found. Should probably error or stop here. I'm not sure which one.")
            with start_action(action_type="ERROR_CANDIDATE") as ctx:
                ctx.log(message_type="ERROR", optimizer="BOCA", message="No unique candidate found.", iteration=self.iterations, all_candidates=all_candidates)


        print("Analysis Done")

        # Re-run optimize

        BOCAOptimization.iteration += 1
        self.iterations += 1
        self.optimize(mode)

    def __normal_decay(self, iterations):
        sigma = -((self.scale ** 2) / (2 * math.log2(self.decay)))  # Assuming it's log2, since this is CS afterall lol.
        #C = self.__c1 * math.exp(-((max(0, iterations - self.offset) ** 2) / 2 * (sigma ** 2)))

        C = self.__c1 * math.exp(-max(0, (self.__c1 + iterations) - self.offset) ** 2 / (2 * sigma ** 2))


        #with start_action(action_type="LOG_DECAY") as ctx:
         #               ctx.log(message_type="INFO", iteration=iterations, C=C, C_1=self.__c1)

        return C

    # THIS RESULTS IN THE SAME CALC AS GINI-IMPORTANCE! WHY DO IT?
    def __get_important_optimizations(self, model, gini_list):
        # gini_list.sort(key=lambda x: x[0], reverse=True)
        importance = []
        decision_trees = model.estimators_

        for index, gini_tuple in enumerate(gini_list):
            impact = 0
            for t in decision_trees:
                impact += t.feature_importances_[index]
            impact /= len(decision_trees)
            importance.append((impact, gini_tuple[0], gini_tuple[1]))  # FORMAT: (Impact, Gini, Flag)

        importance.sort(key=lambda x: x[0], reverse=True)

        return list(map(lambda x: x[2], importance[0:self.num_of_K]))

    def __get_expected_improvement(self, pred):
        pred = np.array(pred).transpose(1, 0)
        m = np.mean(pred, axis=1)
        s = np.std(pred, axis=1)

        def calculate_f():
            z = (self.best_candidate.runtime - m) / s
            return (self.best_candidate.runtime - m) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()

        return f

    def write_results(self):
        complete_table = super().write_results()

        complete_table = complete_table[complete_table["Runtime"] >= 0]

        print(complete_table)
        print(f"Best Candidate: \n"
              + f"   Runtime: {self.best_candidate.runtime} \n"
              + f"   Flags: {self.best_candidate.flags} \n")

        with start_action(action_type="LOG_RESULTS") as ctx:
            ctx.log(message_type="INFO", optimizer="BOCA", no=f"{self.optimizer_number}",
                    best_result=(self.best_candidate.flags, self.best_candidate.runtime),
                    run_allowance=self.run_allowance, iterations=self.iterations)

        complete_table.to_csv(
            f'{self.analysis_dir}/{self.test_name}/{self.test_name}-BOCA-{self.label}-{self.optimizer_number}.csv')


class GeneticOptimizer(Optimizer, ABC):
    optimizer_number = 0

    @log_call(action_type="OPTIMIZER_CREATION", include_args=["test_path", "test_desc"], include_result=False)
    def __init__(self, cfg, test_path, t, test_desc="COMPLETE"):
        super().__init__(cfg, test_path, t, test_desc)
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
        self.best_value = None
        self.optimizer_number = GeneticOptimizer.optimizer_number
        GeneticOptimizer.optimizer_number += 1
        self.initial_size = self.CFG["genetic_settings"]["population_size"]

    def __chromosomes_to_df(self, mode):
        chromosome_table = pd.DataFrame(columns=["ID", "Mode", "Flags", "Fitness"])
        for entry in self.base_log_dictionary:
            # chromosome_table.loc[len(chromosome_table.index)] = [entry, mode, [entry], self.base_log_dictionary[entry]["Runtime"]]
            chromosome_table.loc[len(chromosome_table.index)] = [entry, mode, entry,
                                                                 self.base_log_dictionary[entry]["Runtime"].iloc[0]]

        for chromosome in self.chromosomes:
            chromosome_table.loc[len(chromosome_table.index)] = [chromosome.genetic_id, mode, list(
                filter(lambda key: chromosome.sequence[key] == 1, chromosome.sequence.keys())), chromosome.fitness]

        return chromosome_table

    def __generate_initial_population(self, pop_size):
        chromosomes = []
        for i in range(0, pop_size):
            rand_active_genes = np.random.randint(len(self.flags))
            active_genes = np.random.choice(self.flags, size=rand_active_genes, replace=False)
            chromosomes.append(Chromosome(active_genes, uuid.uuid4()))

        return chromosomes

    def configure_baseline(self, mode):
        baseline_flags = ["-O0", "-O2"]

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

            self.log_dictionary[log_file_name] = {"chromosome": Chromosome([f], f), "mode": mode, "id": f}
            table = self._run_analysis_tool(mode)
            self.log_dictionary.clear()
            self.base_log_dictionary[f] = table

        print("Baseline Configured...")

    def optimize(self, mode):
        print(f"Iteration: {self.iterations}")

        self.csv_dictionary.clear()
        
        tmp_dict = dict()

        print("Length of Chromosome List: ", len(self.chromosomes))

        for log_name, log_listing in self.log_dictionary.items():
            for c in self.chromosomes:
                if c == log_listing["chromosome"]:
                    tmp_dict[log_name] = log_listing
                

        self.log_dictionary.clear()
        self.log_dictionary.update(tmp_dict)


        if (self.iterations >= self.max_iterations) or (self.run_allowance <= 0):
            print("Max Iterations Reached... Terminating")
            self.iterations = 0
            self.tables[mode] = self.__chromosomes_to_df(mode)
            return
        elif self.iterations_with_no_improvement >= self.no_improvement_threshold:
            print("No improvement threshold reached... Terminating")
            self.iterations = 0
            self.tables[mode] = self.__chromosomes_to_df(mode)
            return

        command_list = []

        if self.iterations == 0:
            self.configure_baseline(mode)

        # Set up command to run benchmark for each chromosome
        for c in self.chromosomes:

            if c.need_run:
                log_file_name = f'{self.test_name}-genetic-{mode}-{c.genetic_id}-{self.iterations}-nofib-log'
                command = super()._build_individual_test_command(super()._setup_preset_task(c.get_active_genes()),
                                                                 f'{self.CFG["settings"]["log_output_loc"]}/{log_file_name}',
                                                                 mode)
                command_list.append(command)
                self.log_dictionary[log_file_name] = {"chromosome": c, "mode": mode, "id": c.genetic_id}
                c.need_run = False

        # Run each command
        for c in command_list:
            self.run_allowance -= 1
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

        merged_table = super()._run_analysis_tool(mode)
        # self.log_dictionary.clear()

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

        if len(elite_chromosomes) + len(non_elite_chromosomes) > self.initial_size:
            raise RuntimeError("Population growing error... before linear selection")

        # Get the highest performing values that are not 'Elite'

        selected_list = self.__select_via_linear_ranking(non_elite_chromosomes)
        for c in selected_list:
            c.need_run = True

        # Crossover by Segment Based Crossover

        crossover_list = self.crossover(selected_list)

        self.chromosomes = elite_chromosomes + list(
            (set(crossover_list)) | (set(non_elite_chromosomes) - set(selected_list)))

        if len(self.chromosomes) > self.initial_size:
            raise RuntimeError("Population growing error... after cross over")

        # Mutate them by Gauss By Center OR Bit mask

        self.mutate_genes_via_bitmask()

        # Now, with everything complete, begin optimize again.

        self.chromosomes.sort(key=lambda x: x.fitness, reverse=False)
        best_value = self.chromosomes[0].fitness

        if self.best_value is None or best_value < self.best_value:
            self.best_value = best_value
            self.iterations_with_no_improvement = 0
        else:
            self.iterations_with_no_improvement += 1

        self.iterations += 1
        self.optimize(mode)

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
            if ranked_probability >= random.random():
                selected_list.append(chromosome)

        return selected_list

    def mutate_genes_via_bitmask(self, bit_mask=None):

        for chromosome in self.chromosomes:
            chromosome.mutate_via_bitmask(self.mutation_prob)

    def write_results(self):
        complete_table = super().write_results()

        complete_table = complete_table[complete_table["Fitness"] >= 0]

        best_candidate = max(self.chromosomes, key=lambda x: x.fitness)

        print(complete_table)

        with start_action(action_type="LOG_RESULTS") as ctx:
            ctx.log(message_type="INFO", optimizer="GA", no=f"{self.optimizer_number}",
                    best_result=(best_candidate.genes, best_candidate.fitness), run_allowance=self.run_allowance,
                    iterations=self.iterations)

        complete_table.to_csv(
            f'{self.analysis_dir}/{self.test_name}/{self.test_name}-Genetic-{self.label}-{self.optimizer_number}.csv')

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

        # Use sequential pairing for crossover.

        if len(selected_list) % 2 == 1:
            new_pop_list.append(
                selected_list[len(selected_list) - 1])  # Odd number list need the last element to just be re-introduced

        iterator = iter(selected_list)

        crossing_pairs = list(zip(iterator, iterator))

        # Perform the crossover

        for pair in crossing_pairs:
            a = pair[0]  # Fitter chromosome
            b = pair[1]  # Less fit chromosome
            b = crossover_chromosomes(a, b, binary_mask)  # Worse performing chromosome is replaced.

            new_pop_list.append(a)
            new_pop_list.append(b)

        if len(new_pop_list) > len(selected_list):
            raise RuntimeError("Growth in population during cross over")

        return new_pop_list
