# This application will run several different configuration files on the nofib Haskell benchmarking suite.
import argparse
import os
import subprocess
import json
import yaml
import multiprocessing
from Optimizer import IterativeOptimizer

NOFIB_CONFIG_DIR_PATH = r'..\nofib\mk'
NOFIB_EXEC_PATH = r'../nofib'
NOFIB_LOGS_DIR = r'..\nofib\logs'
CONFIG_PATH = r'ConfigFiles/config.yaml'
FLAG_PRESET_FILE = "presets.json"
TEST_DIRECTORIES = ["imaginary", "parallel", "real", "shake", "shootout", "smp", "spectral"]
CFG = None


def build_individual_test_command(flag_string, process_name):
    # return f'make -C {process_name} {flag_string} NoFibRuns=10 2>&1 | tee logs/{process_name}-nofib-log'
    return f'make -C {process_name} {flag_string}  NoFibRuns={CFG["settings"]["nofib_runs"]} 2>&1 | tee {CFG["settings"]["log_output_loc"]}what100-nofib-log'


def apply_preset_task_all(process_name, flag_string):
    print("Applying Preset: " + process_name)
    command = f"make {flag_string} 2>&1 | tee {process_name}-nofib-log"
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=NOFIB_EXEC_PATH,
        text=True)
    print("Thread applying preset " + process_name + " has completed...")


def setup_preset_task(preset):
    extra_flags = ""
    if preset:
        extra_flags = 'EXTRA_HC_OPTS="'
        for flag in preset:
            extra_flags += flag
        extra_flags += '" '
        return extra_flags
    # apply_preset_task_all(preset['presetName'], extra_flags)
    else:
        print("No flags? What's the point?")
        return ""


def apply_optimizer_task_all(optimizer):
    tests = []

    print("Apply Preset Task to all things")
    #   TODO: Implement


def apply_optimizer_task_one(optimizer, test):
    tests = []
    print(f'Apply Preset Task to: {test}')

    command = build_individual_test_command(setup_preset_task(optimizer.flag_preset), test)
    print(command)
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=NOFIB_EXEC_PATH,
        text=True)
    #   TODO: Implement


def main():
    flag_presets = []
    threads = []
    optimizer = ''
    print("Main Function Call")

    # Set up Argument Parser
    parser = argparse.ArgumentParser(
        prog='run_benchmarks.py',
        description='This program will run optimize the flags for each program in the benchmark',
        epilog='Must select which optimization algorithm to use')

    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument("--all", action="store_true", help="Designate whether one test or all test should be ran.")
    file_group.add_argument("-f", metavar="file_path", help="File path to the test to be optimized")

    optimizer_group = parser.add_mutually_exclusive_group()
    optimizer_group.add_argument("--iterative", dest="optimization_type", action="store_const", const=0,
                                 help="Use iterative optimization to optimize benchmark.")
    # ADD MORE ARGUMENTS HERE AS THEY BECOME AVAILABLE

    args = parser.parse_args()

    # Add the Flag Presets to dictionary. All these presets will be run once per config file.
    try:
        with open(FLAG_PRESET_FILE, "r") as preset_file:
            for line in preset_file:
                flag_presets.append(json.loads(line))

        # Multithreading currently broken. I think the problem is with nofib. I think the quickest thread 'hijacks'
        # the compilation of the tests which causes the other threads to throw an error and exit prematurely. Testing
        # right now to see if the problem goes away during sequential execution.

        match args.optimization_type:
            case 0:
                print("Iterative Optimization Selected...")
                optimizer = IterativeOptimizer(CFG["settings"]["flags"])
            case _:
                print("Default Selected...")

        if args.all:
            apply_optimizer_task_all(optimizer)
        else:
            apply_optimizer_task_one(optimizer, args.f)

        # with multiprocessing.get_context("spawn").Pool(1) as pool:
        #     pool.map(setup_preset_task, flag_presets)
        #     pool.close()
        #     pool.join()

    except IOError:
        print("Unable to open flag preset file...")

        print("All threads have finished executing...")

        # TODO: Extra compilation steps required to benchmark multi-threaded applications

    print("All configuration files have finished executing...")


if __name__ == "__main__":

    try:
        with open(CONFIG_PATH, "r") as cfg_file:
            CFG = yaml.load(cfg_file)

        if CFG is None:
            raise IOError("CFG File is blank!")

        main()

    except IOError as e:
        print("Unable to open Configuration file")
        print(e)
