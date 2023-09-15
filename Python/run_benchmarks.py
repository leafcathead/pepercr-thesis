# This application will run several different configuration files on the nofib Haskell benchmarking suite.
import argparse
import os
import subprocess
import json
import multiprocessing

NOFIB_CONFIG_DIR_PATH = r'..\nofib\mk'
NOFIB_EXEC_PATH = r'../nofib'
NOFIB_LOGS_DIR = r'..\nofib\logs'
MY_CONFIG_DIR_PATH = r'ConfigFiles'
FLAG_PRESET_FILE = "presets.json"
TEST_DIRECTORIES = ["gc", "imaginary", "parallel", "real", "shake", "shootout", "smp", "spectral"]
NUM_OF_CONFIG_FILES = 0


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
    if preset['flags']:
        extra_flags = 'EXTRA_HC_OPTS="'
        for flag in preset['flags']:
            extra_flags += flag
        extra_flags += '" '
        apply_preset_task_all(preset['presetName'], extra_flags)
    else:
        print("No flags? What's the point?")
        exit(0)


def apply_preset_task_one():
    print("Apply Preset Task to One Thing")
    #   TODO: Implement


def main():
    flag_presets = []
    threads = []
    print("Main Function Call")

    # Set up Argument Parser
    parser = argparse.ArgumentParser(
        prog='run_benchmarks.py',
        description='This program will run optimize the flags for each program in the benchmark',
        epilog='Must select which optimization algorithm to use')

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--iterative", dest="optimization_type", action="store_const", const=0,
                       help="Use iterative optimization to optimize benchmark.")
    # ADD MORE ARGUMENTS HERE AS THEY BECOME AVAILABLE

    args = parser.parse_args()

    match args.optimization_type:
        case 0:
            print("Iterative Optimization Selected...")
        case _:
            print("Default Selected...")


    # Add the Flag Presets to dictionary. All these presets will be run once per config file.
    try:
        with open(FLAG_PRESET_FILE, "r") as preset_file:
            for line in preset_file:
                flag_presets.append(json.loads(line))

        # Multithreading currently broken. I think the problem is with nofib. I think the quickest thread 'hijacks'
        # the compilation of the tests which causes the other threads to throw an error and exit prematurely. Testing
        # right now to see if the problem goes away during sequential execution.
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            pool.map(setup_preset_task, flag_presets)
            pool.close()
            pool.join()

    except IOError:
        print("Unable to open flag preset file...")

        print("All threads have finished executing...")

        # TODO: Extra compilation steps required to benchmark multi-threaded applications

    print("All configuration files have finished executing...")


if __name__ == "__main__":
    main()
