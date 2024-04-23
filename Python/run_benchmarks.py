# This application will run several different configuration files on the nofib Haskell benchmarking suite.
import argparse
import os
import subprocess
import json
import yaml
from multiprocessing import Process, Lock, cpu_count, Pool
from Optimizer import IterativeOptimizer, GeneticOptimizer, BOCAOptimizer, IterativeOptimizerPO, GeneticOptimizerPO, BOCAOptimizerPO


NOFIB_CONFIG_DIR_PATH = r'..\nofib\mk'
NOFIB_EXEC_PATH = r'../nofib'
NOFIB_EXEC_PATH_PO = r'../../ghc/nofib'
NOFIB_LOGS_DIR = r'..\nofib\logs'
CONFIG_PATH = r'ConfigFiles/config.yaml'
FLAG_PRESET_FILE = "presets.json"
TEST_DIRECTORIES = ["imaginary", "real", "shake", "shootout", "smp", "spectral"]
TEST_PROGRAMS = ["spectral/sorting", "real/hidden", "real/cacheprof", "real/maillist", "real/prolog", "real/symalg", "spectral/primetest", "spectral/integer", "spectral/power", "imaginary/primes" "real/fulsom","real/rsa","real/fluid","real/parser","real/grep","spectral/calendar","spectral/gcd","spectral/lambda","spectral/mandel","spectral/sphere","shootout/binary-trees"]
CFG = None


# def build_individual_test_command(flag_string, process_name): # return f'make -C {process_name} {flag_string}
# NoFibRuns=10 2>&1 | tee logs/{process_name}-nofib-log' return f'make -C {process_name} {flag_string}  NoFibRuns={
# CFG["settings"]["nofib_runs"]} 2>&1 | tee {CFG["settings"]["log_output_loc"]}what101-nofib-log'


def apply_optimizer_task_all(my_tuple):

    # if mode[0]:
    #     optimizer.optimize("slow")
    # if mode[1]:
    #     optimizer.optimize("norm")
    # if mode[2]:
    #     optimizer.optimize("fast")
    optimizer = my_tuple[0]
    process_name = my_tuple[1]
    mode = my_tuple[2]
    print("Applying Preset: " + process_name)
    optimizer.optimize("fast")
    try:
        optimizer.write_results()
    except IOError as e:
        print("Error writing results to file...")
        print(e)
    print("Thread applying preset " + process_name + " has completed...")


# def setup_preset_task(preset):
#     extra_flags = ""
#     if len(preset) > 0:
#         extra_flags = 'EXTRA_HC_OPTS="'
#         for flag in preset:
#             extra_flags += flag + " "
#         extra_flags += '" '
#         return extra_flags
#     # apply_preset_task_all(preset['presetName'], extra_flags)
#     else:
#         print("No flags? What's the point?")
#         return ""


# def apply_optimizer_task_all(optimizer):
#     tests = []

#     print("DEPRECIATED! WILL REMOVE WHEN I GET AROUND TO IT!")
#     #   TODO: Implement


# Don't forget to run each program with the different levels of difficulty! See nofib documentation!
def apply_optimizer_task_one(optimizer_list, test, modes):
    tests = []
    print(f'Apply Preset Task to: {test}')


    print("Clean and build complete...")

    for optimizer in optimizer_list:
        if modes[0]:
            optimizer.optimize("slow")
        if modes[1]:
            optimizer.optimize("norm")
        if modes[2]:
            optimizer.optimize("fast")

        try:
            optimizer.write_results()
        except IOError as e:
            print("Error writing results to file...")
            print(e)

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

    parser.add_argument("--threaded", required=False, action="store_true", help="Select if the application is intended to be run "
                                                                "multithreaded.")

    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument("--all", action="store_true", help="Designate whether one test or all test should be ran.")
    file_group.add_argument("-f", metavar="file_path", help="File path to the test to be optimized")

    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument("--phase", action="store_true", help="Optimizes for Phase Order insteal of compiler flags. Uses -O2 as default")
    # file_group.add_argument("-f", metavar="file_path", help="File path to the test to be optimized")

    optimizer_group = parser.add_mutually_exclusive_group()
    optimizer_group.add_argument("--iterative", dest="optimization_type", action="store_const", const=0,
                                 help="Use iterative optimization to optimize benchmark.")
    optimizer_group.add_argument("--genetic", dest="optimization_type", action="store_const", const=1,
                                 help="Use genetic optimization to optimize benchmark.")
    optimizer_group.add_argument("--boca", dest="optimization_type", action="store_const", const=2,
                                 help="Use BOCA to optimize benchmark.")

    #mode_group = parser.add_mutually_exclusive_group()
    parser.add_argument("-sm", "--slow", dest="slow", action="store_true",
                                 help="Run the program in only slow mode. Default is to run in all three modes.")
    parser.add_argument("-nm", "--norm", dest="norm", action="store_true",
                                 help="Run the program in only normal mode. Default is to run in all three modes.")
    parser.add_argument("-fm", "--fast", dest="fast", action="store_true",
                                 help="Run the program in only fast mode. Default is to run in all three modes.")

    parser.add_argument("--name", required=False, help="Give a special name to the output file to help distinguish it from other tests.")

    args = parser.parse_args()

    if not(args.slow or args.norm or args.fast):
        mode_list = [1,1,1]
    else:
        mode_list = [args.slow, args.norm, args.fast]

    # Add the Flag Presets to dictionary. All these presets will be run once per config file.
    try:
        with open(FLAG_PRESET_FILE, "r") as preset_file:
            for line in preset_file:
                flag_presets.append(json.loads(line))

        # Multithreading currently broken. I think the problem is with nofib. I think the quickest thread 'hijacks'
        # the compilation of the tests which causes the other threads to throw an error and exit prematurely. Testing
        # right now to see if the problem goes away during sequential execution.

        optimizer_list = []

        print(f'Cleaning and building nofib...')
        command = f"make clean && make boot"
        # result_1 = subprocess.run(
        #     command,
        #     shell=True,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     cwd=NOFIB_EXEC_PATH,
        #     text=True)

        # result_2 = subprocess.run(
        #     command,
        #     shell=True,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     cwd=NOFIB_EXEC_PATH_PO,
        #     text=True)

        match args.optimization_type:
            case 0:
                print("Iterative Optimization Selected...")
                for i in range(0, CFG["settings"]["num_of_optimizer_runs"]):
                    if args.phase:
                        # Phase Order Stuff Here
                        optimizer = IterativeOptimizerPO(CFG, args.f, args.threaded, args.name)
                    else:
                        optimizer = IterativeOptimizer(CFG, args.f, args.threaded, args.name)
                    optimizer_list.append(optimizer)
            case 1:
                print("Genetic Optimization Selected...")
                for i in range(0, CFG["settings"]["num_of_optimizer_runs"]):
                    if args.phase:
                        # Phase Order Stuff Here
                        optimizer = GeneticOptimizerPO(CFG, args.f, args.threaded, args.name)
                    else:
                        optimizer = GeneticOptimizer(CFG, args.f, args.threaded, args.name)
                    optimizer_list.append(optimizer)
            case 2:
                print("BOCA Selected...")
                for i in range(0, CFG["settings"]["num_of_optimizer_runs"]):
                    if args.phase:
                        # Phase Order Stuff Here
                        optimizer = BOCAOptimizerPO(CFG, args.f, args.threaded, args.name)
                    else:
                        optimizer = BOCAOptimizer(CFG, args.f, args.threaded, args.name)
                    optimizer_list.append(optimizer)
            case _:
                raise ValueError("Invalid optimization type.")

        if args.all:
            print("All selected...")

            p_threads = []
            for program in TEST_PROGRAMS:
                p_threads.append((BOCAOptimizerPO(CFG, program, args.threaded, args.name), program, "fast"))
                # p_threads.append(Process(target=apply_optimizer_task_all, args=(optimizer,program,"fast")) )
                        #optimizer_list, test, modes
                # for t in p_threads:
                #     t.start()
                # for t in p_threads:
                #     t.join()
            with Pool(min(30, len(TEST_PROGRAMS))) as p:
                p.map(apply_optimizer_task_all, p_threads)
        else:
            print("One selected...")
            apply_optimizer_task_one(optimizer_list, args.f, mode_list)

        # with multiprocessing.get_context("spawn").Pool(1) as pool:
        #     pool.map(setup_preset_task, flag_presets)
        #     pool.close()
        #     pool.join()

    except IOError as e:
        print("Unable to open flag preset file...")
        print(e)

    print("All threads have finished executing...")

    print("All configuration files have finished executing...")


if __name__ == "__main__":

    try:
        with open(CONFIG_PATH, "r") as cfg_file:
            CFG = yaml.safe_load(cfg_file)

        if CFG is None:
            raise IOError("CFG File is blank!")

        main()

    except IOError as e:
        print("Unable to open Configuration file")
        print(e)
