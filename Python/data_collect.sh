#!/bin/bash

# Define programs
arguments=("spectral/sorting" "real/grep" "real/cacheprof")
threaded_programs=("parallel/parfib")

threaded_flag=false
iterative_flag=false
genetic_flag=false
boca_flag=false
name_arg=""

while getopts "tigbn:" opt; do
    case $opt in
        t)
            threaded_flag=true;;
        i)
            iterative_flag=true;;
        g)
            genetic_flag=true;;
        b)
            boca_flag=true;;
        n)
            name_arg="$OPTARG";;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done


# Additional logic for --iterative, --genetic, --boca, and "optimizer"
if [ "$boca_flag" = true ]; then
    # Threaded Stuff if threaded_flag TRUE
	if [ "$threaded_flag" = true ]; then
		for arg in "${threaded_programs[@]}"; do
			echo "Running threaded program: $arg"
			python3 run_benchmarks.py  --boca -f "$arg" --fast --name "$name_arg"
		done
	fi

	# Loop through the arguments and run the Python program
	for arg in "${arguments[@]}"; do
		echo "Running Python program with argument: $arg"
		python3 run_benchmarks.py --boca -f "$arg" --fast --name "$name_arg"
	done
	
fi

if [ "$genetic_flag" = true ]; then
    # Threaded Stuff if threaded_flag TRUE
	if [ "$threaded_flag" = true ]; then
		for arg in "${threaded_programs[@]}"; do
			echo "Running threaded program: $arg"
			python3 run_benchmarks.py --genetic --threaded -f "$arg" --fast --name "$name_arg"
		done
	fi

	# Loop through the arguments and run the Python program
	for arg in "${arguments[@]}"; do
		echo "Running Python program with argument: $arg"
		python3 run_benchmarks.py --genetic -f "$arg" --fast --name "$name_arg"
	done
fi

if [ "$iterative_flag" = true ]; then
        # Threaded Stuff if threaded_flag TRUE
	if [ "$threaded_flag" = true ]; then
		for arg in "${threaded_programs[@]}"; do
			echo "Running threaded program: $arg"
			python3 run_benchmarks.py --iterative --threaded -f "$arg" --fast --name "$name_arg"
		done
	fi

	# Loop through the arguments and run the Python program
	for arg in "${arguments[@]}"; do
		echo "Running Python program with argument: $arg"
		python3 run_benchmarks.py --iterative -f "$arg" --fast --name "$name_arg"
	done
fi
