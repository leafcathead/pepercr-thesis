#!/bin/bash

# Define programs

arguments=("real/rsa" "real/prolog" "real/grep" "real/fem" "real/cacheprof" "shootout/binary-trees"  "spectral/sorting")
# "parallel/nbody"
threaded_programs=("parallel/parfib")



 threaded_flag= false

 while getopts ":t" opt; do
	 case $opt in
		 t)
			 threaded_flag=true;;
		\?)
		echo "Invalid option: -$OPTARG" >&2
		exit 1
		;;
	esac
done

# Threaded Stuff if threaded_flag TRUE

if [ "$threaded_flag" = true ]; then
	for arg in "${threaded_programs[@]}"; do
		echo "Running threaded program: $arg"
		python3 run_benchmarks.py --iterative  --threaded -f "$arg" --norm --fast
	done
fi

 # Loop through the arguments and run the Python program
 for arg in "${arguments[@]}"; do
	     echo "Running Python program with argument: $arg"
	         python3 run_benchmarks.py --iterative  -f "$arg" --norm --fast
 done

