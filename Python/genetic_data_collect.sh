#!/bin/bash

# Define programs

arguments=("real/rsa" "real/prolog"  "real/grep" "real/fem"  "real/cacheprof"  "shootout/binary-trees"  "spectral/sorting")

threaded_programs=("parallel/parfib" "parallel/nbody")



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
		python3 run_benchmarks.py --genetic --norm --threaded -f "$arg"
	done
fi

 # Loop through the arguments and run the Python program
 for arg in "${arguments[@]}"; do
	     echo "Running Python program with argument: $arg"
	         python3 run_benchmarks.py --genetic --norm -f "$arg"
 done

