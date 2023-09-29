#!/bin/bash

# Define programs

#arguments=("real/rsa" "real/prolog" "real/linear" "real/grep" "real/fem" "real/compress" "real/cacheprof" "real/ben-raytrace" "shootout/binary-trees" "shootout/n-body" "imaginary/primes" "spectral/gcd" "spectral/sorting")
arguments=("real/rsa" "real/prolog")

# threaded_programs=("parallel/parfib" "parallel/ray" "parallel/queens" "parallel/nbody)
threaded_programs=("parallel/cfd")



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
		python3 run_benchmarks.py --iterative --threaded -f "$arg"
	done
fi

 # Loop through the arguments and run the Python program
 for arg in "${arguments[@]}"; do
	     echo "Running Python program with argument: $arg"
	         python3 run_benchmarks.py --iterative -f "$arg"
 done
