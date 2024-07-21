#!/bin/bash

# Directory where the scripts are located
# Replace "/path/to/scripts" with the path to your scripts if they are not in the current directory
cd /path/to/scripts

# Loop over each script matching the pattern 'run_*'
for script in run_*; do
    # Submit the script to Slurm
    sbatch $script
done
