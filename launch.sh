#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --job-name="flatquad_test"
#SBATCH --mem-per-cpu=1024
#SBATCH --output="./flatquad_out.txt"
#SBATCH --error="./flatquad_err.txt"
#SBATCH --open-mode=append

./flatquad_landing_experiment.py
