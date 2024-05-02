#!/bin/bash

sbatch -n 1 --cpus-per-task=1 --time=12:00:00 --mem-per-cpu=16384 --output="./tmp/flatquad_output.txt" --error="./tmp/flatquad_err.txt" --wrap="./flatquad_landing_experiment.py"
