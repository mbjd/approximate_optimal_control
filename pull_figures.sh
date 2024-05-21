#!/usr/bin/env sh

# wandb run id here
RUN=$1

rsync -av dbalduin@euler.ethz.ch:/cluster/scratch/dbalduin/figures_$RUN .
rsync -av dbalduin@euler.ethz.ch:/cluster/home/dbalduin/approximate_optimal_control/tmp /local/home/dbalduin/approximate_optimal_control/figures_$RUN
