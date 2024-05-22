#!/usr/bin/env sh

# wandb run id here

SYS=$1
RUN=$2

rsync -av dbalduin@euler.ethz.ch:/cluster/scratch/dbalduin/${SYS}_runs/$RUN ./euler_runs

# alternative: ALL of them
