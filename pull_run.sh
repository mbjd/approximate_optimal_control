#!/usr/bin/env sh

# wandb run id here

SYS=$1
RUN=$2

if [ -n "$RUN" ]; then
    rsync -av dbalduin@euler.ethz.ch:/cluster/scratch/dbalduin/${SYS}_runs/$RUN ./euler_runs
else
    echo got no run. pls give second argument
    exit 1
fi

# also copy the corresponding plot_data.
# globs on remmote "just" work when escaping!!! too cool
scp dbalduin@euler.ethz.ch:/cluster/scratch/dbalduin/plot_data/${SYS}_${RUN}_\*.msgpack.gz ./plot_data/

# also make some script to pull ALL of the plot data?
