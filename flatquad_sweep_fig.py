#!/usr/bin/env python
import diffrax
import flax
import jax
import jax.numpy as np
from jax import config

config.update("jax_enable_x64", True)

import gzip
import os
import pickle
import warnings
from functools import partial

import ipdb
import matplotlib
import matplotlib.pyplot as pl
import numpy as onp
import scipy
import subprocess
import tqdm
import wandb
import sys

import pontryagin_utils
from fig_config import *
from misc import *
from flatquad_landing_experiment import base_algo_params, define_problem_params

# line fig plots, but for an entire sweep.



def pull_runs(runs):

    # runs = whatever is returned by wandb api :)

    # 2. get the runs from euler if not present already.

    print('pulling output data from euler (only current sweep)...')

    run_data_cmd = ['rsync', '--dry-run']
    run_data_cmd = ['rsync']

    for r in runs:
        run_data_cmd.append(f'--include={r.id}')

    run_data_cmd.append("--include='*.msgpack.gz'")
    run_data_cmd.append("--exclude='*'")
    run_data_cmd.append('-av')
    run_data_cmd.append('--progress')
    run_data_cmd.append('dbalduin@euler.ethz.ch:/cluster/scratch/dbalduin/flatquad_runs/')
    run_data_cmd.append('./euler_runs/')

    # oup = subprocess.run(run_data_cmd)
    # no clue why this works but not the other one
    oup = subprocess.run(' '.join(run_data_cmd), shell=True)

    print('pulling eval/plot data from euler (all runs)...')
    # here we just get everything, much less data
    plot_data_cmd = [
        'rsync',
        '-av',
        '--progress',
        'dbalduin@euler.ethz.ch:/cluster/scratch/dbalduin/plot_data/',
        'plot_data/'
    ]

    oup = subprocess.run(' '.join(plot_data_cmd), shell=True)



def plot_sweep(sysname, sweep_name):

    # 1. use wandb api to get all runs matching that sweep name
    print('fetching runs from wandb...')
    api = wandb.Api()
    runs = api.runs(path=f'mbjd-projects/levelsets_{sysname}', filters={'config.sweep_name': sweep_name})

    # 2. pull the data from euler
    pull_runs(runs)




if __name__ == '__main__':

    sys_name = 'flatquad'
    plot_sweep(sys_name, 'batchsize')
