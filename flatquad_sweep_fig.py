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


def pull_runs(sysname, sweep_name):

    # runs = whatever is returned by wandb api :)
    # 1. use wandb api to get all runs matching that sweep name
    print(f'fetching runs for sweep {sweep_name} from wandb...')
    api = wandb.Api()
    runs = api.runs(path=f'mbjd-projects/levelsets_{sysname}', filters={'config.sweep_name': sweep_name})

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

    return runs



def plot_sweep(sysname, sweep_name, sweep_config):

    # easiest to just pass both sweep name (value of the dummy algoparam arg,
    # used for getting the correct runs) AND the sweep config (which config
    # variable is modified in the sweep), then we can name them idependently.

    # 2. pull the data from euler
    runs = pull_runs(sysname, sweep_name)

    # can we infer this from the runs object?
    # sweep_config = 'active_learning_batchsize'

    # nice name for plotting
    nice_sweep_config = {
        'active_learning_batchsize': 'Active learning batch size $N_\\text{batch}$',
        'weight_decay': 'Weight Decay',
        'nn_layer_dim': 'NN Layer size',
    }[sweep_config]


    batchsizes = [r.config[sweep_config] for r in runs]
    batchsizes_unique = sorted([j for j in set(batchsizes)])

    labels=[
        '$\gamma_1(s) = [-10 + 20 s, 0, 0, 1, 0, 0, 0]$',
        '$\gamma_2(s) = [-10 + 20 s, 0, 0, -1, 0, 5, 0]$',
        '$\gamma_3(s) = [-10 + 20 s, 0, 0, -1, 0, 10, 0]$',
        '$\gamma_4(s) = [-5, 5 s, 0, -1, 5 s, 5, 0]$',
        '$\gamma_5(s) = [0, 0, \sin(2 \pi s), \cos(2 \pi s), 0, 5, 0]$',
        '$\gamma_6(s) = [-5, 0, \sin(2 \pi s), \cos(2 \pi s), 5, 5, 0]$',
    ]


    # put the closed loop / learned percentiles in a dict with key
    # being the swept config.
    fracs = dict()
    for r in runs:
        last = r.history(pandas=False)[-1]
        relevant_config = r.config[sweep_config]
        if relevant_config not in fracs:
            fracs[relevant_config] = []

        try:
            fracs[relevant_config].append((
                last['frac_ratio_005'],
                last['frac_ratio_050'],
                last['frac_ratio_500'],
            ))
        except KeyError:
            # just ignore the cases where it didn't finish...
            print('key not found.')

    # convert to mean with min/max range for plotting.
    # certainly there should be some MUCH nicer way of doing this.....
    arraydict = {k: np.array(v) for k, v in fracs.items() if len(v) > 0}

    min_dict = {k: np.min(v, axis=0) for k, v in arraydict.items()}
    mean_dict = {k: np.mean(v, axis=0) for k, v in arraydict.items()}
    max_dict = {k: np.max(v, axis=0) for k, v in arraydict.items()}

    xs = sorted(arraydict.keys())

    # convert to np array, shaped (N_runs, 3).
    ys_min = np.array([z[1] for z in sorted(min_dict.items())])
    ys_mean = np.array([z[1] for z in sorted(mean_dict.items())])
    ys_max = np.array([z[1] for z in sorted(max_dict.items())])




    fig = pl.figure('sweepfig', figsize=(.6*pagewidth, .5*pagewidth))

    pl.semilogx(xs, ys_mean, label=('Fraction below 5% suboptimality', 'Fraction below 50% suboptimality', 'Fraction below 500% suboptimality'))

    pl.gca().set_prop_cycle(None)
    # fill_between wants to be done individually...
    for j in range(3):
        pl.fill_between(xs, ys_min[:, j], ys_max[:, j], alpha=confidence_band_alpha)

    pl.legend()

    pl.xlabel(nice_sweep_config)
    pl.ylabel('Fraction')
    pl.ylim([0, 1])
    pl.grid('on')

    # second subplot with other stats like N iterations etc?

    fig.tight_layout()
    pl.savefig(f'./{fig_dir}/{sys_name}_sweep_{sweep_name}.{fig_format}', bbox_inches='tight', dpi=dpi)

    if show:
        pl.show()






if __name__ == '__main__':

    sys_name = 'flatquad'
    plot_sweep(sys_name, 'batchsize', 'active_learning_batchsize')
    plot_sweep(sys_name, 'weight_decay', 'weight_decay')
    # plot_sweep(sys_name, 'layerdim', 'nn_layer_dim')
