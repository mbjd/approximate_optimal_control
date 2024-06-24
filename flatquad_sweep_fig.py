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
import datetime
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


def pull_runs(sysname, sweep_name, T=np.inf):

    # sysname: the system name used in problem_params
    # sweep_name: the sweep name as given by algo_params['sweep_name']
    # T: max age of data we include, in hours.

    # runs = whatever is returned by wandb api :)
    # 1. use wandb api to get all runs matching that sweep name
    print(f'fetching runs for sweep {sweep_name} from wandb...')
    api = wandb.Api()
    runs = api.runs(path=f'mbjd-projects/levelsets_{sysname}', filters={'config.sweep_name': sweep_name})

    # remove too old runs. either I am too dumb or wandb documentation is
    # too crappy for me to find out how to do this with filters above.
    print(f'got {len(runs)} runs')
    cutoff_datetime = datetime.datetime.now() - datetime.timedelta(hours=T)
    def is_recent(r):
        run_datetime = datetime.datetime.strptime(r.createdAt, '%Y-%m-%dT%H:%M:%S')
        return run_datetime > cutoff_datetime

    runs = [r for r in runs if is_recent(r)]
    print(f'...{len(runs)} of which are recent enough')

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
    runs = pull_runs(sysname, sweep_name, T=72)

    # can we infer this from the runs object?
    # sweep_config = 'active_learning_batchsize'

    # nice name for plotting
    nice_sweep_config = {
        'active_learning_batchsize': 'Active learning batch size $N_\\text{batch}$',
        'weight_decay': 'Weight Decay',
        'nn_layer_dim': 'NN Layer size',
        'pontryagin_solver_rtol': 'ODE Solver rtol',
        'vx_loss_d': '$\lambda$ Huber width $\delta$',
        'dtmax': 'ODE solver $\Delta t_\\text{max}$',
        'inv_vx_loss_fadeout': 'Loss fadeout $\mu$',
        'lr_final': '$\\text{lr}_\\text{final}$',
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
    fracs_TO = dict()


    # read refsol costs only once.
    fpath = os.path.join(data_dir, f'{sys_name}_refsol_costs.msgpack.gz')
    with gzip.open(fpath, 'rb') as f:
        bs = f.read()
    refsol_outputs = flax.serialization.msgpack_restore(bs)
    refsol_outputs = jtm(np.array, refsol_outputs)  # np array -> jax array


    for r in runs:
        try:
            last = r.history(pandas=False)[-1]
        except IndexError:
            print('run has empty history')
            continue

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
            # data not present in wandb :(((( can we get it otherwise?
            print('key not found.')


        # also plot a bit of the other stats?
        run_id = r.id
        fpath = os.path.join(data_dir, f'{sys_name}_{run_id}_controlcosts_lines.msgpack.gz')
        with gzip.open(fpath, 'rb') as f:
            bs = f.read()
        eval_outputs_lines = flax.serialization.msgpack_restore(bs)
        eval_outputs_lines = jtm(np.array, eval_outputs_lines)  # np array -> jax array

        fpath = os.path.join(data_dir, f'{sys_name}_{run_id}_controlcosts_common.msgpack.gz')
        with gzip.open(fpath, 'rb') as f:
            bs = f.read()
        eval_outputs_common = flax.serialization.msgpack_restore(bs)
        eval_outputs_common = jtm(np.array, eval_outputs_common)  # np array -> jax array

        # calculate control cost wrt TO refsol.

        # what to plot? cdf? some percentiles as above?
        all_TO_refsols = np.concatenate([np.minimum(n['left'], n['right']) for n in refsol_outputs])
        all_closedloop_costs = np.concatenate([oup['costs'] for oup in eval_outputs_lines])

        ipdb.set_trace()

        if relevant_config not in fracs_TO:
            fracs[relevant_config] = []
        fracs_TO[relevant_config].append(None) # todo

    # ipdb.set_trace()


    # can we pull this out into a function to do it for other metrics as
    # well???

    def dict_to_arrays(data_dict):

        # converts a dict {x: [v0, v1, ...]} (with x representing a
        # particular value of the swept variable) to arrays:
        # xs = [x0, x1, ...]
        # vmins = [v0min, v1min, ...]
        # vmeans = [v0mean, v1mean, ...]
        # vmaxs = [v0max, v1max, ...]

        arraydict = {k: np.array(v) for k, v in data_dict.items() if len(v) > 0}

        min_array =  np.array([np.min(v, axis=0) for k, v in sorted(arraydict.items())])
        mean_array =  np.array([np.mean(v, axis=0) for k, v in sorted(arraydict.items())])
        max_array =  np.array([np.max(v, axis=0) for k, v in sorted(arraydict.items())])

        xs = np.array(sorted(arraydict.keys()))

        return xs, min_array, mean_array, max_array

    xs, ys_min, ys_mean, ys_max = dict_to_arrays(fracs)


    fig = pl.figure('sweepfig', figsize=(pagewidth, .6*pagewidth))

    pl.semilogx(xs, ys_mean, label=('Fraction below 5% suboptimality', 'Fraction below 50% suboptimality', 'Fraction below 500% suboptimality'))

    # ipdb.set_trace()
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
    # most sensibly we would probably plot the following:
    #  - control cost on the six sweeps (compared to TO cost and/or by #  itself)
    #    (as cdf of the suboptimality ratio?)
    #  - mean control cost on eval_xs (which are not the same for each run,
    #  not even from the same same distribution bc level sets differ a bit)
    #  - stats like number of steps, runtime, floating point ops even?

    fig.tight_layout()
    pl.savefig(f'./{fig_dir}/{sys_name}_sweep_{sweep_name}.{fig_format}', bbox_inches='tight', dpi=dpi)

    if show:
        pl.show()








if __name__ == '__main__':

    sys_name = 'flatquad'
    plot_sweep(sys_name, 'vx_fadeout', 'inv_vx_loss_fadeout')
    plot_sweep(sys_name, 'lr_final', 'lr_final')
    #plot_sweep(sys_name, 'dtmax', 'dtmax')
    plot_sweep(sys_name, 'vxd', 'vx_loss_d')
    plot_sweep(sys_name, 'batchsize', 'active_learning_batchsize')
    plot_sweep(sys_name, 'weight_decay', 'weight_decay')
    # completely uninteresting sadly
    # plot_sweep(sys_name, 'rtol', 'pontryagin_solver_rtol')
    # plot_sweep(sys_name, 'layerdim', 'nn_layer_dim')
