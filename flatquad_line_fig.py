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
import tqdm

import pontryagin_utils
from fig_config import *
from misc import *
from flatquad_landing_experiment import base_algo_params, define_problem_params

# line fig plots.

sys_name = 'flatquad'

run_id = 'mo8ys11a'
run_id = 'uqf3ybp8'
run_id = '12lxmqhl'


def plot_lines_singlerun(run_id):

    fpath = os.path.join(data_dir, f'{sys_name}_{run_id}_controlcosts_lines.msgpack.gz')
    with gzip.open(fpath, 'rb') as f:
        bs = f.read()
    eval_outputs = flax.serialization.msgpack_restore(bs)
    eval_outputs = jtm(np.array, eval_outputs)  # np array -> jax array

    N_cases = eval_outputs['costs'].shape[0]

    fig = pl.figure('controlcosts_lines', figsize=(pagewidth, 0.6*pagewidth))

    for j in range(N_cases):

        pl.subplot(2, 2, j+1)
        data = jtm(itemgetter(j), eval_outputs)

        xs = np.linspace(0, 1, data['costs'].shape[0])

        pl.plot(xs, data['v_means'], c='C0', label='v mean')

        lower = data['v_means']-sigs*data['v_stds']
        upper = data['v_means']+sigs*data['v_stds']
        pl.fill_between(xs, lower, upper, color='C0', alpha=confidence_band_alpha, label=f'v {sigs}σ confidence')

        pl.plot(xs, data['costs'], color='C1', label='Incurred cost')

        # find sensible ylim.
        ys_sorted = np.concatenate([lower, upper, data['costs']]).sort()
        N = ys_sorted.shape[0]
        ymin = ys_sorted[int(N*0.02)]
        ymax = ys_sorted[int(N*0.98)]
        rel_margin = 0.15
        extent = ymax - ymin
        ymin = ymin - rel_margin * extent
        ymax = ymax + rel_margin * extent
        pl.ylim([ymin, ymax])

        pl.legend()


    fig.tight_layout()
    pl.savefig(f'./{fig_dir}/{sys_name}_controlcosts_lines_{run_id}.{fig_format}', bbox_inches='tight', dpi=dpi)

    if show:
        pl.show()


plot_lines_singlerun(run_id)
