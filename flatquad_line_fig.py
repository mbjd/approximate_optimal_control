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
run_id = 'bmrmmxzq'


def plot_lines_singlerun(run_id):

    fpath = os.path.join(data_dir, f'{sys_name}_{run_id}_controlcosts_lines.msgpack.gz')
    with gzip.open(fpath, 'rb') as f:
        bs = f.read()
    eval_outputs = flax.serialization.msgpack_restore(bs)
    eval_outputs = jtm(np.array, eval_outputs)  # np array -> jax array

    # N_cases = eval_outputs['costs'].shape[0]
    N_cases = len(eval_outputs)  # now it is a list of dicts

    fig = pl.figure('controlcosts_lines', figsize=(pagewidth, .8*pagewidth))
    # manually transcribed curves from levelsets.py / evaluate_directly / eval_controlcost_lines
    labels=[
            '$\gamma_1(s) = [-10 + 20 s, 0, 0, 1, 0, 0, 0]$',
            '$\gamma_2(s) = [-10 + 20 s, 0, 0, -1, 0, 5, 0]$',
            '$\gamma_3(s) = [-10 + 20 s, 0, 0, -1, 0, 10, 0]$',
            '$\gamma_4(s) = [-5, 5 s, 0, -1, 5 s, 5, 0]$',
            '$\gamma_5(s) = [0, 0, \sin(2 \pi s), \cos(2 \pi s), 0, 5, 0]$',
            '$\gamma_6(s) = [-5, 0, \sin(2 \pi s), \cos(2 \pi s), 5, 5, 0]$',
            ]

    for j in range(N_cases):

        ax = pl.subplot(3, 2, j+1)
        ax.text(0.97, 0.03, labels[j],
                verticalalignment='bottom',
                horizontalalignment='right',
                transform=ax.transAxes)

        # data = jtm(itemgetter(j), eval_outputs)
        data = eval_outputs[j]

        xs = np.linspace(0, 1, data['costs'].shape[0])

        pl.plot(xs, data['v_means'], c='C0', label='Mean value $\mu_{\\boldsymbol{\Theta}}$')

        lower = data['v_means']-sigs*data['v_stds']
        upper = data['v_means']+sigs*data['v_stds']

        pl.fill_between(xs, lower, upper, color='C0', alpha=confidence_band_alpha, label='Confidence $\mu_{\\boldsymbol{\Theta}} \pm ' + str(sigs) + '\sigma_{\\boldsymbol{\Theta}}$')

        pl.plot(xs, data['costs'], color='C1', label='Closed loop cost')

        # find sensible ylim.
        ys_sorted = np.concatenate([lower, upper, data['costs']]).sort()
        N = ys_sorted.shape[0]
        ymin = ys_sorted[int(N*0.01)]
        ymax = ys_sorted[int(N*0.99)]
        rel_margin = 0.15
        extent = ymax - ymin
        ymin = ymin - 1.5*rel_margin * extent  # tiny bit more for the text
        ymax = ymax + rel_margin * extent
        pl.ylim([ymin, ymax])
        pl.grid('on')

        # looks much nicer with only 1 legend but is that smart?
        minimal = True
        if minimal:
            # only put legends & axis labels where they look nice.
            if j==0:
                pl.legend()
            if j in (0, 2, 4):
                pl.ylabel('Control cost')
            if j in (4, 5):
                pl.xlabel('s')

        else:
            # always everything.
            pl.legend()
            pl.xlabel('s')
            pl.ylabel('Control cost')


    fig.tight_layout()
    pl.savefig(f'./{fig_dir}/{sys_name}_controlcosts_lines_{run_id}.{fig_format}', bbox_inches='tight', dpi=dpi)

    if show:
        pl.show()


plot_lines_singlerun(run_id)
