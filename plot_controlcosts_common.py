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

# make flatquad plots.
# step 1: make data with levelsets.evaluate. write all sorts of control
# cost evaluations, plots of specific regions, everything important.
# step 2: read data here, make plots. do as little extra computation as
# possible.

run_id = 'mo8ys11a'
run_id = 'uqf3ybp8'
run_id = '12lxmqhl'
run_id = 'h6ysrbmi'
show=True

configs = [
        # ('flatquad', 'h6ysrbmi'),
        ('flatquad', 'bmrmmxzq'),
        # ('orbits', 'i2tcnb3h'),
]

# controlcosts common

def plot_controlcosts_common(sysname, run_id):


    fpath = os.path.join(data_dir, f'{sysname}_{run_id}_controlcosts_common.msgpack.gz')
    with gzip.open(fpath, 'rb') as f:
        bs = f.read()
    eval_outputs = flax.serialization.msgpack_restore(bs)
    eval_outputs = jtm(np.array, eval_outputs)  # np array -> jax array

    fig = pl.figure('controlcost vs v_mean', figsize=(pagewidth, 0.4*pagewidth))
    pl.subplot(121)
    costs = eval_outputs['costs']
    pl.loglog(eval_outputs['v_mean'], costs/eval_outputs['v_mean'], '. ', alpha=scatter_alpha)
    # TODO unify w report notation...
    # also 'cost' and 'value' kind of clash. use only one term?
    pl.grid('on')
    pl.xlabel('Mean value $\mu_{\\boldsymbol{\Theta}}$')
    pl.ylabel('$V^\\text{cl}_{\\boldsymbol{\Theta}} / \mu_{\\boldsymbol{\Theta}}(x)$')

    pl.subplot(122)
    pl.semilogx((costs/eval_outputs['v_mean']).sort(), np.linspace(0, 1, costs.shape[0]))
    pl.grid('on')
    pl.xlabel('r')
    pl.ylabel('$P \left(V^\\text{cl}_{\\boldsymbol{\Theta}} / \mu_{\\boldsymbol{\Theta}}(x) \leq r \\right)$')

    fig.tight_layout()
    pl.savefig(f'./{fig_dir}/{sysname}_costscatter_{run_id}.{fig_format}', bbox_inches='tight', dpi=dpi)

    # fig = pl.figure('controlcost cdf', figsize=(pagewidth, 0.4*pagewidth))
    # pl.semilogx((costs/eval_outputs['v_mean']).sort(), np.linspace(0, 1, costs.shape[0]))
    # pl.grid('on')
    # pl.xlabel('r')
    # pl.ylabel('P(incurred cost / estimated value $\leq$ r)')

    # fig.tight_layout()
    # pl.savefig(f'./{fig_dir}/{sysname}_costcdf_{run_id}.{fig_format}', bbox_inches='tight', dpi=dpi)

    # make same plot but with respect to V_ref, and only on states γ(.)
    # read lines eval results
    fpath = os.path.join(data_dir, f'{sysname}_{run_id}_controlcosts_lines.msgpack.gz')
    with gzip.open(fpath, 'rb') as f:
        bs = f.read()
    eval_outputs_lines = flax.serialization.msgpack_restore(bs)
    eval_outputs_lines = jtm(np.array, eval_outputs_lines)  # np array -> jax array

    # read refsol outputs.
    fpath = os.path.join(data_dir, f'{sysname}_refsol_costs.msgpack.gz')
    with gzip.open(fpath, 'rb') as f:
        bs = f.read()
    refsol_outputs = flax.serialization.msgpack_restore(bs)
    refsol_outputs = jtm(np.array, refsol_outputs)  # np array -> jax array

    # ipdb.set_trace()

    # left_refcosts = refsol_outputs[j]['left']
    # right_refcosts = refsol_outputs[j]['right']
    # optimal_refsol = np.minimum(left_refcosts, right_refcosts)

    Vcl = [n['costs'] for n in eval_outputs_lines]
    Vmean = [n['v_means'] for n in eval_outputs_lines]
    Vref = [np.minimum(n['left'], n['right']) for n in refsol_outputs]

    Vcl = np.concatenate(Vcl)
    Vref = np.concatenate(Vref)
    # ipdb.set_trace()

    fig = pl.figure('controlcost vs v_ref', figsize=(pagewidth, 0.4*pagewidth))
    pl.subplot(121)
    costs = Vcl
    pl.loglog(Vref, Vcl/Vref, '. ', alpha=scatter_alpha)
    # TODO unify w report notation...
    # also 'cost' and 'value' kind of clash. use only one term?
    pl.grid('on')
    pl.xlabel('Reference value $V_\\text{ref}(x)$')
    pl.ylabel('$V^\\text{cl}_{\\boldsymbol{\Theta}} / V_\\text{ref}(x)$')

    pl.subplot(122)
    pl.plot((Vcl/Vref).sort(), np.linspace(0, 1, Vref.shape[0]))
    pl.grid('on')
    pl.xlim((.9, 2))
    pl.xlabel('r')
    pl.ylabel('$P \left(V^\\text{cl}_{\\boldsymbol{\Theta}} / V_\\text{ref}(x) \leq r \\right)$')

    fig.tight_layout()
    pl.savefig(f'./{fig_dir}/{sysname}_costscatter_Vref_{run_id}.{fig_format}', bbox_inches='tight', dpi=dpi)


    if show:
        pl.show()


for c in configs:
    plot_controlcosts_common(*c)
