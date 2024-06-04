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

# controlcosts common.

fpath = os.path.join(data_dir, f'flatquad_{run_id}_controlcosts_common.msgpack.gz')
with gzip.open(fpath, 'rb') as f:
    bs = f.read()
eval_outputs = flax.serialization.msgpack_restore(bs)
eval_outputs = jtm(np.array, eval_outputs)  # np array -> jax array

pl.figure('controlcost vs v_mean')
ipdb.set_trace()
costs = eval_outputs['costs']
pl.loglog(eval_outputs['v_mean'], costs, '. ')

pl.figure('controlcost cdf')
pl.semilogx((costs/eval_outputs['v_mean']).sort(), np.linspace(0, 1, costs.shape[0]))
pl.show()

ipdb.set_trace()
