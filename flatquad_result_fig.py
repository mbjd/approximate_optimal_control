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
from flatquad_experiment import base_algo_params, define_problem_params

# make flatquad plots.
# step 1: make data with levelsets.evaluate. write all sorts of control
# cost evaluations, plots of specific regions, everything important.
# step 2: read data here, make plots. do as little extra computation as
# possible.

run_id = '133742069'

# TODO the rest
