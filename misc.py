import jax
import jax.numpy as np
import numpy as onp
import diffrax

import nn_utils
import plotting_utils
import pontryagin_utils
import ddp_optimizer
import visualiser
import ct_basics

import matplotlib
import matplotlib.pyplot as pl
import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf

import ipdb
import time
import tqdm
import operator

from jax.tree_util import tree_map as jtm

# various small utility functions.

def rnd(a, b):
    # relative norm difference. useful for checking if matrices or vectors are close
    return np.linalg.norm(a - b) / np.maximum(np.linalg.norm(a), np.linalg.norm(b))

def count_floats(pytree):
    # counts total number of elements of a pytree with jax array nodes.
    float_dtype = np.zeros(1).dtype  # will be float32 or float64 depending on settings.
    node_sizes = jax.tree_util.tree_map(lambda n: n.size if n.dtype == float_dtype else 0, pytree)
    return jax.tree_util.tree_reduce(operator.add, node_sizes)

