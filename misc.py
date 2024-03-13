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


# various small utility functions.

def rnd(a, b):
    # relative norm difference. useful for checking if matrices or vectors are close
    return np.linalg.norm(a - b) / np.maximum(np.linalg.norm(a), np.linalg.norm(b))

def count_floats(pytree):
    # counts total number of elements of a pytree with jax array nodes.
    node_sizes = jax.tree_util.tree_map(lambda n: n.size, pytree)
    return jax.tree_util.tree_reduce(operator.add, node_sizes)
