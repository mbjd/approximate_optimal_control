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



def find_max_l_bisection(sols, v_k, problem_params):

	# this function is unnecessary (aimed towards a misguided goal). but it has 
	# a working implementation of bisection across time axis. come here if we need 
	# this again anytime. 

	def state_at_vk_bisection(sol):

		# find the state at which v = v_k by bisection on the time axis.
		# everything using the interpolated solution. 

		# how many do we need? log2(time interval / final t tolerance)
		# this here = log2(5 / 5e-6)
		iters = 20

		# initially we assume left > v_k, right < v_k 
		# (v monotonously decreasing.)

		def f_scan(time_interval, input):

			# time_interval needs to be np.array of shape (2,) for jax.lax.select to work.
			left, right = time_interval

			# does it work even if we take another convex combination here? 
			# to account for "skewedness" of the v(t) function? 
			# probably not worth the marginal gains. regula falsi exists too 
			# but plain bisection is good enough.
			mid = time_interval.mean()

			vmid = sol.evaluate(mid)['v']

			# if vmid is lower, mid becomes right
			# if higher, mid becomes left. 
			next_time_interval = jax.lax.select(
				vmid < v_k,
				np.array([left, mid]),  # on_true
				np.array([mid, right]),  # on_false
			)

			return next_time_interval, vmid
		
		init_time_interval = np.array([sol.t1, sol.t0])

		# if v_k is not in this interval, the bisection result is meaningless.
		# maybe smarter to check in the end if vmid reaches our target?
		init_v_interval = jax.vmap(sol.evaluate)(init_time_interval)['v']
		result_usable = np.logical_and(init_v_interval[0] >= v_k, v_k >= init_v_interval[1])

		ts_final, vmids = jax.lax.scan(f_scan, init_time_interval, None, length=100)

		# if v_k not in interval replace the result by NaN :) 
		ts_final = ts_final + (np.nan * ~result_usable)
		state = sol.evaluate(ts_final.mean())

		return state

	# TODO consider case where the solution does not intersect the value level. 
	
	# sol0 = jtm(itemgetter(0), sols)
	# state = state_at_vk_bisection(sol0)
	# ipdb.set_trace()

	ys = jax.vmap(state_at_vk_bisection)(sols)

	def l_of_y(y):
		x = y['x']
		vx = y['vx']
		u = pontryagin_utils.u_star_2d(x, vx, problem_params)
		return problem_params['l'](x, u)

	all_ls = jax.vmap(l_of_y)(ys)

	ipdb.set_trace()
	max_l = np.nanmax(all_ls)	
	return max_l

