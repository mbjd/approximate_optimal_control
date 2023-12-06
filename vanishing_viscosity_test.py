#!/usr/bin/env python
import jax
import jax.numpy as np
import optax
from jax_tqdm import scan_tqdm

from nn_utils import nn_wrapper
import pontryagin_utils

import matplotlib.pyplot as pl
import numpy as onp
import scipy

import ipdb
from functools import partial

# define "orbits" problem

# example from idea dump. orbit-like thing where move in circles. if outside the
# unit circle, move in one direction, inwards other. input u moves "orbit radius".
# aim to stabilise (0, 1)

def f(t, x, u):
    rotspeed = (x[0]**2 + x[1]**2 - 1).reshape(u.shape)
    # mat = np.diag([u, u]) * u + np.array([[0, -1], [1, 0]]) * rotspeed

    mat = np.array([[u, -rotspeed], [rotspeed, u]]).squeeze()

    return mat @ x  # weird pseudo linear thing


def l(t, x, u):
    Q = np.eye(2)
    err = x - np.array([0, 1])
    distpenalty = err.T @ Q @ err
    rotspeed = x[0]**2 + x[1]**2 - 1
    vpenalty = (x[0]**2 + x[1]**2 - 1)**2
    inp_penalty = 10 * u**2
    return (vpenalty + 0.1 * distpenalty + inp_penalty).reshape()


# from http://www.mwm.im/lqr-controllers-with-python/
def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    # ref Bertsekas, p.151
    # first, try to solve the ricatti equation
    # somehow this stuff only works with old np.matrix types and * multiplication
    X = onp.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    # compute the LQR gain
    K = onp.matrix(scipy.linalg.inv(R) * (B.T * X))
    eigVals, eigVecs = scipy.linalg.eig(A - B * K)
    return K, X, eigVals



x_eq = np.array([0., 1.])

# linearise around equilibrium.
# terminal lqr controller for inf horizon approximation
# store these somewhere? or hope for jit to save us?
zero_u = np.zeros(1)
A = jax.jacobian(f, argnums=1)(0., x_eq, zero_u)
B = jax.jacobian(f, argnums=2)(0., x_eq, zero_u).reshape((2, 1))
Q = jax.hessian(l, argnums=1)(0., x_eq, zero_u)
R = jax.hessian(l, argnums=2)(0., x_eq, zero_u)

K0_inf, P0_inf, eigvals = lqr(A, B, Q, R)


def h(x):

    # terminal value.
    return (x.T @ P0_inf @ x).reshape()


problem_params = {
    'system_name': 'orbits',
    'f': f,
    'l': l,
    'h': h,
    'T': np.inf,
    'nx': 2,
    'nu': 1,
    'U_interval': [-.2, .2],
    'terminal_constraint': False,
    'V_max': 3.8,
}

algo_params = {
    'nn_layersizes': (32, 32, 32),
    'N_collocationpts': 1024,

    'lr_staircase': True,
    'lr_staircase_steps': 6,
    'lr_init': 0.0005,
    'lr_final': 0.0001,
}


key = jax.random.PRNGKey(0)


# @partial(jax.jit, static_argnums=(1,))
def stationary_hjb_loss(pt, V_nn, nn_params, eps, vmin, vmax):

    # pt: single point of shape (nx,) to evaluate loss
    # V_nn, nn_params: NN approx. solution
    # eps > 0: contribution of viscosity term
    # vmin > 0: value level below which we use the lqr solution as boundary condition.
    # vmax > vmin: value level above which the loss becomes 0.
    #       in the future replace this case distinction stuff by sampling points?

    # value and derivatives.
    # need only sum(trace(hessian)) -- is there a better way?
    nx = problem_params['nx']
    V_nn_val = V_nn(nn_params, pt)
    V_nn_x = jax.jacobian(V_nn, argnums=1)(nn_params, pt).reshape(nx)
    V_nn_xx = jax.hessian(V_nn, argnums=1)(nn_params, pt).reshape((nx, nx))

    offset = pt - x_eq
    V_lqr = offset.T @ P0_inf @ offset
    V_grad_lqr = 2 * P0_inf @ offset  # this is the gradient of the quad form right?

    loss_lqr = (V_nn_val - V_lqr)**2 + np.sum(np.square(V_nn_val - V_lqr))
    loss_lqr = loss_lqr.reshape()  # shape (1,) -> ()

    t = 0.  # remove this completely sometime?

    # stationary hjb eq:
    # ∀x: min_u [ l(x, u) + V_x f(x, u) ] = 0
    # V(x_eq) = 0, V_x(x_eq) = 0

    # viscosity approximation: ∀x: min_u [ l(x, u) + V_x f(x, u) ] = ε∆V(x)
    # (see e.g. https://web.math.ucsb.edu/~jhateley/paper/sHJe.pdf)

    u_star = pontryagin_utils.u_star_new(pt, V_nn_x, problem_params)
    hamiltonian = problem_params['l'](t, pt, u_star) + V_nn_x.T @ problem_params['f'](t, pt, u_star)
    viscosity_term = eps * np.sum(np.trace(V_nn_xx))

    # put asymmetric loss here to encourage V over/underapproximation
    loss_hjb = np.square(hamiltonian - viscosity_term)

    # maybe smooth version of this?
    # if V_lqr is below vmin, use lqr loss fct
    α = (V_lqr < vmin).astype(float)
    total_loss = (α) * loss_lqr + (1-α) * loss_hjb

    # just a small test
    # total_loss = loss_lqr

    # if V larger than max, ignore.
    return (total_loss * (V_nn_val < vmax)).reshape()




# grad of loss wrt nn params, for a batch of pts with shape (N_pts, nx)
def stationary_hjb_loss_batch(pts, V_nn, nn_params, eps, vmin, vmax):
    loss_vmap = jax.vmap(stationary_hjb_loss, in_axes=(0, None, None, None, None, None))
    losses = loss_vmap(pts, V_nn, nn_params, eps, vmin, vmax)
    return np.mean(losses, axis=0)

loss_val_grad = jax.value_and_grad(stationary_hjb_loss_batch, argnums=2)

# try to solve time independent (inf horizon) hjb equation :)
V_nn = nn_wrapper(
    input_dim=problem_params['nx'],
    layer_dims = algo_params['nn_layersizes'],
    output_dim=1
)

nn_key, key = jax.random.split(key, 2)
nn_params = V_nn.nn.init(nn_key, np.zeros((problem_params['nx'],)))

# initially, try fixed collcoation points uniformly distributed
# for 2d this should be ok. later try better things.
data_key, key = jax.random.split(key, 2)
pts = jax.random.uniform(
    data_key,
    shape=(algo_params['N_collocationpts'], problem_params['nx']),
    minval=-2, maxval=2,
)


# eps = .1
# print(stationary_hjb_loss(x_eq+np.array([.7, -.7]), V_nn, nn_params, eps, 0.2, 5))
# ipdb.set_trace()

# exponential decay. this will go down from lr_init to lr_final over
# the whole training duration.
# if lr_staircase, then instead of smooth decay, we have stepwise decay
# with
N_lr_steps = algo_params['lr_staircase_steps']

total_decay = algo_params['lr_final'] / algo_params['lr_init']

# not algo params :o
total_iters = 200000


lr_schedule = optax.exponential_decay(
        init_value = algo_params['lr_init'],
        transition_steps = total_iters // N_lr_steps,
        decay_rate = (total_decay) ** (1/N_lr_steps),
        end_value=algo_params['lr_final'],
        staircase=algo_params['lr_staircase']
)

optim = optax.adam(learning_rate=lr_schedule)
opt_state = optim.init(nn_params)


# test this shit

epss = np.ones(total_iters) * 10
# at the start, train with vmin=vmax to only fit terminal LQR value.
# then, start increasing vmax.
vboundary = np.linspace(0, 5, total_iters)
vmins = np.ones(total_iters) * 0.5
vmaxs = np.maximum(vmins, vboundary)

@scan_tqdm(total_iters, print_rate=10)
def step(carry, n):

    # unpack the 'carry' state
    nn_params, opt_state = carry

    # apparently jax-tqdm only works when step n is the input.
    # so we get our actual inputs like this:
    eps = epss[n]
    vmin = vmins[n]
    vmax = vmaxs[n]

    # new set of points every time
    pts = jax.random.uniform(
        jax.random.PRNGKey(n),
        shape=(64, problem_params['nx']),
        minval=-2, maxval=2,
    )

    # calculate loss gradient :)
    loss, grad = loss_val_grad(pts, V_nn, nn_params, eps, vmin, vmax)

    # do optimisation step
    updates, opt_state_new = optim.update(grad, opt_state)
    nn_params_new = optax.apply_updates(nn_params, updates)

    output = {
            'lr': lr_schedule(opt_state[0].count),
            'loss': loss,
            'params': nn_params_new,
    }

    new_carry = (nn_params_new, opt_state_new)

    return new_carry, output


init_carry = (nn_params, opt_state)

# the training loop!
final_carry, oups = jax.lax.scan(step, init_carry, np.arange(total_iters))

nn_params, _ = final_carry


# plot EVERYTHING

pl.figure('training pts')
pl.scatter(pts[:, 0], pts[:, 1])

pl.figure('domain scheduling')
pl.plot(epss, label='eps')
pl.plot(vmins, label='vmin')
pl.plot(vmaxs, label='vmax')
pl.legend()

pl.figure('training stuff')
pl.semilogy(oups['loss'], label='loss')
pl.semilogy(oups['lr'], label='lr')
pl.legend()

pl.figure('random selection of nn params')
pl.plot(oups['params']['params']['Dense_2']['kernel'].reshape(1000, -1), color='black', alpha=.05)


# plot loss on test points.
data_key, key = jax.random.split(key, 2)
testpts = jax.random.uniform(
    data_key,
    shape=(128, problem_params['nx']),
    minval=-2, maxval=2,
)

# def stationary_hjb_loss(pt, V_nn, nn_params, eps, vmin, vmax):

# batch testpts

# insert V_nn fixed
loss_Vnn_partial = lambda pt, nn_params, eps, vmin, vmax: stationary_hjb_loss(pt, V_nn, nn_params, eps, vmin, vmax)
value_Vnn_partial = lambda pt, nn_params: stationary_hjb_loss(pt, V_nn, nn_params, .0, .0, 1000.)

losses_batch = jax.vmap(loss_Vnn_partial, in_axes=(0, None, None, None, None))
value_batch = jax.vmap(value_Vnn_partial, in_axes=(0, None))
# vmap over everything concerned with the (optimisaion steps) time axis
losses_batch_params = jax.vmap(losses_batch, in_axes=(None, 0, 0, 0, 0))
value_batch_params = jax.vmap(value_batch, in_axes=(None, 0))

# all_losses = losses_batch_params(testpts, oups['params'], epss, vmins, vmaxs)
# all_vs = value_batch_params(testpts, oups['params'])
#
# pl.figure('losses')
# pl.plot(all_losses, color='black', alpha=.1)
#
# pl.figure('values')
# pl.plot(all_vs, color='black', alpha=.1)
pl.show()

ipdb.set_trace()
