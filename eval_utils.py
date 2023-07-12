import jax
import jax.numpy as np

import diffrax

import pontryagin_utils

import ipdb


def closed_loop_eval_nn_ensemble(problem_params, algo_params, V_nn, nn_params, x0s):

    def u_star_fct(x):

        # get optimal control input with NN ensemble.
        nx = problem_params['nx']
        assert x.shape == (nx,)

        # here we vmap the apply function along the 0 axis of nn_params, and
        # at the same time kind of un-vmap it along the xs axis, as we have
        # just one x here.

        xs = x[None, :]
        costate_estimates = jax.vmap(V_nn.apply_grad, in_axes=(0, None))(nn_params, xs)
        costate_estimates = costate_estimates.reshape(algo_params['nn_ensemble_size'], nx)
        costate = costate_estimates.mean(axis=0)

        # what kind of idiot wrote this function
        u_star = pontryagin_utils.u_star_costate(
                problem_params['f'],
                problem_params['l'],
                costate,
                0.,  # t = 0
                x,
                problem_params['nx'],
                problem_params['nu'],
                problem_params['U_interval'],
        )

        return u_star

    return closed_loop_eval_general(problem_params, algo_params, u_star_fct, x0s)


def closed_loop_eval_general(problem_params, algo_params, ustar_fct, x0s):

    # make closed loop simulations, and record the control cost.
    # ustar_fct should be a jax function, X -> U. no time dependence.

    T = algo_params['sim_T']
    dt = algo_params['sim_dt']

    f = problem_params['f']
    l = problem_params['l']
    nx = problem_params['nx']

    @jax.jit
    def dynamics_extended(t, z, args=None):
        # one extra state to record control cost.
        # optimal input is applied here according to the model.

        x, cost = np.split(z, [nx])

        ustar = ustar_fct(x)

        xdot = f(t, x, ustar)
        cost_dot = l(t, x, ustar)

        zdot = np.concatenate([xdot, cost_dot.reshape(1,)])
        return zdot


    def solve_single(x0):
        term = diffrax.ODETerm(dynamics_extended)
        solver = diffrax.Tsit5()  # recommended over usual RK4/5 in docs

        max_steps = int(T / dt)
        saveat = diffrax.SaveAt(steps=True)

        # initialise 0 control cost
        z0 = np.concatenate([x0, np.zeros(1)])

        solution = diffrax.diffeqsolve(
                term, solver, t0=0, t1=T, dt0=dt, y0=z0,
                saveat=saveat, max_steps=max_steps,
        )

        # this should return the last calculated (= non-inf) solution.
        return solution

    # sol = solve_single(np.array([0, 2]))

    solve_multiple = jax.vmap(solve_single)
    all_sols = solve_multiple(x0s)

    return all_sols

def compute_controlcost(problem_params, all_sols):
    # all_sols exactly as returned by the closed_loop_eval_general.

    control_costs = all_sols.ys[:, -1, problem_params['nx']]
    mean_cost = control_costs.mean()

    return mean_cost
