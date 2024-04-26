# basic tools for handling continuous time trajectories.
# mostly in the form of a diffrax sol object for a SINGLE trajectory. 
# use vmap if you want more. 

import jax
import jax.numpy as np
import matplotlib.pyplot as pl

import ipdb


def find_zero_on_trajectory(sol, fct, lower, upper, tol=1e-5, i_max=50):
    '''
    find the zero of some function along the trajectory. 
    
    sol: diffrax solution object. 
    fct: function from solution state to real numbers. 
    lower, upper: independent variables with fct(lower) fct(upper) < 0.

    atm it does basic old bisection. TODO combine w/ regula falsi for 
    probably better convergence in most cases? 
    '''

    ipdb.set_trace()
    f_lower = fct(sol.evaluate(lower)) 
    f_upper = fct(sol.evaluate(upper))

    assert f_lower * f_upper < 0, 'initial points for bisection have same sign'

    # atm just basic bisection. 
    def body_fun(val):

        # val = whatever we would otherwise keep in global state. 
        lower, f_lower, upper, f_upper, i = val

        mid = 0.5 * (lower + upper)
        f_mid = fct(sol.evaluate(mid))

        # if mid and upper have the same sign, mid is to the right of our zero. 
        mid_is_above = f_mid * f_upper > 0

        new_val = jax.lax.select(
            mid_is_above,
            (lower, f_lower, mid, f_mid, i+1),
            (mid, f_mid, upper, f_upper, i+1),
        )

        return new_val

    def cond_fun(val):

        lower, f_lower, upper, f_upper, i = val

        not_too_long = i_max > 50
        not_accurate_enough = np.minimum(np.abs(f_lower), np.abs(f_upper))

        # while not_too_long and not_accurate enough, do the loop.
        # this means if one of them changes (too long or accurate enough)
        # we stop. 
        return np.logical_or(not_too_long, not_accurate_enough)


    init_val = (lower, f_lower, upper, f_upper, 0)

    val = jax.lax.while_loop(cond_fun, body_fun, init_val)

    return val








def find_minimum_on_trajectory(sol, fct):

    '''
    plan: 
    - find minimum of f on saved points
    - find the zero of t -> d/dt fct(x(t))
      - the "find zero" function can do this :) 
      - with lower/upper given by adjacent saved points
      - differentiating is easy bc. the interpolation is just a piecewise polynomial. 
    - profit 
    '''
