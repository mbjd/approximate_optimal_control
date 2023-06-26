import jax
import jax.numpy as np
import ipdb

import diffrax

# we want to find the value function V.
# for that we want to approximately satisfy the hjb equation:
#
#    0 = V_t(t, x) + inf_u { l(t, x, u) + V_x(t, x).T @ f(t, x, u) }
#    V(x, T) = h(x)
#
# for this first case let us consider
#    f(t, x, u) = f_tx(t, x) + g(t, x) @ u
#    l(t, x, u) = l_tx(t, x) + u.T @ R @ u.
# so a control affine system with cost quadratic in u. this will make
# the inner minimization simpler:
#
#       argmin_u l(t, x, u) + V_x(t, x).T @ f(t, x, u)
#     = argmin_u l_x(x) + u.T @ R @ u + V_x(t, x).T @ (f_tx(t, x) + g(t, x) @ u)
#     = argmin_u          u.T @ R @ u + V_x(t, x).T @ (             g(t, x) @ u)
#
# and, with A = V_x(t, x).T @ g(t, x):
#
#     = argmin_u u.T @ R @ u + A @ u
#
# which is an unconstrained convex QP, solved by setting the gradient to zero.
#
#     = u s.t. 0 = (R + R.T) u + A
#     = solution of linear system (R + R.T, -A)
#
# this is implemented in the following - the pointwise minimization over u of the hamiltonian.
# we have a fake version of function overloading -- these do the same thing but with different inputs.
# mostly the same though, only cosmetic difference, in the end they all reduce down to the first one.

def u_star_matrices(R, A):
    # A is a row vector here...
    u_star_unconstrained = np.linalg.solve(R + R.T, -A.T)
    # return u_star_unconstrained
    return np.clip(u_star_unconstrained, -1, 1)

def u_star_functions(f, l, V, t, x, nx, nu):
    # assuming that l is actually of the form l(t, x, u) = l_tx(t, x) + u.T @ R @ u,
    # the hessian R is independent of u. R should be of shape (nu, nu).
    zero_u = np.zeros((1, 1))
    R = jax.hessian(l, argnums=2)(t, x, zero_u).reshape(1, 1)

    grad_V_x = jax.jacobian(V, argnums=1)(t, x).reshape(1, nx)
    grad_f_u = jax.jacobian(f, argnums=2)(t, x, zero_u).reshape(nx, nu)
    A = grad_V_x @ grad_f_u        # should have shape (1, nu)

    return u_star_matrices(R, A)

def u_star_costate(f, l, costate, t, x, nx, nu):
    zero_u = np.zeros(1)  # u is a rank-1 array!
    R = jax.hessian(l, argnums=2)(t, x, zero_u).reshape(1, 1)

    # costate = grad_V_x.T (costate colvec, grad_V_x col vec)
    # grad_V_x = jax.jacobian(V, argnums=1)(t, x).reshape(1, nx)
    grad_f_u = jax.jacobian(f, argnums=2)(t, x, zero_u).reshape(nx, nu)
    A = costate.T @ grad_f_u        # should have shape (1, nu)

    return u_star_matrices(R, A)



def define_extended_dynamics(problem_params):

    # the dynamics governing state, costate and value evolution according to
    # pontryagin minimum principle. normal in forward time.
    # return as a function taking only (t, y) arguments.
    # suitable for diffrax ODETerm.

    f  = problem_params['f' ]
    l  = problem_params['l' ]
    h  = problem_params['h' ]
    T  = problem_params['T' ]
    nx = problem_params['nx']
    nu = problem_params['nu']

    def f_forward(t, y, args=None):

        # unpack. english names to distinguish from function arguments...
        state   = y[0:nx]
        costate = y[nx:2*nx]
        value   = y[2*nx]

        # define ze hamiltonian for that time.
        H = lambda x, u, λ: l(t, x, u) + λ.T @ f(t, x, u)

        u_star = u_star_costate(f, l, costate, t, state, nx, nu)

        # the first line is just a restatement of the dynamics
        # but doesn't it look cool with those partial derivatives??
        # the jacobian has shape (1, 1, nx, 1)? wtf?
        state_dot   =  jax.jacobian(H, argnums=2)(state, u_star, costate).reshape(nx)
        costate_dot = -jax.jacobian(H, argnums=0)(state, u_star, costate).reshape(nx)
        value_dot   = -l(t, state, u_star).reshape(1)

        y_dot = np.concatenate([state_dot, costate_dot, value_dot])
        return y_dot

    return f_forward



def make_pontryagin_solver(problem_params, algo_params):

    f_forward = define_extended_dynamics(problem_params)

    # solve pontryagin backwards, for vampping later.
    # slightly differently parameterised than in other version.
    def pontryagin_backward_solver(y0, t0, t1):

        # setup ODE solver
        term = diffrax.ODETerm(f_forward)
        solver = diffrax.Tsit5()  # recommended over usual RK4/5 in docs

        # negative if t1 < t0, backward integration just works
        assert algo_params['pontryagin_sampler_dt'] > 0
        dt = algo_params['pontryagin_sampler_dt'] * np.sign(t1 - t0)

        # what if we accept that we could create NaNs?
        max_steps = int(1 + problem_params['T'] / algo_params['pontryagin_sampler_dt'])

        # maybe easier to control the timing intervals like this?
        saveat = diffrax.SaveAt(steps=True)

        # and solve :)
        solution = diffrax.diffeqsolve(
                term, solver, t0=t0, t1=t1, dt0=dt, y0=y0,
                saveat=saveat, max_steps=max_steps,
        )

        # this should return the last calculated (= non-inf) solution.
        return solution, solution.ys[solution.stats['num_accepted_steps']-1]

    # vmap = gangster!
    # vmap only across first argument.
    batch_pontryagin_backward_solver = jax.jit(jax.vmap(
        pontryagin_backward_solver, in_axes=(0, None, None)
    ))

    return batch_pontryagin_backward_solver


def make_pontryagin_solver_wrapped(problem_params, algo_params):

    # construct the terminal extended state

    # helper function, expands the state vector x to the extended state vector y = [x, λ, v]
    # λ is the costate in the pontryagin minimum principle
    # h is the terminal value function
    def x_to_y_terminalcost(x, h):
        costate = jax.grad(h)(x)
        v = h(x)
        ipdb.set_trace()
        y = np.concatenate([x, costate, v])

        return y

    # slight abuse here: the same argument is used for λ as for x, be aware
    # that x_to_y in that case is a misnomer, it should be λ_to_y.
    # but as with a terminal constraint we are really just searching a
    # distribution over λ instead of x, but the rest stays the same.
    def x_to_y_terminalconstraint(λ, h=None):
        x = np.zeros(problem_params['nx'])
        costate = λ
        v = np.zeros(1)
        y = np.concatenate([x, costate, v])
        return y

    if problem_params['terminal_constraint']:
        # we instead assume a zero terminal constraint.
        x_to_y = x_to_y_terminalconstraint
    else:
        raise NotImplementedError('free terminal state has not been tested for a *long* time')
        x_to_y = x_to_y_terminalcost

    x_to_y_vmap = jax.vmap(lambda x: x_to_y(x))

    batch_pontryagin_backward_solver = make_pontryagin_solver(problem_params, algo_params)

    T = problem_params['T']

    wrapped_solver = lambda λT: batch_pontryagin_backward_solver(x_to_y_vmap(λT), T, 0)

    return wrapped_solver
