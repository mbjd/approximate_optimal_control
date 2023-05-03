import jax
import jax.numpy as np

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
    zero_u = np.zeros((1, 1))
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
        state_dot   =  jax.jacobian(H, argnums=2)(state, u_star, costate).reshape(nx, 1)
        costate_dot = -jax.jacobian(H, argnums=0)(state, u_star, costate).reshape(nx, 1)
        value_dot   = -l(t, state, u_star)

        y_dot = np.vstack([state_dot, costate_dot, value_dot])
        return y_dot

    return f_forward
