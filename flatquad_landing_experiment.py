import jax
import jax.numpy as np
import matplotlib.pyplot as pl

import main
import pontryagin_utils

if __name__ == '__main__':

    # classic 2D quad type thing. 6D state.
    m = 20  # kg
    g = 9.81 # m/s^2
    r = 0.5 # m
    I = m * (r/2)**2 # kg m^2 / radian (???)
    umax = m * g * 1.2 / 2  # 20% above hover thrust

    def f(t, x, u):

        # unpack for easier names
        Fl, Fr = u
        posx, posy, vx, vy, Phi, omega = x

        xdot = np.array([
            vx,
            vy,
            -np.sin(Phi) * (Fl + Fr) / m,
            np.cos(Phi) * (Fl + Fr) / m - g,
            omega,
            (Fr-Fl) * r / I,
        ])

        return xdot

    def l(t, x, u):
        Fl, Fr = u
        posx, posy, vx, vy, Phi, omega = x

        state_length_scales = np.array([1, 1, 1, 1, np.deg2rad(10), np.deg2rad(45)])
        Q = np.diag(1/state_length_scales**2)
        state_cost = x.T @ Q @ x

        # can we just set an input penalty that is zero at hover?
        # penalise x acc, y acc, angular acc here.
        # this here is basically a state-dependent linear map of the inputs, i.e. M(x) u with M(x) a 3x2 matrix.
        # the overall input cost will be acc.T M(x).T Q M(x) acc, so for each state it is still a nice quadratic in u.
        accelerations = np.array([
            -np.sin(Phi) * (Fl + Fr) / m,
            np.cos(Phi) * (Fl + Fr) / m - g,
            (Fr - Fl) * r / I,
        ])

        accelerations_lengthscale = np.array([1, 1, 1])

        input_cost = accelerations.T @ np.diag(1/accelerations_lengthscale**2) @ accelerations

        return state_cost + input_cost


    def h(x):
        # irrelevant if terminal constraint
        Qf = 1 * np.eye(6)
        return (x.T @ Qf @ x).reshape()


    state_length_scales = np.array([2, 1, 2, 1, np.deg2rad(45), np.deg2rad(45)])
    x_sample_cov = np.diag(state_length_scales**2)

    problem_params = {
        'system_name': 'flatquad',
        'f': f,
        'l': l,
        'h': h,
        'T': 2,
        'nx': 6,
        'nu': 2,
        'U_interval': [np.zeros(2), umax*np.ones(2)],  # but now 2 dim!
        'terminal_constraint': True,
        'V_max': 2,
    }

    algo_params = algo_params = {
            'pontryagin_solver_dt': 1/1024,
            'pontryagin_solver_adaptive': True,
            'pontryagin_solver_rtol': 1e-6,
            'pontryagin_solver_atol': 1e-4,
            'pontryagin_solver_dense': False,
            'pontryagin_solver_maxsteps': 1024,

            'sampler_plot': True,

            'sampler_dt': 1 / 64,
            'sampler_burn_in': 8,
            'sampler_N_chains': 1,  # with pmap this has to be 4
            'samper_samples_per_chain': 2 ** 8,  # actual samples = N_chains * samples
            'sampler_steps_per_sample': 32,
            'sampler_plot': False,
            'sampler_tqdm': True,

            'x_sample_cov': x_sample_cov/2,
    }

    # pontryagin_solver = pontryagin_utils.make_pontryagin_solver(problem_params, algo_params)
    wrapped_pontryagin_solver = pontryagin_utils.make_pontryagin_solver_wrapped(problem_params, algo_params)
    # for resting the inner min_u H(u). i think it works now.
    # pontryagin_utils.u_star_2d(np.zeros(6), np.zeros(6), problem_params)
    # pontryagin_utils.u_star_2d(np.zeros(6), np.array([0, 0, 0, 0, 10, 100]), problem_params)
    #
    # for a in np.linspace(0, 120, 121):
    #     pontryagin_utils.u_star_2d(np.zeros(6), a * np.array([0, 0, 0, 0, 0, 1]), problem_params)


    # to test the pontryagin solver.
    # N_trajs = 128
    # lamTs = jax.random.normal(jax.random.PRNGKey(0), shape=(N_trajs, 6)) @ np.diag(10*np.array([0.05, 1, 0.05, 1, 0.01, 0.001]))
    #
    # sol, _ = pontryagin_solver(np.column_stack([0*lamTs, lamTs, np.zeros((N_trajs))]), 1, 0)
    #
    # names = ('x', 'y', 'vx', 'vy', 'Phi', 'omega')
    # for i in range(6):
    #     pl.subplot(611 + i)
    #     pl.plot(sol.ts.T, sol.ys[:, :, i].T, color='black', alpha=0.1)
    #     pl.ylabel(names[i])
    #
    # pl.figure()
    #
    # pl.plot(sol.ys[:, :, 0].T, sol.ys[:, :, 1].T, alpha=0.2, color='black')
    # pl.show()

    # lamTs = np.linspace(np.zeros(6), np.array([0, 0, 0, 0, 0.1, 0]), 11)
    # lamTs = jax.random.normal(jax.random.PRNGKey(0), shape=(512, 6)) @ np.diag(np.array([1, 1, 1, 1, 0.01, 0.001]))

    # y0s, lamTs = main.sample_uniform(problem_params, algo_params, jax.random.PRNGKey(1))

    sysname = problem_params['system_name']
    lamTs = np.load(f'datasets/last_lamTs_{sysname}.npy')
    y0s = np.load(f'datasets/last_y0s_{sysname}.npy')

    print((np.diff(y0s[:, 0]) != 0).mean())
    lamTs_chosen = jax.random.choice(jax.random.PRNGKey(1), lamTs, (500,), axis=0)
    sols, y0s = wrapped_pontryagin_solver(lamTs_chosen)
    # dy0_dlamT = jax.jit(jax.jacobian(lambda lamT: wrapped_pontryagin_solver(lamT.reshape(1, -1))[1].squeeze()[0:problem_params['nx']]))

    # having lots of trouble using sols.evaluate or sols.interpolation.evaluate in any way. thus, just use sol.ts \o/
    def plot_trajectories(ts, ys, color='green', alpha='.1'):

        # plot trajectories.
        pl.plot(ys[:, :, 0].T, ys[:, :, 1].T, color='green', alpha=.1)

        # sols.ys.shape = (N_trajectories, N_ts, 2*nx+1)
        # plot attitude with quiver.
        arrow_x = ys[:, :, 0].reshape(-1)
        arrow_y = ys[:, :, 1].reshape(-1)
        attitudes = ys[:, :, 4].reshape(-1)
        arrow_len = 0.5
        u = np.sin(-attitudes) * arrow_len
        v = np.cos(attitudes) * arrow_len

        pl.quiver(arrow_x, arrow_y, u, v, color='green', alpha=0.1)

    # plot_trajectories(sols.ts, sols.ys)

    # pl.figure()
    # v0s = sols.ys[:, :, -1].reshape(-1)
    # v0s = np.nan_to_num(v0s, posinf=-1)
    # pl.hist(v0s, bins=20)
    # # pl.show()

    pl.figure()
    # plot all v(t)'s
    pl.plot(sols.ts[:, :].T, sols.ys[:, :, -1].T, alpha=.1, color='blue')
    pl.show()




    print('hi')


