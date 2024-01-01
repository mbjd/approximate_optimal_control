#!/usr/bin/env python

import jax
import jax.numpy as np
import matplotlib.pyplot as pl

import pontryagin_utils

import ipdb
import time

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
        posx, posy, Phi, vx, vy, omega = x

        xdot = np.array([
            vx,
            vy,
            omega,
            -np.sin(Phi) * (Fl + Fr) / m,
            np.cos(Phi) * (Fl + Fr) / m - g,
            (Fr-Fl) * r / I,
        ])

        return xdot

    def l(t, x, u):
        Fl, Fr = u
        posx, posy, Phi, vx, vy, omega = x

        state_length_scales = np.array([1, 1, np.deg2rad(10), 1, 1, np.deg2rad(45)])
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
        'V_max': 500,
    }


    algo_params = {
            'pontryagin_solver_dt': 2 ** -8,  # not really relevant if adaptive
            'pontryagin_solver_adaptive': True,
            'pontryagin_solver_atol': 1e-4,
            'pontryagin_solver_rtol': 1e-4,
            'pontryagin_solver_maxsteps': 1024,
            'pontryagin_solver_dense': False,
    }

    # similar to orbits example: find local lqr value function


    # important to linearise about equilibrium, not zero!
    # that linearisation would be uncontrollable in this case.
    # equilibrium in y direction:
    #   0 = np.cos(Phi) * (Fl + Fr) / m - g,
    #   0 =               (Fl + Fr) / m - g,
    #   Fl + Fr = mg
    # so, to also have no torque, we need: Fl = Fr = mg/2

    u_eq = np.ones(problem_params['nu']) * m * g / 2

    x_eq = np.zeros(problem_params['nx'])
    A = jax.jacobian(f, argnums=1)(0., x_eq, u_eq)
    B = jax.jacobian(f, argnums=2)(0., x_eq, u_eq).reshape((problem_params['nx'], problem_params['nu']))
    Q = jax.hessian(l, argnums=1)(0., x_eq, u_eq)
    R = jax.hessian(l, argnums=2)(0., x_eq, u_eq)

    # cheeky controllability test
    ctrb =  np.hstack([np.linalg.matrix_power(A, j) @ B for j in range(problem_params['nx'])])
    if np.linalg.matrix_rank(ctrb) < problem_params['nx']:
        raise ValueError('linearisation not controllable aaaaah what did you do idiot')

    K0_inf, P0_inf, eigvals = pontryagin_utils.lqr(A, B, Q, R)

    # compute sqrtm of P, the value function matrix.
    # this is to later map points x from a small unit sphere to points Phalf x which are on a small value level set.
    P_eigv, P_eigvec = np.linalg.eigh(P0_inf)
    Phalf = P_eigvec @ np.diag(np.sqrt(P_eigv)) @ P_eigvec.T

    maxdiff = np.abs(Phalf @ Phalf - P0_inf).max()
    assert maxdiff < 1e-4, 'sqrtm(P) calculation failed or inaccurate'

    N_trajectories = 10

    # find N points on the unit sphere.
    # well known approach: find N standard normal points, divide by norm.
    # key = jax.random.PRNGKey(0)
    key = jax.random.PRNGKey(int(time.time() * 10000))
    normal_pts = jax.random.normal(key, shape=(N_trajectories, problem_params['nx']))
    unitsphere_pts = normal_pts /  np.sqrt((normal_pts ** 2).sum(axis=1))[:, None]

    # warning, row vectors
    xTs = unitsphere_pts @ np.linalg.inv(Phalf) * 0.01
    # test if it worked
    vTs =  jax.vmap(lambda x: x @ P0_inf @ x.T)(xTs)
    assert np.allclose(vTs, vTs[0]), 'terminal values shitty'

    vT = vTs[0]

    # gradient of V(x) = x.T P x is 2P x, but here they are row vectors.
    lamTs = jax.vmap(lambda x: x @ P0_inf * 2)(xTs)

    yTs = np.hstack([xTs, lamTs, np.zeros((N_trajectories, 1))])

    pontryagin_solver = pontryagin_utils.make_pontryagin_solver_reparam(problem_params, algo_params)

    # sols = jax.vmap(pontryagin_solver)
    sol = pontryagin_solver(yTs[0], vTs[0], 10)
    sols = jax.vmap(pontryagin_solver, in_axes=(0, None, None))(yTs, vTs[0], problem_params['V_max'])



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

    def plot_trajectories_meshcat(ts, ys, vis=None):

        '''
        tiny first draft of meshcat visualisation :o

        this is with the 2D quad model so 3D is kind of pointless but cool to have later.
        to do:
         - find out why the time needs to be scaled (otw too fast)
         - find a sensible way to plot a 'swarm' of multiple trajectories at once
         - force arrows to show input
         - do some 3d thing :)
        '''

        # strip out the trailing (now leading bc. reversed back to physical time) infs
        ys = ys[ys[:, 0] < np.inf]
        ts = ts[ts < np.inf]

        import meshcat
        import meshcat.geometry as g
        import meshcat.transformations as tf

        new_vis = False
        if vis is None:
            vis = meshcat.Visualizer()
            new_vis = True

        box_width = 1
        box_aspect = .1
        box = g.Box([box_width*box_aspect, box_width, box_width*box_aspect**2])

        # show "quadrotor" as a flat-ish long box
        vis.delete()
        vis['box'].set_object(box)

        # also have two lines representing the rotor forces.
        # here they are of fixed length, we will animate their length
        # by setting a correspondingly scaling transform.
        # verts_l = np.array([[0, box_width/2, 0], [0, box_width/2, 1]]).T
        # vis['box']['arrow_l'].set_object(g.Line(g.PointsGeometry(verts_l), g.MeshBasicMaterial(color=0xff0000)))
        # verts_r = np.array([[0, -box_width/2, 0], [0, -box_width/2, 1]]).T
        # vis['box']['arrow_r'].set_object(g.Line(g.PointsGeometry(verts_r), g.MeshBasicMaterial(color=0xff0000)))

        # or, do them as cylinders to get thicker.
        # the cylinder is in a "cylinder frame" which moves the cylinder accordingly.

        arrow_length = .25
        vis['box/cyl_left_frame/cyl_left'].set_object(g.Cylinder(arrow_length, .01), g.MeshLambertMaterial(color=0xff0000))
        vis['box/cyl_left_frame'].set_transform(tf.concatenate_matrices(
            tf.translation_matrix([0, -box_width/2, arrow_length/2]),
            tf.rotation_matrix(np.pi/2, [1, 0, 0]),
        ))

        vis['box/cyl_right_frame/cyl_right'].set_object(g.Cylinder(arrow_length, .01), g.MeshLambertMaterial(color=0xff0000))
        vis['box/cyl_right_frame'].set_transform(tf.concatenate_matrices(
            tf.translation_matrix([0, box_width/2, arrow_length/2]),
            tf.rotation_matrix(np.pi/2, [1, 0, 0]),
        ))


        # scale force cylinder length like this:
        # vis['box/cyl_left_frame/cyl_left'].set_transform(tf.scale_matrix(0.1, direction=[0, 1, 0], origin=[0, -arrow_length/2, 0]))
        ipdb.set_trace()

        anim = meshcat.animation.Animation()

        # 'ts' is value here. we would like to go down the value.
        # actual t is y[-1] after x and λ.
        # minT = ts.min()
        minT = ys[:, -1].min()
        for (i, v) in enumerate(ts):
            y = ys[i]
            t = y[-1]
            T = tf.translation_matrix([0,y[0], y[1]])
            R = tf.rotation_matrix(y[2], np.array([1, 0, 0]))
            anim_t = 25*float(t - minT)
            print(f'i={i}, v={v}, t={t}, anim_t={anim_t}')
            with anim.at_frame(vis, anim_t) as frame:

                # set rigid body position from rotation and translation.
                transform = tf.concatenate_matrices(T, R)
                frame['box'].set_transform(transform)

                # and show the force 'arrows'
                # why does this result in weird stuff even though if we do the same thing by hand in ipdb, before anim=..., it does exactly the right thing???
                nx = problem_params['nx']
                ustar = pontryagin_utils.u_star_2d(y[0:nx], y[nx:2*nx], problem_params)
                print(ustar/umax)
                frame['box/cyl_left_frame/cyl_left'].set_transform(tf.scale_matrix(ustar[0]/umax, direction=[0, 1, 0], origin=[0, -arrow_length/2, 0]))
                frame['box/cyl_right_frame/cyl_right'].set_transform(tf.scale_matrix(ustar[1]/umax, direction=[0, 1, 0], origin=[0, -arrow_length/2, 0]))

        vis.set_animation(anim, repetitions=np.inf)

        if new_vis:
            vis.open()

        return vis

    # just one trajectory atm
    # ts = np.linspace

    vis = None
    for j in range(N_trajectories):
        vis = plot_trajectories_meshcat(sols.ts[j], sols.ys[j], vis)
        ipdb.set_trace()

    # plot_trajectories(sols.ts, sols.ys)

    ipdb.set_trace()

