import jax
import jax.numpy as np
import matplotlib.pyplot as pl
import rrt_sampler

import pontryagin_utils

import ipdb
import time
import numpy as onp
import tqdm

import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf



class TextTexture(geom.Texture):
    def __init__(self, text, font_size=100, font_face='sans-serif'):
        super(TextTexture, self).__init__()
        self.text = text
        # font_size will be passed to the JS side as is; however if the
        # text width exceeds canvas width, font_size will be reduced.
        self.font_size = font_size
        self.font_face = font_face

    def lower(self, object_data):
        return {
            u"uuid": self.uuid,
            u"type": u"_text",
            u"text": self.text,
            u"font_size": self.font_size,
            u"font_face": self.font_face,
        }


def plot_trajectories_meshcat(sols, vis=None, arrows=False, reparam=True, colormap=None, color=None, line=False):

    '''
    visualise flatquad trajectories nicely. 

    input is one of:
     - diffrax solution object from single solve with system state = sol.ys[:, 0:6]
     - diffrax solution object from vmapped solve w/ same format
     - diffrax solution object with pyTree solution, where sol.ys['x'] is the system state

    time axis is always taken literally, i.e. artificial time squeezing or v/t reparameterisation
    will mess things up. 
    '''


    # unpack correctly if solution state is a dict.
    # we only want system state in any case.
    if type(sols.ys) == dict:
        sols_ys = sols.ys['x']
        sols_ts = sols.ys['t']
    else:
        sols_ys = sols.ys
        sols_ts = sols.ts

    if len(sols_ys.shape) == 2:

        print('visualiser given single sol. sneakily making it look like several solutions.')

        sols_ys = sols_ys[None, :, :]
        sols_ts = sols_ts[None, :]

    # filter out the trajectories containing NaN.
    # inf is fine since the pre-allocated solution arrays are padded with inf. 
    # sols.ys.shape == (N_sols, N_timesteps, nx)
    # if for one solution, any state at any timestep contains NaN, don't show it. 
    is_usable = np.logical_not(np.isnan(sols_ys).any(axis=(1, 2)))
    sols_ys = sols_ys[is_usable]
    sols_ts = sols_ts[is_usable]

    if vis is None:
        vis = meshcat.Visualizer()
    else:
        print('using given visualiser. pray that the typing duck approves of it')
        print(f'it looks like this: {vis}')

    if sols_ys.shape[0] > 4000:

        print('meshcat visualiser: trajectory database large ({sols_ys.shape[0]})')
        print('probably you will encounter memory issues when opening the visualiser')

        # to save data, show random subsample, whatever. 
        ipdb.set_trace()

    if (sols_ts < 0).any():
        # negative time also seems to not work. Therefore, we first change -inf to +inf
        sols_ts = sols_ts.at[np.isneginf(sols.ts)].set(np.inf)

        min_t = sols_ts.min()      # which is < 0 at this point unless -inf were the only
        sols_ts = sols.ts - min_t  # subtracting negative = adding positive. big brain


    # scale force cylinder length like this:
    # vis['box/cyl_left_frame/cyl_left'].set_transform(tf.scale_matrix(0.1, direction=[0, 1, 0], origin=[0, -arrow_length/2, 0]))

    arrow_length = .25

    def make_quad(vis, basepath, color=None, opacity=1, disp_name=None):
        box_width = 1
        box_aspect = .1
        box = geom.Box([box_width*box_aspect, box_width, box_width*box_aspect**2])

        # show "quadrotor" as a flat-ish long box
        if color is None:
            # g.MeshPhongMaterial(map=g.TextTexture('Hello, world!')
            if disp_name is None:
                vis[basepath].set_object(box, geom.MeshLambertMaterial(opacity=opacity))
            else:
                disp_name = str(disp_name)  # in case it is an int or something
                material = geom.MeshLambertMaterial(opacity=opacity, map=TextTexture(disp_name))
                vis[basepath].set_object(box, material)


        else:
            vis[basepath].set_object(box, geom.MeshLambertMaterial(color, opacity=opacity))



        # also have two lines representing the rotor forces.
        # here they are of fixed length, we will animate their length
        # by setting a correspondingly scaling transform.
        # verts_l = np.array([[0, box_width/2, 0], [0, box_width/2, 1]]).T
        # vis['box']['arrow_l'].set_object(geom.Line(geom.PointsGeometry(verts_l), geom.MeshBasicMaterial(color=0xff0000)))
        # verts_r = np.array([[0, -box_width/2, 0], [0, -box_width/2, 1]]).T
        # vis['box']['arrow_r'].set_object(geom.Line(geom.PointsGeometry(verts_r), geom.MeshBasicMaterial(color=0xff0000)))

        # or, do them as cylinders to get thicker.
        # the cylinder is in a "cylinder frame" which moves the cylinder accordingly.

        if arrows:
            vis[basepath]['cyl_left_frame/cyl_left'].set_object(geom.Cylinder(arrow_length, .01), geom.MeshLambertMaterial(color=0xff0000))
            vis[basepath]['cyl_left_frame/cyl_left'].set_transform(onp.eye(4))
            vis[basepath]['cyl_left_frame'].set_transform(tf.concatenate_matrices(
                tf.translation_matrix([0, -box_width/2, arrow_length/2]),
                tf.rotation_matrix(np.pi/2, [1, 0, 0]),
            ))

            vis[basepath]['cyl_right_frame/cyl_right'].set_object(geom.Cylinder(arrow_length, .01), geom.MeshLambertMaterial(color=0xff0000))
            vis[basepath]['cyl_right_frame/cyl_right'].set_transform(onp.eye(4))

            vis[basepath]['cyl_right_frame'].set_transform(tf.concatenate_matrices(
                tf.translation_matrix([0, box_width/2, arrow_length/2]),
                tf.rotation_matrix(np.pi/2, [1, 0, 0]),
            ))

    # somehow though even when we put it into its own function it works perfectly when 
    # applied to the original visualiser but the force "arrows" are wrong when using frame
    def move_quad(vis, basepath, y):
        # vis = instance of meshcat visualiser, or frame in case of animation
        # y = extended system state with state, costate, time. 
        # names hardcoded ugly i know
        t = y[-1]
        T = tf.translation_matrix([0,y[0], y[1]])
        R = tf.rotation_matrix(y[2], np.array([1, 0, 0]))

        transform = tf.concatenate_matrices(T, R)
        vis[basepath].set_transform(transform)

        if arrows:
            nx = problem_params['nx']
            ustar = pontryagin_utils.u_star_2d(y[0:nx], y[nx:2*nx], problem_params)
            vis[basepath]['cyl_left_frame/cyl_left'].set_transform(tf.scale_matrix(ustar[0]/umax, direction=[0, 1, 0], origin=[0, -arrow_length/2, 0]))
            vis[basepath]['cyl_right_frame/cyl_right'].set_transform(tf.scale_matrix(ustar[1]/umax, direction=[0, 1, 0], origin=[0, -arrow_length/2, 0]))

    
    N_sols = sols_ys.shape[0]

    

    anim = meshcat.animation.Animation()

    # visualise which trajectories are close and which far from our sampled point. 
    # opacity ~ -shortest_traj_dists
    # closest = 0, furthest = 1
    # inv_opacities = (shortest_traj_dists - shortest_traj_dists.min()) / (shortest_traj_dists.max() - shortest_traj_dists.min())
    # opacities = 1 - inv_opacities

    assert not (colormap is not None and color is not None), 'cannot specify both color and colormap. choose one.'

    for sol_i in tqdm.tqdm(range(N_sols)):

        # this is just a hacky way to get different names if we re-use the same vis 
        # for plotting multiple vmapped sols. otherwise we choose the same names again
        # and overwrite the previous vis paths. essentially we make a "folder" for each
        # different vmapped sols object. 
        quad_name = f'{id(sols)}/quad_{sol_i}'

        if colormap is not None:
            color = pl.colormaps[colormap](sol_i/N_sols)
            r, g, b, a = color
            hexcolor = (int(r*255) << 16) + (int(g*255) << 8) + int(b*255)
            make_quad(vis, quad_name, color=hexcolor, opacity=a)
        elif color is not None:
            # color 
            r, g, b = color
            hexcolor = (int(r*255) << 16) + (int(g*255) << 8) + int(b*255)
            make_quad(vis, quad_name, color=hexcolor)
        else:  # it is None
            make_quad(vis, quad_name, disp_name=sol_i)

        min_t = sols_ys[sol_i, :, -1].min()
        for t, y in zip(sols_ts[sol_i], sols_ys[sol_i]):
            
            # data is inf-padded by diffrax. 
            if np.any(y == np.inf):
                break

            # this is not used anymore. for any time reparametersation, use a dict state with 
            # state['x'] = system state and state['t'] = physical time. if type(sol.ys) == dict
            # this form of state will be assumed (see start of function)
            '''
            if not t_is_indep:
                # assume instead that t is in the last 'extended state' variable. 
                # will fail silently (and produce very weird visualisation) if this some other variable
                t = y[-1]
            '''

            # apparently we have 30 fps and the animation time is in frames
            anim_t = 30*float(t)

            with anim.at_frame(vis, anim_t) as frame:
                move_quad(frame, quad_name, y)

        if line:
            # also plot the corresponding trajectory as a line. 
            # it wants a point array of shape nx3. 
            # sols_ys.shape is (N trajectories, N timesteps, 2nx+1)
            # only the points where we don't have inf. 
            pt_array = sols_ys[sol_i, sols_ts[sol_i] != np.inf, 0:2]
            # fill in the missing z values. 
            pt_array = np.column_stack([np.zeros((pt_array.shape[0], 1)), pt_array])

            # now i am really suffering from the lack of documentation. 
            # copy pasting these examples makes nothing appear in the viz. 
            # https://github.com/meshcat-dev/meshcat-python/blob/master/examples/lines.ipynb
            raise NotImplementedError('lines not supported yet :(.')


    # this un-animates any previous ones sadly :(
    vis.set_animation(anim, repetitions=np.inf)
    
    # schrödinger fancy/ugly color scheme...
    # vis['/Background'].set_property('top_color', [0xb5/256, 0x17/256, 0x9e/256])
    # vis['/Background'].set_property('bottom_color', [0x48/256, 0x0c/256, 0xa8/256])
    return vis
