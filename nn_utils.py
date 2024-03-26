# jax
import jax
import jax.numpy as np
import optax
import jax_tqdm

# cheating on equinox :/
import flax
from flax import linen as nn
from typing import Sequence, Optional

# but not for everything
from equinox import filter_jit

# other, trivial stuff
# import numpy as onp
# import matplotlib.pyplot as pl

import ipdb

# from tqdm import tqdm
from functools import partial
from misc import *







def train_test_split(ys, train_frac=0.9):

    assert 0. < train_frac <= 1., '<gordon ramsey voice> this "training fraction" is not even a fraction you donkey'

    split_idx = int(train_frac * ys['x'].shape[0])

    train_ys = jax.tree_util.tree_map(lambda n: n[:split_idx], ys)
    test_ys  = jax.tree_util.tree_map(lambda n: n[split_idx:], ys)

    return train_ys, test_ys



class coordinate_transformer(object):

    '''

    before fitting the NN, we like to transform the state to a different state
    space, z = T(x). whereas the original state space is isomorphic to R^n and
    thus easy for simulation and control theory, it often does not accurately
    reflect state space topology. Thus we transform to a new state space which
    is usually a nx-dimensional manifold embedded in R^nz (nz > nx).

    For standard machine learning tasks this is trivial, but with gradient and
    hessian observation we need to be careful to transform them correctly.
    this class does that.

    canonical example: if x is an angle theta, z can be [cos theta, sin theta].

    we have a doppelt gemoppelt kind of situation here. The manifold Z is
    defined in two ways:
     - Z = T(X). whatevern the transformation sends the state space to
     - Z = {z: c(z) = 0} with a constraint function c. 

    The first gives an easy way to find a tangent space basis, the second 
    a normal space basis. We could do with just one but then we'd have to do some
    hacky SVD type stuff to find the other basis, subject to numerical errors. 
    this also serves as a sanity check that the manifold is indeed what we defined 
    it to be. 

    -> update, maybe this is dumb. is it not simpler to just define the functions
    V_nn(T(x)) in the loss, and differentiate the whole thing wrt x? 

    '''

    def __init__(self, problem_params):

        assert 'T' in problem_params, 'coordinate transformation T must be given in problem_params'
        assert 'c' in problem_params, 'constraint function c must be given in problem_params'

        # if the functions have incorrect inputs/outputs i guess we will find out later

        # T: X -> Z defines manifold as Z = T(X)
        self.T = problem_params['T']
        # c: R^nz -> R^(nz-nx) defines manifold as Z = {z: c(z) = 0}
        self.c = problem_params['c']

        # for these 'sanity check' evaluations it would be nice if the
        # transformations don't have any sort of singularity at 0. 

        # dimensionality of original state space
        self.nx = problem_params['nx']
        # dimensionality of euclidean space in which transformed state space is embedded
        self.nz = self.T(np.zeros(self.nx)).shape[0]

        ndiff = self.nz - self.nx
        nc = self.c(np.zeros(self.nz)).shape[0]
        if not ndiff == nc:
            print('manifold dimensions somehow wrong.')
            print(f'dim(original state space X) = {self.nx}')
            print(f'dim(embedding space) = {self.nz}')
            print(f'dim(constraint function) = {self.nc}')
            raise ValueError('look at what I just printed dumbass')


        dTdx = jax.jacobian(T)

    def transform_y(y):

        pass
        # y a dict with entries 'x', 'v', 'vx' (, 'vxx' optionally)

        # what do we want in the transformed version? 





class data_normaliser(object):

    def __init__(self, train_ode_states):

        # train_ode_states: dict with entries 'x', 'v', 'vx'.
        # ['x'].shape == (N_pts, nx)
        # ['v'].shape == (N_pts,)
        # ['vx'].shape == (N_pts, nx)

        # we scale each x component and v to zero mean and unit variance.
        # then some standard differentiation rules tell us how vx and vxx
        # must be changed under the linear change of variables.

        x_means = train_ode_states['x'].mean(axis=0)
        x_stds  = train_ode_states['x'].std(axis=0)

        self.normalise_x = lambda x: (x - x_means) / x_stds
        self.unnormalise_x = lambda xn: xn * x_stds + x_means

        v_mean = train_ode_states['v'].mean()
        v_std  = train_ode_states['v'].std()

        self.normalise_v = lambda v: (v-v_mean) / v_std
        self.unnormalise_v = lambda vn: vn * v_std + v_mean

        # now the hard part, vx.
        self.normalise_vx = lambda vx: vx * x_stds / v_std
        self.unnormalise_vx = lambda vx_n: (vx_n / x_stds) * v_std

        # proper multivariate version would be: vxx transformed = A vxx A.T
        # where A is the coordinate transformation
        # https://math.stackexchange.com/questions/1514680/gradient-and-hessian-for-linear-change-of-coordinates
        self.normalise_vxx = lambda vxx: np.diag(x_stds) @ vxx @ np.diag(x_stds).T / v_std
        self.unnormalise_vxx = lambda vxx_n: np.diag(1/x_stds) @ (vxx_n * v_std) @ np.diag(1/x_stds)

    def normalise_all(self, train_ode_states):
        # old format where everything was stacked into an array
        print('normaliser: old format is not recommended!')
        nn_xs  = jax.vmap(self.normalise_x)(train_ode_states['x'])
        nn_vs  = jax.vmap(self.normalise_v)(train_ode_states['v'])
        nn_vxs = jax.vmap(self.normalise_vx)(train_ode_states['vx'])
        # to get the data format expected by the nn code...
        # maybe change this so that we can use a nicer dict format?
        # with entries 'x' 'v' 'vx' and maybe 'vxx'?
        nn_ys  = np.column_stack([nn_vxs, nn_vs])

        return nn_xs, nn_ys

    def normalise_all_dict(self, train_ode_states):

        oup = {
            'x': jax.vmap(self.normalise_x)(train_ode_states['x']),
            'v': jax.vmap(self.normalise_v)(train_ode_states['v']),
            'vx': jax.vmap(self.normalise_vx)(train_ode_states['vx']),
        }

        if 'vxx' in train_ode_states:
            oup['vxx'] = jax.vmap(self.normalise_vxx)(train_ode_states['vxx'])

        return oup



class my_nn_flax(nn.Module):

    # simple, fully connected NN class.
    # for bells & wistles -> nn_wrapper class :)

    features: Sequence[int]
    output_dim: Optional[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(features=feat)(x)
            x = nn.softplus(x)

        if self.output_dim is not None:
            x = nn.Dense(features=self.output_dim)(x)

        # return x.reshape()  # finally get rid of all those shitty (.., 1) shapes
        return x.squeeze()  # finally get rid of all those shitty (.., 1) shapes


class nn_wrapper():

    # all the usual business logic around the NN.
    # initialisation, data loading, training, loss plotting

    def __init__(self, input_dim, layer_dims, output_dim, has_t=False):

        self.input_dim  = input_dim
        self.layer_dims = layer_dims
        self.output_dim = output_dim

        self.nn = my_nn_flax(features=layer_dims, output_dim=output_dim)

        # somehow this won't work if we put exactly the same but as a decorator.
        self.ensemble_init_and_train = partial(jax.vmap, in_axes=(0, None, None, None))(self.init_and_train)


    # so we can use it like a function :)
    def __call__(self, params, x):
        return self.nn.apply(params, x)


    def init_nn_params(self, key):
        params = self.nn.init(key, np.zeros((self.input_dim,)))
        return params


    def init_and_train(self, key, xs, ys, algo_params):
        '''
        key(s): an array of PRNG keys
        other arguments same as train.

        returns: params, a dict just like usual, but each entry has an extra leading
                 dimension arising from the vmap.
        '''

        raise NotImplementedError('this still uses old data format')
        params = self.nn.init(key, np.zeros((self.input_dim,)))
        params, outputs = self.train(xs, ys, params, algo_params, key)
        return params, outputs


    def ensemble_mean_std(self, params, xs):
        '''
        each node of params should have an extra leading dimension,
        as generated by ensemble_init_and_train.

        returns two arrays of shape (N_points, 1+nx), where the second index
        is 0 for the value output and 1...nx+1 for the costate/value gradient.
        '''

        raise NotImplementedError('this still uses old data format')
        outputs = jax.vmap(self.nn.apply, in_axes=(0, None))(params, xs)
        grad_outputs = jax.vmap(self.apply_grad, in_axes=(0, None))(params, xs)

        all_outputs = np.concatenate([outputs, grad_outputs], axis=2)

        means = all_outputs.mean(axis=0)
        stds = all_outputs.std(axis=0)

        return means, stds



    def sobolev_loss(self, key, y, params, algo_params):

        # this is for a *single* datapoint in dict form. vmap later.
        # needs a PRNG key for the hvp in random direction. this is
        # only a stochastic approximation of the actual sobolev loss.

        # y is the dict with training data.
        # tree_map(lambda z: z.shape, ys) should be: {
        #  'x': (nx,), 'v': (1,), 'vx': (nx,), 'vxx': (nx, nx)
        # }

        # algo_params['nn_sobolev_weights'] gives the (nonnegative!) relative
        # weights associated with the v, vx, and vxx losses. set the latter
        # one or two to imitate more "basic" nn variants.

        # this somehow messes up the batch vmap (i think?)
        # v_pred, vx_pred = jax.value_and_grad(self.nn.apply, argnums=1)(params, x)
        # maybe that is better? if squeeze()ing at the end of nn definition, shapes are the same

        nn_apply_fct = self.nn.apply


        if algo_params['use_transform']:

            raise NotImplementedError('lots of issues with this')
            # specifically the issue: the data we are getting here is normalised, but T 
            # operatoes in the unnormalised state space! 

            # what if here we instead just give the original data points, and make a "unified" 
            # transformation T which both does data normalisation and mapping to the manifold? 
            # then we can basically just specify losses asking that: 

            # basically, we "absorb" the whole data normalisation into the NN as a kind of 0-th layer
            # (but also the v scaling should influcence the output....)

            # V_nn(T(x)) = v(x)
            # jac_x [V_nn(T(x))] = v_x(x)
            # hess_x [V_nn(T(x))] = v_xx(x)  (both sides right multiplied with random v as always)

            # if we want to avoid doing the transformation every time (which is probably low overhead)
            # we would neet to store the following, for each training point:
            # 


            # replace the whole apply function by apply(params, .) ∘ T
            # this way we avoid all the differential geometry, tangent and normal spaces
            # everything is just a nonlinear smooth function and just works. 
            assert 'T' in algo_params
            T = algo_params['T']

            nn_apply_fct = lambda params, y: self.nn.apply(params, T(y))


        v_pred = self.nn.apply(params, y['x'])
        vx_pred = jax.jacobian(self.nn.apply, argnums=1)(params, y['x'])

        v_loss  = (v_pred - y['v' ]) ** 2
        vx_loss = np.sum((vx_pred - y['vx']) ** 2)

        # if there are three weights they are for (v, vx, vxx). if only two, (v, vx).
        if algo_params['nn_sobolev_weights'].shape == (3,):

            # instead of calculating the whole hessian (of v_pred wrt x) and comparing
            # it with y['vxx'], we instead compute the hessian vector product in a
            # random direction, inspired by https://arxiv.org/pdf/1706.04859.pdf.

            # the hvp is not the second directional derivative. If the hvp is
            # H d (a vector), the second directional derivative would be d.T H d (scalar).

            # the hvp has intuitive meaning if we drop one level of differentiation. say
            # lambda(x) = vx(x) is the costate function. Then vxx(x) is the gradient of that,
            # Dx lambda(x). Thus the value hvp is just the directional derivative of the costate
            # function in a random direction. should be great :)

            # is there some catch to do with data normalisation? are some directions
            # "more likely" than others here? dunno really.

            # random vector on unit sphere.
            # would it be just as good to just choose one of the basis vectors [0, .., 1, .., 0]?
            # then we basically extract one column of the hessian. that idea is also mentioned here:
            # https://www.semanticscholar.org/reader/6edc6ff5a92567ff119f69266c291bab1285357f

            direction = jax.random.normal(key, shape=(self.input_dim,))
            direction = direction / np.linalg.norm(direction)

            # as 'training datapoint' the simple hessian-vector product.
            # it has the same size as the costate which seems reasonable.
            hvp_label = y['vxx'] @ direction

            # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#hessian-vector-products-with-grad-of-grad
            # tbh i have no clue how this works
            f = lambda x: self.nn.apply(params, x)
            hvp_pred = jax.grad( lambda x: np.vdot( jax.grad(f)(x), direction ) )(y['x'])

            # the naive way for comparison. seems to be correct :)
            # hvp_pred_naive = jax.hessian(self.nn.apply, argnums=1)(params, y['x']) @ direction
            # print(rnd(hvp_pred, hvp_pred_naive))

            vxx_loss = np.sum((hvp_label - hvp_pred)**2)

            # make convex combination by normalising weights.
            # multiplying this by a constant is the same as adjusting the learning rate so
            # we might as well take that degree of freedom away.
            weights = algo_params['nn_sobolev_weights'] / np.sum(algo_params['nn_sobolev_weights'])
            sobolev_losses = np.array([v_loss, vx_loss, vxx_loss])

            # we can have two outputs, the first of which is the one being differentiated if we use
            # jax.value_and_grad(..., has_aux=True) later.
            return weights @ sobolev_losses, sobolev_losses


        elif algo_params['nn_sobolev_weights'].shape == (2,):

            weights = algo_params['nn_sobolev_weights'] / np.sum(algo_params['nn_sobolev_weights'])
            sobolev_losses = np.array([v_loss, vx_loss])

            # append nan to aux output to have equal shapes for plotting
            sobolev_losses_emptyvxx = np.array([v_loss, vx_loss, np.nan])

            return weights @ sobolev_losses, sobolev_losses_emptyvxx

        else:
            raise ValueError('nn sobolev weight must be an array of shape (3,) (including vxx) or (2,) (without vxx)')


    # vmap the loss across a batch and get its mean.
    # tuple output so the gradient is only taken of the first argument below (with has_aux=True)
    def sobolev_loss_batch_mean(self, k, params, ys, algo_params):

        # the size of the actual batch, not what algo_params says.
        # then we can use the same function e.g. for evaluating loss on test set.
        ks = jax.random.split(k, ys['x'].shape[0])

        losses, loss_terms = jax.vmap(self.sobolev_loss, in_axes=(0, 0, None, None))(ks, ys, params, algo_params)

        # mean across batch dim.
        # should be scalar and (3,) respectively.
        return np.mean(losses), np.mean(loss_terms, axis=0)



    @filter_jit
    def train_sobolev(self, key, ys, nn_params, algo_params, ys_test=None):

        '''
        new training method. main changes wrt self.train:

         - data format is now this:
           ys a dict with keys:
             'x': (N_pts, nx) array of state space points just like before.
             'v': (N_pts, 1) array of value function evaluations at those x.
             'vx': (N_pts, nx) array of value gradient = costate evaluations
           optionally:
             'vxx': (N_pts, nx, nx) array of value hessians = costate jacobian evaluations.

           this should make it easy to train with or without hessians with the same code.
           maybe we can also initially train with v and vx and only "fine-tune" with the hessian?

         - loss includes (optionally) the hessian error. specifically a stochastic approximation
           of it: a hessian-vector product with a randomly chosen direction vector. idea from 
           Czarnecki et al.: https://arxiv.org/abs/1706.04859

         - testset generation not in here. do it using train_test_split in this file. pass
           ys_test to evaluate test loss during training (full test dataset every step!)

        TODO as of now the test set is just a random subset of all points.
        should we instead take a couple entire trajectories as test set? because
        if 90% of the points on some trajectory are in the training set it is kind of
        not a huge feat to have low loss on the remaining 10%
        if OTOH we test with entire trajectories unseen in training the test loss kind of
        is more meaningful...

        '''


        # make sure it is of correct shape?
        testset_exists = ys_test is not None

        # does this still make sense?
        N_datapts = ys['x'].shape[0]
        batchsize = algo_params['nn_batchsize']
        N_epochs = algo_params['nn_N_epochs']

        # we want: total_iters * batchsize == N_epochs * N_datapts. therefore:
        total_iters = (N_epochs * N_datapts) // batchsize

        # exponential decay. this will go down from lr_init to lr_final over
        # the whole training duration.
        # if lr_staircase, then instead of smooth decay, we have stepwise decay
        # with
        N_lr_steps = algo_params['lr_staircase_steps']

        total_decay = algo_params['lr_final'] / algo_params['lr_init']

        # regardless of whether or not steps are used, the decay rate 
        # sets the decay *per transition step*. 
        lr_schedule = optax.exponential_decay(
                init_value = algo_params['lr_init'],
                transition_steps = total_iters // N_lr_steps,
                decay_rate = (total_decay) ** (1/N_lr_steps),
                end_value=algo_params['lr_final'],
                staircase=algo_params['lr_staircase']
        )

        optim = optax.adam(learning_rate=lr_schedule)
        opt_state = optim.init(nn_params)

        # ipdb.set_trace()


        def update_step(key, ys, opt_state, params):

            # differentiate the whole thing wrt argument 1 = nn params.
            (loss, loss_terms), grad = jax.value_and_grad(self.sobolev_loss_batch_mean, argnums=1, has_aux=True)(
                key, params, ys, algo_params
            )

            updates, opt_state = optim.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            return opt_state, params, loss_terms


        def f_scan(carry, input_slice):
            # unpack the 'carry' state
            nn_params, opt_state, k = carry

            k_batch, k_loss, k_test, k_new = jax.random.split(k, 4)

            # obtain minibatch
            batch_idx = jax.random.choice(k_batch, N_datapts, (batchsize,))
            ys_batch = jax.tree_util.tree_map(lambda node: node[batch_idx], ys)

            # do the thing!!1!1!!1!
            opt_state_new, nn_params_new, loss_terms = update_step(
                k_loss, ys_batch, opt_state, nn_params
            )

            aux_output = {
                'lr': lr_schedule(opt_state[0].count),
                'train_loss_terms': loss_terms,
                # outputting ALL params here is possible but not recommended. 
                # will use huge memory and slow everything down (update, it seems equally fast during training...)
                # anyway vmapping the whole trainign procedure gives much better model diversity out of the box. 
                # 'params': nn_params_new,
            }

            # if given, calculate test loss.
            # probably quite expensive to do this every iteration though...
            # this if is "compile time"
            if ys_test is not None:
                k_test = jax.random.PRNGKey(0)  # just one sample. nicer plots :)
                test_loss, test_loss_terms = self.sobolev_loss_batch_mean(
                    k_test, nn_params_new, ys_test, algo_params
                )
                aux_output['test_loss_terms'] = test_loss_terms

            new_carry = (nn_params_new, opt_state_new, k_new)
            return new_carry, aux_output

        if algo_params['nn_progressbar']:
            # somehow this gives an error from within the library :(
            # NOT ANYMORE thanks patrick!!
            # https://github.com/mbjd/approximate_optimal_control/issues/1
            f_scan = jax_tqdm.scan_tqdm(n=total_iters)(f_scan)
            pass


        # the training loop!
        # currently the input argument is unused -- could also put the PRNG key there.
        # or sobolev loss weights if we decide to change them during training...
        init_carry = (nn_params, opt_state, key)
        final_carry, outputs = jax.lax.scan(f_scan, init_carry, np.arange(total_iters))

        nn_params, _, _ = final_carry

        return nn_params, outputs


    @filter_jit
    def train_sobolev_ensemble(self, key, ys, problem_params, algo_params, ys_test=None):

        # train ensemble by vmapping the whole training procedure with different prng key. 
        # now the key affects both initialisation and batch selection for each nn. 

        init_key, train_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, algo_params['nn_ensemble_size'])
        train_keys = jax.random.split(train_key, algo_params['nn_ensemble_size'])

        vmap_params_init = jax.vmap(self.nn.init, in_axes=(0, None))(init_keys, np.zeros(problem_params['nx']))

        # to trick around the optional argument. there is probably a neater way...
        train_with_key_and_params = lambda k, params_init: self.train_sobolev(k, ys, params_init, algo_params, ys_test=ys_test)
        
        return jax.vmap(train_with_key_and_params, in_axes=(0, 0))(train_keys, vmap_params_init)


    @filter_jit
    def train_sobolev_ensemble_from_params(self, key, ys, init_params_vmap, algo_params, ys_test=None):

        # train ensemble by vmapping the whole training procedure with
        # different prng key AND from vmapped params. 

        # is this implemented in some very wrong way? does this add another
        # axis of vmapping? suspiciously slow atm but only when passing test
        # data... is that the reason? if the test dataset is much larger than
        # the batches we would kind of expect that tbh

        keys = jax.random.split(key, algo_params['nn_ensemble_size'])

        train_with_key_and_params = lambda k, params: self.train_sobolev(k, ys, params, algo_params, ys_test=ys_test)

        # vmap only the random key. even keep same initialisation!
        return jax.vmap(train_with_key_and_params, in_axes=(0, 0))(keys, init_params_vmap)
