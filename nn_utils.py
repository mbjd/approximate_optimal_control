# jax
import jax
import jax.numpy as np
import optax
import diffrax

# cheating on equinox :/
import flax
from flax import linen as nn
from typing import Sequence, Optional

# other, trivial stuff
import numpy as onp

import tk as tkinter
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as pl

import ipdb

import time
from tqdm import tqdm



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

        return x



class nn_wrapper():

    # all the usual business logic around the NN.
    # initialisation, data loading, training, loss plotting

    def __init__(self, input_dim, layer_dims, output_dim, key):

        self.input_dim  = input_dim
        self.layer_dims = layer_dims
        self.output_dim = output_dim

        self.nn = my_nn_flax(features=layer_dims, output_dim=output_dim)

        # could make these arguments as well...
        factor = .2
        lr_schedule = optax.piecewise_constant_schedule(
                init_value=0.05,
                boundaries_and_scales = {
                    100: factor,
                    200: factor,
                    300: factor,
                    500: factor,
                    1000: factor,
                    2000: factor,
                }
        )

        self.optim = optax.adam(learning_rate=lr_schedule)

        self.loss_val_grad = jax.value_and_grad(self.loss, argnums=0)

        # turns out if we remove the last dimension (which made everything a
        # column vector) we stop having the weird problem of wrong output shape.
        self.params = self.nn.init(key, np.zeros((self.input_dim,)))

        self.opt_state = self.optim.init(self.params)

    # so we can use it like a function :)
    def __call__(self, x):
        return self.nn.apply(self.params, x)

    # just an example loss function, very standard.
    def point_loss(self, params, x, y):
        y_pred = self.nn.apply(params, x)
        return (y_pred - y)**2

    def loss(self, params, xs, ys):
        losses = jax.vmap(self.point_loss, in_axes=(None, 0, 0))(params, xs, ys)
        return np.mean(losses)


    def train(self, xs, ys, algo_params, key):

        batchsize = algo_params['nn_batchsize']
        N_epochs = algo_params['nn_N_epochs']


        # make a test set or not.
        testset_fraction = algo_params['nn_testset_fraction']
        assert testset_fraction <= 1, 'testset cannot be larger than total data, dumbass'
        assert 0 <= testset_fraction, 'who in their right mind would want a test set of negative size? '
        testset_exists = testset_fraction > 0

        N_datapts = xs.shape[0]
        xs_test = ys_test = None

        if testset_exists:
            # prepare test data, shrink main data.
            N_train = int((1-testset_fraction) * N_datapts)

            split_idx = np.array([N_train])
            xs, xs_test = np.split(xs, split_idx)
            ys, ys_test = np.split(ys, split_idx)

            # so that this still refers to just training data
            N_datapts = xs.shape[0]



        # make minibatches

        N_datapts_equal_batches = N_datapts - N_datapts % batchsize

        N_batches = N_datapts_equal_batches / batchsize
        assert N_batches == int(N_batches), 'N_batches size not integer'
        N_batches = int(N_batches)

        @jax.jit
        def generate_minibatch_index_array(key):

            # discard a bit of data to make equally sized batches
            data_idx_shuffled = jax.random.permutation(key, np.arange(N_datapts))[0:N_datapts_equal_batches]

            # batched_data_idx[i, :] = data indices for batch number i.
            batched_data_idx = data_idx_shuffled.reshape(N_batches, batchsize)
            return batched_data_idx


        total_iters = N_batches * N_epochs
        losses = onp.zeros(total_iters)
        test_losses = onp.zeros(total_iters)

        @jax.jit
        def update_step(xs, ys, opt_state, params):

            loss_val, loss_grad = self.loss_val_grad(params, xs, ys)

            updates, opt_state = self.optim.update(loss_grad, opt_state)
            params = optax.apply_updates(params, updates)

            return opt_state, params, loss_val

        @jax.jit
        def eval_test_loss(xs, ys, params):
            return self.loss(params, xs, ys)

        for i in tqdm(range(total_iters)):

            # re-shuffle batch after epoch.
            i_batch = i % N_batches
            if i_batch == 0:
                key, subkey = jax.random.split(key)
                batched_data_idx = generate_minibatch_index_array(subkey)

            # get the batch
            batch_idx = batched_data_idx[i_batch, :]

            xs_batch = xs[batch_idx]
            ys_batch = ys[batch_idx]

            self.opt_state, self.params, loss_val = update_step(
                    xs_batch, ys_batch, self.opt_state, self.params
            )

            losses[i] = loss_val

            if testset_exists:
                test_losses[i] = eval_test_loss(xs_test, ys_test, self.params)

        if algo_params['nn_plot_training']:
            pl.semilogy(losses, label='training loss')
            if testset_exists:
                pl.semilogy(test_losses, label='test loss')
            pl.legend()
            pl.show()

