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
from functools import partial



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

    def __init__(self, input_dim, layer_dims, output_dim):

        self.input_dim  = input_dim
        self.layer_dims = layer_dims
        self.output_dim = output_dim

        self.nn = my_nn_flax(features=layer_dims, output_dim=output_dim)


    # so we can use it like a function :)
    def __call__(self, x, params):
        return self.nn.apply(params, x)

    # just an example loss function, very standard.
    def point_loss(self, params, x, y):
        y_pred = self.nn.apply(params, x)
        return (y_pred - y)**2

    def loss(self, params, xs, ys):
        losses = jax.vmap(self.point_loss, in_axes=(None, 0, 0))(params, xs, ys)
        return np.mean(losses)

    def init_nn_params(self, key):
        params = self.nn.init(key, np.zeros((self.input_dim,)))
        return params


    # this will not work because of the self argument
    # TODO find out in the future whether we can somehow jit this whole function
    @partial(jax.jit, static_argnames=['self', 'algo_params'])
    def train(self, xs, ys, nn_params, algo_params, key):

        batchsize = algo_params['nn_batchsize']
        N_epochs = algo_params['nn_N_epochs']


        # make a test set or not.
        testset_fraction = algo_params['nn_testset_fraction']
        assert testset_fraction <= 1, 'testset cannot be larger than total data, dumbass'
        assert 0 <= testset_fraction, 'who in their right mind would want a test set of negative size? '
        testset_exists = testset_fraction > 0

        N_datapts = xs.shape[0]
        xs_test = ys_test = None

        # this type of control flow should be invisible to jit tracing, as for every new value
        # of algo_params (static argname) we re-jit the function
        if testset_exists:
            # prepare test data, shrink main data.
            # for 'compile-time constants' we use onp here, otherwise it tries
            # to make tracer, even though it depends on static argument.

            split_idx_float = (1-testset_fraction) * N_datapts
            split_idx_float_array = onp.array([split_idx_float])
            split_idx_int_array = split_idx_float_array.astype(onp.int32)

            xs, xs_test = np.split(xs, split_idx_int_array)
            ys, ys_test = np.split(ys, split_idx_int_array)

            # so that this still refers to just training data
            N_datapts = xs.shape[0]


        # make minibatches
        # throw away some data to make batch size evenly divide the data
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

        total_iters = N_batches * N_epochs
        losses = np.zeros(total_iters)
        test_losses = np.zeros(total_iters)

        # turns out if we remove the last dimension (which made everything a
        # column vector) we stop having the weird problem of wrong output shape.

        optim = optax.adam(learning_rate=lr_schedule)
        opt_state = optim.init(nn_params)


        # @jax.jit
        def update_step(xs, ys, opt_state, params):

            loss_val_grad = jax.value_and_grad(self.loss, argnums=0)
            loss_val, loss_grad = loss_val_grad(params, xs, ys)

            updates, opt_state = optim.update(loss_grad, opt_state)
            params = optax.apply_updates(params, updates)

            return opt_state, params, loss_val

        # @jax.jit
        def eval_test_loss(xs, ys, params):
            return self.loss(params, xs, ys)

        # TODO use lax control flow here...
        # look at jax.lax.scan to do this:
        # basically a fancy version to write a for loop, but nice for jit

        #  input 0       input 1       input 2      ......
        #    v             v             v
        #  comp. 0   >   comp. 1   >   comp. 2      ......
        #    v             v             v
        #  result 0      result 1      result 2      ......

        # what does the computation need as inputs?
        # i (to know which batch to take)
        #   or the batch itself already...
        #
        # state to pass to next computation is basically (opt_state, nn_params)
        #
        # outputs/results would be train / test loss

        # shuffle data
        key, subkey = jax.random.split(key)
        all_batch_data_indices = generate_minibatch_index_array(subkey)

        def f_scan(carry, input_slice):
            # unpack the 'carry' state
            opt_state, nn_params, i_batch = carry

            # obtain minibatch
            # is this usage of all_batch_data_indices already functionally impure?
            # or is it just becoming a constant within a pure function?
            batch_data_indices = all_batch_data_indices[i_batch, :]

            xs_batch = xs[batch_data_indices]
            ys_batch = ys[batch_data_indices]

            # profit
            opt_state_new, nn_params_new, loss = update_step(
                    xs_batch, ys_batch, opt_state, nn_params
            )

            # output & carry state
            test_loss = eval_test_loss(xs_test, ys_test, nn_params)
            output_slice = (loss, test_loss)

            new_carry = (opt_state_new, nn_params_new, (i_batch + 1) % N_batches)

            return new_carry, output_slice

        # the training loop!
        # currently the input argument xs is unused (-> None) -- we keep track of the batch
        # number in the i_batch counter, and the map from batch index to data is 'baked' into
        # the f_scan function. probably the nicer way would be to have the data as input.
        # but it works as it is \o/
        init_carry = (opt_state, nn_params, 0)
        final_carry, outputs = jax.lax.scan(f_scan, init_carry, None, length=total_iters)

        opt_state, nn_params, _ = final_carry
        train_losses, test_losses = outputs

        return nn_params, train_losses, test_losses


    # last working version before changing to lax control flow for jit compatibility,
    # and probably also changing input/output signature a bit
    def train_no_jit(self, xs, ys, nn_params, algo_params, key):

        batchsize = algo_params['nn_batchsize']
        N_epochs = algo_params['nn_N_epochs']


        # make a test set or not.
        testset_fraction = algo_params['nn_testset_fraction']
        assert testset_fraction <= 1, 'testset cannot be larger than total data, dumbass'
        assert 0 <= testset_fraction, 'who in their right mind would want a test set of negative size? '
        testset_exists = testset_fraction > 0

        N_datapts = xs.shape[0]
        xs_test = ys_test = None

        # this type of control flow should be invisible to jit tracing, as for every new value
        # of algo_params (static argname) we re-jit the function
        if testset_exists:
            # prepare test data, shrink main data.
            # for 'compile-time constants' we use onp here, otherwise it tries
            # to make tracer, even though it depends on static argument.

            split_idx_float = (1-testset_fraction) * N_datapts
            split_idx_float_array = onp.array([split_idx_float])
            split_idx_int_array = split_idx_float_array.astype(onp.int32)

            xs, xs_test = np.split(xs, split_idx_int_array)
            ys, ys_test = np.split(ys, split_idx_int_array)

            # so that this still refers to just training data
            N_datapts = xs.shape[0]


        # make minibatches
        # throw away some data to make batch size evenly divide the data
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

        total_iters = N_batches * N_epochs
        losses = np.zeros(total_iters)
        test_losses = np.zeros(total_iters)

        # turns out if we remove the last dimension (which made everything a
        # column vector) we stop having the weird problem of wrong output shape.

        optim = optax.adam(learning_rate=lr_schedule)
        opt_state = optim.init(nn_params)


        # @jax.jit
        def update_step(xs, ys, opt_state, params):

            loss_val_grad = jax.value_and_grad(self.loss, argnums=0)
            loss_val, loss_grad = loss_val_grad(params, xs, ys)

            updates, opt_state = optim.update(loss_grad, opt_state)
            params = optax.apply_updates(params, updates)

            return opt_state, params, loss_val

        # @jax.jit
        def eval_test_loss(xs, ys, params):
            return self.loss(params, xs, ys)

        for i in range(total_iters):

            i_batch = i % N_batches
            if i_batch == 0:
                key, subkey = jax.random.split(key)
                batched_data_idx = generate_minibatch_index_array(subkey)

            # get the batch
            batch_idx = batched_data_idx[i_batch, :]

            xs_batch = xs[batch_idx]
            ys_batch = ys[batch_idx]

            opt_state, nn_params, loss_val = update_step(
                    xs_batch, ys_batch, opt_state, nn_params
            )

            losses = losses.at[i].set(loss_val)

            if testset_exists:
                test_loss = eval_test_loss(xs_test, ys_test, nn_params)
                test_losses = test_losses.at[i].set(test_loss)

        if algo_params['nn_plot_training']:
            pl.semilogy(losses, label='training loss')
            if testset_exists:
                pl.semilogy(test_losses, label='test loss')
            pl.legend()
            pl.show()

        return nn_params


    # this does NOT work yet, this was an attempt to jit the whole train function
    # once and then save it for rerunning, but it seems to be kinda difficult.
    # probably in the unflatten function we need to match the __init__ of the class
    # but then do we recreate everything? we have kind of an involved constructor...

    def _tree_flatten(self):

        children = ()

        # static values / configuration parameters.
        aux_data = {
                'input_dim': self.input_dim,
                'layer_dims': self.layer_dims,
                'output_dim': self.output_dim,
        }

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
