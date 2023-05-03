# jax
import jax
import jax.numpy as np
import optax
import diffrax

# cheating on equinox :/
import flax
from flax import linen as nn
from typing import Sequence, Optional

# but not for everything
from equinox import filter_jit

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

    # same loss functions but for the prediction *gradient*
    def point_gradient_loss(self, params, x, label_grad):
        prediction_grad = jax.grad(self.nn.apply, argnums=1)(params, x)
        return (prediction_grad - label_grad)**2

    def gradient_loss(self, params, xs, y_grads):
        losses = jax.vmap(self.point_loss, in_axes=(None, 0, 0))(params, xs, y_grads)
        return np.mean(losses)

    def loss_with_grad(self, params, xs, ys, grad_penalty=0):
        # loss function considering value AND gradient error.
        # here, ys is not of shape (N, 1) but of shape (N, nx+1)
        # as constructed in:
        # array_juggling -> sol_array_to_train_data -> if algo_params['nn_V_gradient_penalty'] > 0.

        # the last axis contains the costate = value gradient at [0:nx]
        # the value at [-1].

        value_loss = self.loss(params, xs, ys[:, -1])
        gradient_loss = self.gradient_loss(params, xs, ys[:, 0:-1])

        # scale everything by 1 + grad_penalty so the losses still have similar magnitude
        full_loss = (value_loss + grad_penalty * gradient_loss) / (1 + grad_penalty)
        # return also the individual losses.

        aux_outputs = {
                'full_train_loss': full_loss,
                # watch out - these are the unscaled versions, so they don't add up to full_loss
                'value_train_loss': value_loss,
                'gradient_train_loss': gradient_loss,
        }

        return full_loss, aux_outputs



    def init_nn_params(self, key):
        params = self.nn.init(key, np.zeros((self.input_dim,)))
        return params



    @filter_jit
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

        def generate_minibatch_index_array(key):

            # discard a bit of data to make equally sized batches
            data_idx_shuffled = jax.random.permutation(key, np.arange(N_datapts))[0:N_datapts_equal_batches]

            # batched_data_idx[i, :] = data indices for batch number i.
            batched_data_idx = data_idx_shuffled.reshape(N_batches, batchsize)
            return batched_data_idx


        total_iters = N_batches * N_epochs

        # could make these arguments as well...
        factor = .2
        lr_schedule = optax.piecewise_constant_schedule(
                init_value=0.02,
                boundaries_and_scales = {
                    100: factor,
                    200: factor,
                    300: factor,
                    500: factor,
                    1000: factor,
                    2000: factor,
                }
        )


        lr_schedule = optax.exponential_decay(
                init_value = 0.01,
                transition_steps = total_iters,
                decay_rate = 1e-2,
                end_value=1e-6
        )

        losses = np.zeros(total_iters)
        test_losses = np.zeros(total_iters)

        optim = optax.adam(learning_rate=lr_schedule)
        opt_state = optim.init(nn_params)


        def update_step(xs, ys, opt_state, params):

            # full gradient-including loss function in all cases. makes it easier, and the
            # 0 * .... is probably thrown out in jit anyway when the penalty is 0.
            loss_fct = partial(self.loss_with_grad, grad_penalty=algo_params['nn_V_gradient_penalty'])
            loss_val_grad = jax.value_and_grad(loss_fct, argnums=0, has_aux=True)


            # the terminology is a bit confusing, but:
            # - loss(_val)_grad = gradient of whatever loss w.r.t. NN parameters
            # - loss_with_grad = loss function penalising value and value gradients w.r.t. inputs

            # now, the whole outut is a dictionary called aux_output with arbitrary keys and (numerical) values
            # when plotting we just iterate over whatever keys we get.
            (loss_val, aux_output), loss_grad = loss_val_grad(params, xs, ys)

            updates, opt_state = optim.update(loss_grad, opt_state)
            params = optax.apply_updates(params, updates)
            return opt_state, params, aux_output

        def eval_test_loss(xs, ys, params):
            return self.loss(params, xs, ys)

        def eval_test_loss_with_grad(xs, ys, params):

            test_loss, aux_output = self.loss_with_grad(params, xs, ys, grad_penalty=algo_params['nn_V_gradient_penalty'])
            return test_loss, aux_output


        # the training loop is written with jax.lax.scan
        # basically a fancy version to write a for loop, but nice for jit

        # iteration:       0            1            2
        #
        #                input        input        input
        #                  v            v            v
        # init state  >   comp.    >   comp.    >   comp.   >    ......
        #                  v            v            v
        #                result       result       result

        # inputs: NONE (all wrapped into f_scan)
        #
        # state to pass to next computation, called 'carry' in docs:
        # (nn params, optimiser state, minibatch index)
        #
        # outputs: (training loss, test loss)

        # shuffle data
        key, subkey = jax.random.split(key)
        all_batch_data_indices = generate_minibatch_index_array(subkey)

        def f_scan(carry, input_slice):
            # unpack the 'carry' state
            nn_params, opt_state, i_batch = carry

            # obtain minibatch
            # is this usage of all_batch_data_indices already functionally impure?
            # or is it just becoming a constant within a pure function?
            batch_data_indices = all_batch_data_indices[i_batch, :]

            xs_batch = xs[batch_data_indices]
            ys_batch = ys[batch_data_indices]

            # profit
            opt_state_new, nn_params_new, aux_output = update_step(
                    xs_batch, ys_batch, opt_state, nn_params
            )

            aux_output['lr'] = lr_schedule(opt_state[0].count)

            # if gradient penalty = 0, then loss_val is just a scalar loss
            # if gradient penalty > 0, then loss_val is a tuple:
            #   (full_loss, (value_loss, gradient_loss))

            # output & carry state
            test_loss, test_aux_output = eval_test_loss_with_grad(xs_test, ys_test, nn_params)

            # pro move
            for k in test_aux_output.keys():
                new_key = k.replace('train', 'test')
                aux_output[new_key] = test_aux_output[k]

            new_carry = (nn_params_new, opt_state_new, (i_batch + 1) % N_batches)

            return new_carry, aux_output

        # the training loop!
        # currently the input argument xs is unused (-> None) -- we keep track of the batch
        # number in the i_batch counter, and the map from batch index to data is 'baked' into
        # the f_scan function. probably the nicer way would be to have the data as input.
        # but it works as it is \o/
        init_carry = (nn_params, opt_state, 0)
        final_carry, outputs = jax.lax.scan(f_scan, init_carry, None, length=total_iters)

        nn_params, _, _ = final_carry

        return nn_params, outputs



    # now this works - I just had to make the constructor very simple
    # and do all the important stuff in the train() method, which is now
    # jitted so we can do whatever we want there without incurring huge
    # python overhead every time.
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
