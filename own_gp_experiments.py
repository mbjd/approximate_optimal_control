#!/usr/bin/env python

import jax
import jax.numpy as np

# this is put on hold, probably a dumb idea to implement it from scratch...

class GPWithGradientObservations():
    '''
    what do we do here?
    - self.params with all hyperparameters, method to minimise -log(marginal likelihood)
    - some way to specify kernels? or something
    - derivative kernel stuff
    '''
    def __init__(self):
        # kernel_fct should be a function taking in two x points as
        # jax.numpy.array of shape (xdim).

        # we model a function f as in y = f(x).
        # also we like to incorporate gradient information, that is,
        # measurements of dy/dx. In out application these will occur
        # the same inputs as the observations of y, so same xs here.

        self.xs = None
        self.ys = None
        self.y_grads = None

        if kernel_fct is not None:
            self.kernel_fct = kernel_fct
        else:
            # standard: RBF kernel with dimension dependent length scale.
            def kernel_fct(x1, x2, params):
                length_scales = params['length_scales']




    def predict(self, xs):
        '''
        TODO write prediction code.
        '''
        pass

