#!/usr/bin/env python

import jax
import jax.numpy as jnp
import flax

import numpy as onp

A = jnp.arange(10)

bs = flax.serialization.msgpack_serialize(A)
B = flax.serialization.msgpack_restore(bs)

print(jax.tree_util.tree_map(type, A))
print(jax.tree_util.tree_map(type, B))
