import jax
import torch

from typing import Literal
from jax import numpy as jnp
from jax import grad, jit, vmap

def accuracy(y, y_hat):
    return jnp.mean(y == y_hat)