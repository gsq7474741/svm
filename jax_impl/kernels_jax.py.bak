import jax
import torch

from typing import Literal
from jax import numpy as jnp
from jax import grad, jit, vmap

KERNELS = {"linear": 0, "polynomial": 1, "gaussian": 2}


def linear_kernel(x_, x):
    r"""
    Linear kernel function.

    :param x_: (D) support vectors
    :param x: (D) input data
    :return: (1) kernel values
    """

    return jnp.dot(x_, x.T)


def polynomial_kernel(x_, x, p=2):
    r"""
    Polynomial kernel function.

    :param x_: (D) support vectors
    :param x: (D) input data
    :param p: (1) kernel degree
    :return: (1) kernel values
    """

    return (jnp.dot(x_, x.T) + 1) ** p


def rbf_kernel(x_, z, sigma=1):
    r"""
    Gaussian kernel function. (RBF)

    :param x_: (D) support vectors
    :param z: (D) input data
    :param sigma: (1) kernel width
    :return: (1) kernel values
    """

    return jnp.exp(-jnp.sum((x_ - z) ** 2, axis=1) / (2 * (sigma ** 2)))


def kernel_map(x_, z, kernel: int):
    if kernel == 0:
        return linear_kernel(x_, z)
    elif kernel == 1:
        return polynomial_kernel(x_, z)
    elif kernel == 2:
        return rbf_kernel(x_, z)
    else:
        raise ValueError(f"Kernel {kernel} not supported")
