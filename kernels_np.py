import torch

import numpy as np

KERNELS = {"linear": 0, "polynomial": 1, "gaussian": 2}


def linear_kernel(x_, x):
    r"""
    Linear kernel function.

    :param x_: (D) support vectors
    :param x: (D) input data
    :return: (1) kernel values
    """

    return np.dot(x_, x.T)


def polynomial_kernel(x_, x, p=2):
    r"""
    Polynomial kernel function.

    :param x_: (D) support vectors
    :param x: (D) input data
    :param p: (1) kernel degree
    :return: (1) kernel values
    """

    return (np.dot(x_, x.T) + 1) ** p


def rbf_kernel(x_, z, sigma=1):
    r"""
    Gaussian kernel function. (RBF)

    :param x_: (D) support vectors
    :param z: (D) input data
    :param sigma: (1) kernel width
    :return: (1) kernel values
    """

    return np.exp(-np.sum((x_ - z) ** 2, axis=1) / (2 * (sigma ** 2)))


def kernel_map(x_, z, kernel: int):
    if kernel == 0:
        return linear_kernel(x_, z)
    elif kernel == 1:
        return polynomial_kernel(x_, z)
    elif kernel == 2:
        return rbf_kernel(x_, z)
    else:
        raise ValueError(f"Kernel {kernel} not supported")


class KernelBase:
    def __init__(self, kernel: str):
        self.kernel_name = kernel
        self.kernel_id = KERNELS[kernel]

    def __call__(self, x, z):
        pass


class LinearKernel(KernelBase):
    r"""
    Linear kernel function.

    :return: (1) kernel values
    """

    def __init__(self):
        super().__init__("linear")

    def __call__(self, x, z):
        # np.einsum("n...,m...->nm...", x, z)
        if x.ndim == 1:
            return np.dot(x, z.T)
        else:
            return np.einsum("nd,md->nm", x, z)
        # return np.dot(x, z.T)


class PolynomialKernel(KernelBase):
    r"""
    Polynomial kernel function.

    :param p: (1) kernel degree
    :return: (1) kernel values
    """

    def __init__(self, p=2):
        super().__init__("polynomial")
        self.p = p

    def __call__(self, x, z):
        return (np.dot(x, z.T) + 1) ** self.p


class RBFKernel(KernelBase):
    r"""
    Gaussian kernel function. (RBF)

    :param sigma: (1) kernel width
    :return: (1) kernel values
    """

    def __init__(self, sigma=1):
        super().__init__("gaussian")
        self.sigma = sigma

    def __call__(self, x, z):
        # return np.exp(-np.sum((x - z) ** 2, axis=1) / (2 * (self.sigma ** 2)))
        # z_e = np.expand_dims(z, axis=1)

        # [np.sum(np.sum((x - z) * (x - z),axis=-1),axis=-1),
        #  np.sum((x - z[0]) * (x - z[0])),
        #  np.sum((x - z[1]) * (x - z[1])),
        #  np.sum((x - z[2]) * (x - z[2]))]

        # [np.exp(np.sum(np.sum((x - z) * (x - z), axis=-1), axis=-1) / (-1 * self.sigma ** 2)),
        #  np.exp(np.sum((x - z[0]) * (x - z[0])) / (-1 * self.sigma ** 2)),
        #  np.exp(np.sum((x - z[1]) * (x - z[1])) / (-1 * self.sigma ** 2)),
        #  np.exp(np.sum((x - z[2]) * (x - z[2])) / (-1 * self.sigma ** 2))]

        if z.ndim != 1:
            return np.exp(np.sum((x - z) * (x - z), axis=-1) / (-1 * self.sigma ** 2))
        else:
            return np.exp(np.sum((x - z) * (x - z)) / (-1 * self.sigma ** 2))
