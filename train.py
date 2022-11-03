import model
import kernels
import mertics
import jax
import torch

from jax import numpy as jnp
from jax import grad, jit, vmap

key = jax.random.PRNGKey(0)

if __name__ == '__main__':
    # x = jnp.array([[3, 3], [4, 3], [1, 1]])
    # y = jnp.array([1, 1, -1])
    x = 2*(jnp.linspace(0, 10, 100))+2
    x.at[50:].set(x[50:]+3)
    y = jnp.concatenate([jnp.ones(50), -jnp.ones(50)])
    kernel = kernels.KERNELS['linear']
    c = 1
    tol = 1e-3
    max_passes = 10

    alpha = jax.random.uniform(key, (100,), minval=-1, maxval=1)
    b = model.calculate_b(alpha, x, y, kernel, c)
    print(f'b:{b}')

    svm_predict = jit(vmap(
        model.svm_predict, in_axes=(0, None, None, None, None, None)),
        static_argnames=['kernel'])
    res = svm_predict(x, kernel, x, y, alpha, b)
    # print(f'res:{res}')

    acc = mertics.accuracy(y, res)
    print(f'acc:{acc}')

    # smo = jit(model.smo, static_argnames=['kernel', 'c', 'tol', 'max_passes'])

    trained_alpha, trained_b = model.smo(x, y, kernel, c, tol, max_passes)
    trained_res = svm_predict(x, kernel, x, y, trained_alpha, trained_b)
    trained_acc = mertics.accuracy(y, trained_res)
    print(f'trained_acc:{trained_acc}')