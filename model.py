import jax
import torch

from typing import Literal
from jax import numpy as jnp
from jax.lax import while_loop, cond, fori_loop, switch
from jax import grad, jit, vmap
from kernels import kernel_map, linear_kernel, polynomial_kernel, rbf_kernel

key = jax.random.PRNGKey(0)


# kernel_map = {"linear": 0, "polynomial": 1, "gaussian": 2}
# kernel_list = [linear_kernel, polynomial_kernel, gaussian_kernel]

def svm_forward(x, kernel: int, x_, y_, alpha, b):
    r"""
    SVM prediction function.

    :param x: (D) input data
    :param kernel: (1) kernel function
    :param x_: (n, D) support vectors
    :param y_: (n) support vector labels
    :param alpha: (n) support vector weights
    :param b: (1) bias
    :return: (1) prediction labels
    """

    return jnp.sum(alpha * y_ * kernel_map(x_, x, kernel)) + b


def svm_predict(x, kernel: int, x_, y_, alpha, b):
    r"""
    SVM prediction function.

    :param x: (D) input data
    :param kernel: (1) kernel function
    :param x_: (n, D) support vectors
    :param y_: (n) support vector labels
    :param alpha: (n) support vector weights
    :param b: (1) bias
    :return: (1) prediction labels
    """

    return jnp.sign(svm_forward(x, kernel, x_, y_, alpha, b))


# @jit
def calculate_b(alpha, x_, y_, kernel: int, c):
    r"""
    Calculate bias from the best alpha.

    :param alpha: (n) support vector weights
    :param x_: (n, D) support vectors
    :param y_: (n) support vector labels
    :param kernel: (1) kernel function
    :param c: (1) regularization parameter
    :return: (1) bias
    """
    for idx, a in enumerate(alpha):
        if 0 < a < c:
            return y_[idx] - jnp.sum(alpha * y_ * kernel_map(x_, x_[idx], kernel), axis=0)
    else:
        raise ValueError("No support vectors found")


def hinge_loss(y, y_pred):
    r"""
    Hinge loss function.

    :param y: (1) labels
    :param y_pred: (1) predictions
    :return: (1) loss
    """
    return jnp.max(0, 1 - y * y_pred)


def smo(x_, y_, kernel, c, tol=1e-3, max_passes=5):
    r"""
    Sequential Minimal Optimization algorithm.

    :param x_: (n, D) support vectors
    :param y_: (n) support vector labels
    :param kernel: (1) kernel function
    :param c: (1) regularization parameter
    :param tol: (1) tolerance
    :param max_passes: (1) maximum number of passes
    :return: (n) alpha
    """
    n = x_.shape[0]
    alpha = jnp.zeros(n)
    b = 0.
    passes = 0

    def pass_it(*args):
        pass

    def print_eta_le_0(i, n, alpha, b, passes, num_changed_alphas):
        jax.debug.print('\033[1;031m WARNING  eta <= 0')
        return i, n, alpha, b, passes, num_changed_alphas

    def print_delta_too_small(delta):
        jax.debug.print('\033[1;031m WARNING   a_j update too small, delta = {}', delta)

    def while_loop_body(n, alpha, b, passes):
        num_changed_alphas = 0

        # for i in range(n):
        def for_loop_body(i, n, alpha, b, passes, num_changed_alphas):
            g_xi = svm_forward(x_[i], kernel, x_, y_, alpha, b)
            E_i = g_xi - y_[i]
            # if (y_[i] * E_i < -tol and alpha[i] < c) or \
            #         (y_[i] * E_i > tol and alpha[i] > 0):
            j = jax.random.randint(key, shape=(), minval=0, maxval=n)
            while_loop(lambda j: j == i, lambda j: jax.random.randint(key, shape=(), minval=0, maxval=n), j)
            # while j == i:
            #     j = jax.random.randint(key, shape=(), minval=0, maxval=n)

            # if j == i and j < n - 1:
            #     j += 1
            # else:
            #     j -= 1

            # g_xi = svm_forward(x_[i], kernel, x_, y_, alpha, b)
            g_xj = svm_forward(x_[j], kernel, x_, y_, alpha, b)
            E_j = g_xj - y_[j]

            eta = kernel_map(x_[i], x_[i], kernel) + \
                  kernel_map(x_[j], x_[j], kernel) - \
                  2 * kernel_map(x_[i], x_[j], kernel)

            def if_brunch(i, n, alpha, b, passes, num_changed_alphas):
                a_i_old, a_j_old = alpha[i], alpha[j]

                L, H = cond(y_[i] == y_[j], lambda a_j_old, a_i_old, c: (
                    jnp.maximum(0, a_j_old + a_i_old - c), jnp.minimum(c, a_j_old + a_i_old)),
                            lambda a_j_old, a_i_old, c: (
                                jnp.maximum(0, a_j_old - a_i_old), jnp.minimum(c, c + a_j_old - a_i_old)), a_j_old,
                            a_i_old, c)

                # if y_[i] == y_[j]:
                #     L = jnp.maximum(0, a_j_old + a_i_old - c)
                #     H = jnp.minimum(c, a_j_old + a_i_old)
                # else:
                #     L = jnp.maximum(0, a_j_old - a_i_old)
                #     H = jnp.minimum(c, c + a_j_old - a_i_old)

                a_j_new = jnp.clip(a_j_old + y_[j] * (E_i - E_j) / eta, L, H)
                a_i_new = a_i_old + y_[i] * y_[j] * (a_j_old - a_j_new)

                delta = jnp.abs(a_j_new - a_j_old)
                # jax.debug.print('WARNING   a_j update too small, delta = {}', delta)
                cond(delta < 1e-5, lambda delta: print_delta_too_small(delta), pass_it, delta)
                # if jnp.abs(a_j_new - a_j_old) < tol:
                #     print(f'WARNING   a_j update too small, delta = {a_j_new - a_j_old}')
                #     continue

                # alpha[i], alpha[j] = a_i_new, a_j_new
                alpha = alpha.at[i].set(a_i_new)
                alpha = alpha.at[j].set(a_j_new)

                b_i = -E_i - y_[i] * kernel_map(x_[i], x_[i], kernel) * (a_i_new - a_i_old) - \
                      y_[j] * kernel_map(x_[i], x_[j], kernel) * (a_j_new - a_j_old) + b
                b_j = -E_j - y_[i] * kernel_map(x_[i], x_[j], kernel) * (a_i_new - a_i_old) - \
                      y_[j] * kernel_map(x_[j], x_[j], kernel) * (a_j_new - a_j_old) + b
                # if 0 < a_i_new < c:
                #     b = b_i
                # elif 0 < a_j_new < c:
                #     b = b_j
                # else:
                #     b = (b_i + b_j) / 2

                b = cond(jax.lax.bitwise_and(0 < a_i_new, a_i_new < c), lambda a_j_new, b_i, b_j, c: b_i,
                         lambda a_j_new, b_i, b_j, c: cond(jax.lax.bitwise_and(0 < a_j_new, a_j_new < c),
                                                           lambda a_j_new, b_i, b_j, c: b_j,
                                                           lambda a_j_new, b_i, b_j, c: (b_i + b_j) / 2, a_j_new, b_i,
                                                           b_j,
                                                           c), a_j_new, b_i, b_j, c)

                num_changed_alphas += 1
                jax.debug.print('\033[0m INFO   iteration:{}  i:{}  pair_changed:{}', passes, i,
                                num_changed_alphas)
                return i, n, alpha, b, passes, num_changed_alphas

            cond(eta <= 0, print_eta_le_0, if_brunch, i, n, alpha, b, passes, num_changed_alphas)

            # if eta <= 0:
            #     print('WARNING  eta <= 0')
            #     continue
            # if num_changed_alphas == 0:
            #     passes += 1
            # else:
            #     passes = 0

        _, alpha, b, passes, num_changed_alphas = fori_loop(0, n, lambda i, *a: for_loop_body(i, n, alpha, b, passes,
                                                                                              num_changed_alphas),
                                                            (n, alpha, b, passes, num_changed_alphas))

        passes += 1
        passes = cond(num_changed_alphas == 0, lambda passes: passes + 1, lambda passes: 0, passes)
        jax.debug.print('\033[0m passes number: {}', passes)

        # else:
        #     continue

        return [n, alpha, b, passes]

    _, alpha, b, _ = while_loop(lambda *a: passes < max_passes,
                                lambda *a: while_loop_body(n, alpha, b, passes), [n, alpha, b, passes])

    return alpha, b


if __name__ == '__main__':
    print(jax.devices())

    # a = jnp.array([1, 2, 3])
    # b = jnp.array([4, 5, 6])
    # print(a * b)
