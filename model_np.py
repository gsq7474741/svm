from typing import Literal

import numpy as np

from kernels_np import KernelBase


# kernel_map = {"linear": 0, "polynomial": 1, "gaussian": 2}
# kernel_list = [linear_kernel, polynomial_kernel, gaussian_kernel]

def svm_forward(x, kernel: KernelBase, x_, y_, alpha, b):
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

    return np.sum(alpha * y_ * kernel(x_, x), axis=0) + b


def svm_predict(x, kernel: KernelBase, x_, y_, alpha, b):
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
    np.sign(np.einsum('nm->m', np.einsum('n,nm->nm', alpha, np.einsum('n,nm->nm', y_, np.einsum('md,nd->nm', x, x_)))))
    return np.sign(np.sum(alpha * y_ * kernel(x, x_), axis=0) + b)


# @jit
def calculate_b(alpha, x_, y_, kernel: KernelBase, c):
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
            return y_[idx] - np.sum(alpha * y_ * kernel(x_, x_[idx]), axis=0)
    else:
        raise ValueError("No support vectors found")


def hinge_loss(y, y_pred):
    r"""
    Hinge loss function.

    :param y: (1) labels
    :param y_pred: (1) predictions
    :return: (1) loss
    """
    return np.max(0, 1 - y * y_pred)


class SMO:
    def __init__(self, x: np.ndarray, y: np.ndarray, c: float, tol: float, max_passes: int, kernel: KernelBase):
        """
        构造函数
        :param x: 特征向量，二维数组
        :param y: 数据标签，一维数组
        :param c: 对松弛变量的容忍程度，越大越不容忍
        :param tol: 完成一次迭代误差要求
        :param max_passes: 迭代次数
        """
        self.SVLabel = None
        self.SVAlpha = None
        self.w = None
        self.x = x
        self.y = y.T
        self.c = c
        self.tol = tol
        self.max_passes = max_passes
        self.kernel = kernel
        self.eps = 1e-4
        self.n = self.x.shape[0]
        self.d = self.x.shape[1]

        self.alpha = np.zeros(self.n)  # 初始化alpha
        self.b = 0.0
        self.error_cache = np.array(np.zeros((self.n, 2)))  # 错误缓存

        self.SV = ()  # 最后保留的支持向量
        self.SVIndex = None  # 支持向量的索引

        self.K = np.zeros((self.n, self.n))  # 先求内积，

    def calc_E(self, k):
        """
        计算第k个数据的误差
        :param k:
        :return:
        """
        # 因为这是训练阶段，用数据集的数据，所以可以直接这样做，这里先把内积全部求了，

        return np.dot(self.alpha * self.y, self.K[:, k]) + self.b - self.y[k]

    def update_Ek(self, k):
        self.error_cache[k] = [1, (self.calc_E(k))]

    def select_j_rand(self, i, m):
        j = i
        while j == i:
            j = np.random.randint(0, m)
        return j

    def clip_alpha(self, a_j, L, H):
        if a_j > H:
            a_j = H
        if L > a_j:
            a_j = L
        return a_j

    def select_j(self, i, Ei):
        """
        在确定了 i 的前提下，按照最大步长取另一个 j
        :param i:
        :param Ei:
        :return:
        """
        max_E = 0.0
        j = 0
        E_j = 0.0
        valid_e_cache_list = np.nonzero(self.error_cache[:, 0])[0]
        if len(valid_e_cache_list) > 1:
            for k in valid_e_cache_list:
                if k == i:
                    continue
                E_k = self.calc_E(k)
                delta_E = abs(Ei - E_k)
                if delta_E > max_E:
                    j = k
                    max_E = delta_E
                    E_j = E_k
            return j, E_j
        else:
            j = self.select_j_rand(i, self.n)
            E_j = self.calc_E(j)
            return j, E_j

    def update_alpha(self, i: int) -> Literal[0, 1]:
        """
        选择 i 之后，更新参数
        """
        E_i = self.calc_E(i)

        if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.c) or \
                (self.y[i] * E_i > self.tol and self.alpha[i] > 0):

            self.update_Ek(i)

            j, E_j = self.select_j(i, E_i)

            alpha_i_old = self.alpha[i].copy()
            alpha_j_old = self.alpha[j].copy()

            if self.y[i] != self.y[j]:
                L = np.maximum(0, self.alpha[j] - self.alpha[i])
                H = np.minimum(self.c, self.c + self.alpha[j] - self.alpha[i])
            else:
                L = np.maximum(0, self.alpha[j] + self.alpha[i] - self.c)
                H = np.minimum(self.c, self.alpha[i] + self.alpha[j])
            if L == H:
                return 0

            eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if eta >= 0:
                return 0

            self.alpha[j] -= self.y[j] * (E_i - E_j) / eta
            self.alpha[j] = self.clip_alpha(self.alpha[j], L, H)
            self.update_Ek(j)

            if np.abs(alpha_j_old - self.alpha[j]) < self.eps:
                # 目标迭代完成，不需要再迭代，而且所有参数已经保留，理论上是不应该保留当前的参数的，但是也正因为反正相差不多，所以可以
                return 0

            self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])
            self.update_Ek(i)

            b1 = self.b - E_i - self.y[i] * self.K[i, i] * (self.alpha[i] - alpha_i_old) - \
                 self.y[j] * self.K[i, j] * (self.alpha[j] - alpha_j_old)
            b2 = self.b - E_j - self.y[i] * self.K[i, j] * (self.alpha[i] - alpha_i_old) - \
                 self.y[j] * self.K[j, j] * (self.alpha[j] - alpha_j_old)

            if 0 < self.alpha[i] < self.c:
                self.b = b1
            elif 0 < self.alpha[j] < self.c:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.
            return 1
        else:
            return 0

    def fit(self):
        """
        训练模型，找到 alpha 的近似最优解
        :return:
        """
        passes = 0
        entry_set = True
        num_changed_alpha_pair = 0
        for i in range(self.n):
            for j in range(self.n):
                self.K[i, j] = self.kernel(self.x[i, :], self.x[j, :])

        while passes < self.max_passes and ((num_changed_alpha_pair > 0) or entry_set):
            num_changed_alpha_pair = 0
            if entry_set:
                for i in range(self.n):
                    num_changed_alpha_pair += self.update_alpha(i)
                passes += 1
            else:
                non_bounds = np.nonzero((self.alpha > 0) * (self.alpha < self.c))[0]
                for i in non_bounds:
                    num_changed_alpha_pair += self.update_alpha(i)
                passes += 1
            if entry_set:
                entry_set = False
            elif num_changed_alpha_pair == 0:
                entry_set = True
        # 保存模型参数
        self.SVIndex = np.nonzero(self.alpha)[0]
        self.SV = self.x[self.SVIndex]
        self.SVAlpha = self.alpha[self.SVIndex]
        self.SVLabel = self.y[self.SVIndex]

        # 清空中间变量
        # self.x = None
        # self.K = None
        # self.y = None
        # self.alpha = None
        # self.error_cache = None

    def calcw(self):
        """
        计算w，结果是一个数组，长度是特征向量的长度
        :return:
        """
        for i in range(self.n):
            self.w += np.dot(self.alpha[i] * self.y[i], self.x[i, :])
        # w_np = np.sum(np.dot(self.alpha * self.y, self.x), axis=-1)
        # d = 3

    def predict(self, input_x):
        """
        输入待预测的数据，输出结果的标签
        :param input_x: 待预测数据，要是数组形式，二维数组
        :return: 一个列表，包含结果
        """
        # forward_value = np.sum(self.SVAlpha * self.SVLabel * self.kernel(self.SV, np.expand_dims(input_x, axis=1)),
        #                        axis=-1) + self.b
        # forward_value = np.where(np.abs(forward_value) < self.eps, np.random.uniform(-1, 1), forward_value)
        forward_value = self.forward(input_x)
        result_np = np.where(forward_value > 0, 1, -1)
        return result_np

    def forward(self, input_x):
        """
        输入待预测的数据，输出结果
        :param input_x: 待预测数据，要是数组形式，二维数组
        :return: 一个列表，包含结果
        """
        forward_value = np.sum(self.SVAlpha * self.SVLabel * self.kernel(self.SV, np.expand_dims(input_x, axis=1)),
                               axis=-1) + self.b
        # tmp_np = np.where(np.abs(tmp_np) < self.eps, np.random.uniform(-1, 1), tmp_np)
        # result_np = np.where(tmp_np > 0, 1, -1)
        return forward_value


if __name__ == '__main__':
    pass
    # a = jnp.array([1, 2, 3])
    # b = jnp.array([4, 5, 6])
    # print(a * b)
