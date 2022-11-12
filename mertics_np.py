import numpy as np


def accuracy(y, gt):
    return np.mean(y == gt)
