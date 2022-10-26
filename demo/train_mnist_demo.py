# -*- coding: utf-8 -*-
"""
train mnist demo
"""
import numpy as np
from mlp_resnet import train_mnist
import needle as ndl


def train_mnist_1(batch_size, epochs, optimizer, lr, weight_decay, hidden_dim):
    np.random.seed(1)
    out = train_mnist(batch_size, epochs, optimizer, lr, weight_decay, hidden_dim, data_dir="./data")
    return np.array(out)


if __name__ == '__main__':
    import os
    os.chdir("../")
    train_mnist_1(250, 1, ndl.optim.SGD, 0.001, 0.01, 100)
