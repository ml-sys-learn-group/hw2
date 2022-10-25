"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for w in self.params:
            w_u = (self.momentum * self.u.get(w, 0) + (1 - self.momentum) * (w.grad + self.weight_decay*w.data)).data
            self.u[w] = w_u
            w.data = w.data + (-self.lr) * w_u
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # TODO: optimize tensor num
        self.t += 1
        for w in self.params:
            grad = w.grad.data
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay*w.data
            w_m = (self.beta1 * self.m.get(w, 0) + (1 - self.beta1) * grad).data
            self.m[w] = w_m
            w_v = (self.beta2 * self.v.get(w, 0) + (1 - self.beta2) * (grad ** 2)).data
            self.v[w] = w_v
            
            # add bias correction
            cor_w_m = w_m/(1-self.beta1**self.t)
            cor_w_v = w_v/(1-self.beta2**self.t)
            
            # update weight
            w.data = w.data + (-self.lr)* (cor_w_m/(cor_w_v**0.5 + self.eps)).data
            
            
        ### END YOUR SOLUTION
