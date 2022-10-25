"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from functools import reduce
import needle as ndl


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []

class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            fan_in=in_features, fan_out=out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(ops.transpose(init.kaiming_uniform(
                fan_in=out_features, fan_out=1, device=device, dtype=dtype), (0 , 1)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = ops.matmul(X, self.weight)
        if self.bias:
            return output + ops.broadcast_to(self.bias, output.shape)
        else:
            return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        flaten_size = reduce(lambda x,y: x*y, X.shape[1:])
        return ops.reshape(X, shape=(X.shape[0], flaten_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = x
        for module in self.modules:
            output = module.forward(output)
        return output
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        class_num = logits.shape[-1]
        y_onehot = init.one_hot(class_num, y)
        logit_y = ops.summation(logits * y_onehot, axes=1)
        return ops.summation((ops.logsumexp(logits, axes=1) - logit_y)/logits.shape[0])
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            batch_sz = x.shape[0]
            ex = ops.summation(x, axes=0, keepdims=True) / batch_sz
            broad_ex = ops.broadcast_to(ex, x.shape)    
            var_x_raw = ops.summation((x - broad_ex)**2, axes=0, keepdims=True) / batch_sz
        else:
            ex = ops.reshape(self.running_mean, shape=(1, self.dim))
            broad_ex = ops.broadcast_to(ex, x.shape)    
            var_x_raw = ops.reshape(self.running_var, shape=(1, self.dim))
        
        var_x = (var_x_raw + self.eps) ** 0.5
        broad_var = ops.broadcast_to(var_x, x.shape)
        broad_w = ops.broadcast_to(self.weight, x.shape)
        broad_b = ops.broadcast_to(self.bias, x.shape)
        norm_x = broad_w  * (x - broad_ex)/broad_var + broad_b

        if self.training:
            batch_m = ops.reshape(ex, shape=self.dim)
            batch_var = ops.reshape(var_x_raw, shape=self.dim)
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * batch_m
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * batch_var

        return norm_x
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.w = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.b = Parameter(init.zeros(1, dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # mean of x, shape: (batch_sz,)
        ex = ops.summation(x, axes=1, keepdims=True) / self.dim
        broad_ex = ops.broadcast_to(ex, x.shape)
        # covariance
        var_x_raw = ops.summation((x - broad_ex)**2, axes=1, keepdims=True) / self.dim
        var_x = (var_x_raw + self.eps) ** 0.5
        broad_var = ops.broadcast_to(var_x, x.shape)
        broad_w = ops.broadcast_to(self.w, x.shape)
        broad_b = ops.broadcast_to(self.b, x.shape)
        norm_x = broad_w  * (x - broad_ex)/broad_var + broad_b
        return norm_x
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            p = 1.0 - self.p
            mask = init.randb(*x.shape, p=p)
            return x * mask / p
        else:
            # do not dropout anything when training flag is False
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



