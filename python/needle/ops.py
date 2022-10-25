"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from .autograd import make_list
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_node = node.inputs[0]
        return self.scalar * power_scalar(input_node, self.scalar - 1) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad = out_grad / rhs
        rgrad = out_grad * (-lhs) * power_scalar(rhs, -2)
        return lgrad, rgrad
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        result = a / self.scalar
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dim_size = len(a.shape)
        trans_axies = self.build_trans_axis(dim_size, self.axes)
        return array_api.transpose(array_api.array(a), axes=trans_axies)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION

    def build_trans_axis(self, dim_size, axes):
        """
        Args:
            dim_size:
            axes:
        Returns:
        """
        trans_axies = list(range(dim_size))
        if axes is None:
            trans_axies[-1] = dim_size-2
            trans_axies[-2] = dim_size-1
        else:
            trans_axies[axes[0]] = axes[1]
            trans_axies[axes[1]] = axes[0]
        return trans_axies


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        return reshape(out_grad, input_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class ExpandDims(TensorOp):
    """
    expand dims
    Args:
        TensorOp (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, axes):
        super().__init__()
        self.axes = axes
    
    def compute(self, a):
        return array_api.expand_dims(a, axis=self.axes)
    
    def gradient(self, out_grad, node):
        input_shape = node.input[0].shape
        return reshape(out_grad, input_shape)

def expand_dims(a, axes):
    """
    Args:
        a (_type_): _description_
        axes (_type_): _description_

    Returns:
        _type_: _description_
    """
    return ExpandDims(axes=axes)(a)



class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        sum_axis = self.get_sum_axes(input, out_grad)
        input_shape = input.shape
        return reshape(summation(out_grad, axes=sum_axis), input_shape)
        ### END YOUR SOLUTION
    
    def get_sum_axes(self, input_tensor, out_grad_tensor):
        """
        Args:
            input_tensor:
            out_grad_tensor:
        Returns:
        """
        all_axis = range(len(out_grad_tensor.shape))
        input_dim = len(input_tensor.shape)
        out_dim = len(out_grad_tensor.shape)

        rev_out_shape = list(out_grad_tensor.shape)
        rev_input_shape = list(input_tensor.shape)

        rev_out_shape.reverse()
        rev_input_shape.reverse()

        sum_axes = []
        for index in all_axis:
            if index >= input_dim:
                sum_axes.append(out_dim - 1 - index)
            else:
                if rev_input_shape[index] == 1 and rev_out_shape[index] > 1:
                    sum_axes.append(out_dim - 1 - index)

        return tuple(sum_axes)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims = False):
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes, keepdims=self.keepdims)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        if self.keepdims:
            trans_grad = broadcast_to(out_grad, input.shape)
        else:
            trans_grad = broadcast_tensor(out_grad, input.shape, self.axes)
        return trans_grad
        ### END YOUR SOLUTION

def broadcast_tensor(input_tensor, target_shape, axes):
    """
    Args:
        broadcast tensor to suitable shape, if axes is None, will broadcast in original mode
        input_tensor (_type_): _description_
        target_shape (_type_): _description_
        axes (_type_): _description_
    """
    if axes is None:
        target_tensor = broadcast_to(input_tensor, target_shape)
        return target_tensor
    else:
        axes = make_list(axes)
        exp_input_tensor = expand_dims(input_tensor, axes=axes)
        target_tensor = broadcast_to(exp_input_tensor, target_shape)
        return target_tensor


def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims=keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad = matmul(out_grad, transpose(rhs))
        lgrad = self.sum_via_input_dim(lhs, lgrad)
        rgrad = matmul(transpose(lhs), out_grad)
        rgrad = self.sum_via_input_dim(rhs, rgrad)
        return lgrad, rgrad
        ### END YOUR SOLUTION

    def sum_via_input_dim(self, input_tensor, grad_tensor):
        """
        Args:
            input_tensor:
            grad_tensor:

        Returns:
        """
        input_dim = len(input_tensor.shape)
        grad_dim = len(grad_tensor.shape)
        if grad_dim == input_dim:
            return grad_tensor
        else:
            sum_axis = tuple(range(grad_dim - input_dim))
            return summation(grad_tensor, sum_axis)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_node = node.inputs[0]
        return out_grad / input_node
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_node = node.inputs[0]
        return out_grad * exp(input_node)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        mask = relu(input) / input
        grad = out_grad * mask
        return grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Max(TensorOp):
    """
    return max value of give array via given axes
    Args:
        TensorOp (_type_)
    """
    def __init__(self, axes: Optional[tuple] = None, keepdims = False):
        self.axes = axes
        self.keepdims = keepdims
    
    def compute(self, a):
        return array_api.max(a, axis=self.axes, keepdims=self.keepdims)
    
    def gradient(self, out_grad, node):
        raise NotImplementedError()


def max(a, axes=None, keepdims=False):
    """
    Args:
        a (_type_): _description_
        axes (_type_, optional): _description_. Defaults to None.
        keepdims (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    return Max(axes=axes, keepdims=keepdims)(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        broad_max = array_api.broadcast_to(max_z, Z.shape)
        delta_z = Z - broad_max
        z_exp = array_api.exp(delta_z)
        z_sum = array_api.sum(z_exp, axis=self.axes)
        log_sum_exp = array_api.log(z_sum) + array_api.squeeze(max_z)
        return log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_node = node.inputs[0]
        input_max = max(input_node, self.axes, keepdims=True)
        input_max_broad = broadcast_to(input_max, input_node.shape)
        input_delta = input_node - input_max_broad
        input_exp = exp(input_delta)
        sum_exp = summation(input_exp, axes=self.axes, keepdims=True)
        sum_exp_broad = broadcast_to(sum_exp, input_node.shape)
        broad_out_grad = broadcast_tensor(out_grad, input_node.shape, axes=self.axes)
        grad = input_exp/sum_exp_broad * broad_out_grad
        return grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
