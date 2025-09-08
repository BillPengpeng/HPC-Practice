"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

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


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b);
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #a, b = node.inputs[0].cached_data, node.inputs[1].cached_data
        #out_grad_a = b * array_api.power(a, b - 1) * out_grad.cached_data
        #out_grad_b = array_api.power(a, b) * array_api.log(a) * out_grad.cached_data
        #return Tensor(out_grad_a), Tensor(out_grad_b);
        a, b = node.inputs
        out_grad_a = b * power(a, b - 1) * out_grad
        out_grad_b = power(a, b) * log(a) * out_grad
        return out_grad_a, out_grad_b
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar);
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # a = node.inputs[0].cached_data
        # out_grad_a = self.scalar * array_api.power(a, self.scalar - 1) * out_grad.cached_data
        # return Tensor(out_grad_a)
        a = node.inputs[0]
        return self.scalar * power_scalar(a, self.scalar - 1) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b);
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # a, b = node.inputs[0].cached_data, node.inputs[1].cached_data
        # out_grad_a = 1 / b * out_grad.cached_data
        # out_grad_b = -1 * array_api.divide(a, b * b) * out_grad.cached_data
        # return Tensor(out_grad_a), Tensor(out_grad_b);
        a, b = node.inputs
        out_grad_a = divide(out_grad, b)
        out_grad_b = -1 * divide(a, b * b) * out_grad
        return out_grad_a, out_grad_b;
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar);
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # a = node.inputs[0].cached_data
        # out_grad_a = 1 / self.scalar * out_grad.cached_data
        # return Tensor(out_grad_a);
        return divide_scalar(out_grad, self.scalar);
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # print("self.axes:", self.axes, a, a.ndim);
        if self.axes is None:
          return array_api.swapaxes(a, a.ndim-2, a.ndim-1);
        return array_api.swapaxes(a, self.axes[-2], self.axes[-1]);
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # out_grad_np = out_grad.cached_data
        # if self.axes is None:
        #   return array_api.swapaxes(out_grad_np, out_grad_np.ndim-2, out_grad_np.ndim-1);
        # out_grad_a = array_api.swapaxes(out_grad_np, self.axes[-2], self.axes[-1]);
        # return Tensor(out_grad_a);
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape);
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # a = node.inputs[0].cached_data
        # dst_shape = a.shape
        # out_grad_a = array_api.reshape(out_grad.cached_data, dst_shape);
        # return Tensor(out_grad_a);
        return reshape(out_grad, node.inputs[0].shape);
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape);
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ori_shape = node.inputs[0].shape
        input_shape = list(reversed(list(ori_shape)))
        self_shape = list(reversed(list(self.shape)))
        broad_dims = []
        for i, d in enumerate(self_shape):
          if i >= len(input_shape) or input_shape[i] != d:
            broad_dims.append(len(self_shape) - i - 1)
        return out_grad.sum(axes=tuple(broad_dims)).reshape(ori_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        # 20241129 add
        if isinstance(self.axes, int):
          self.axes = tuple([self.axes])

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes);
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # a = node.inputs[0].cached_data
        # expand_out_grad = array_api.expand_dims(out_grad.cached_data, self.axes)
        # out_grad_a = array_api.broadcast_to(expand_out_grad, a.shape);
        # return Tensor(out_grad_a);
        input_shape = list(node.inputs[0].shape)
        shrink_dims = range(len(input_shape)) if self.axes is None else self.axes
        # print("shrink_dims:", shrink_dims, self.axes)
        for i in shrink_dims:
          input_shape[i] = 1
        return out_grad.reshape(tuple(input_shape)).broadcast_to(node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b;
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # a = node.inputs[0].cached_data
        # b = node.inputs[1].cached_data
        # # print(a.shape, b.shape, array_api.swapaxes(a, a.ndim-2, a.ndim-1).shape, array_api.swapaxes(b, b.ndim-2, b.ndim-1).shape)
        # out_grad_a = out_grad.cached_data @ array_api.swapaxes(b, b.ndim-2, b.ndim-1)
        # out_grad_b = array_api.swapaxes(a, a.ndim-2, a.ndim-1) @ out_grad.cached_data
        # if out_grad_a.shape != a.shape:
        #   out_grad_a = array_api.sum(out_grad_a, axis=tuple(list(range(len(out_grad_a.shape) - len(a.shape)))))
        # if out_grad_b.shape != b.shape:
        #   out_grad_b = array_api.sum(out_grad_b, axis=tuple(list(range(len(out_grad_b.shape) - len(b.shape)))))
        # return Tensor(out_grad_a), Tensor(out_grad_b);
        a, b = node.inputs
        out_grad_a = matmul(out_grad, transpose(b))
        out_grad_b = matmul(transpose(a), out_grad)
        if out_grad_a.shape != a.shape:
          out_grad_a = summation(out_grad_a, axes=tuple(list(range(len(out_grad_a.shape) - len(a.shape)))))
        if out_grad_b.shape != b.shape:
          out_grad_b = summation(out_grad_b, axes=tuple(list(range(len(out_grad_b.shape) - len(b.shape)))))
        return out_grad_a, out_grad_b;
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a;
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad;
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a);
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # a = node.inputs[0].cached_data
        # out_grad_a = 1.0 / a * out_grad.cached_data; 
        # return Tensor(out_grad_a);
        return divide(out_grad, node.inputs[0]);
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a);
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # a = node.inputs[0].cached_data
        # out_grad_a = array_api.exp(a) * out_grad.cached_data; 
        # return Tensor(out_grad_a);
        return exp(node.inputs[0]) * out_grad;
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0);
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].cached_data
        out_grad_a = (a > 0) * out_grad.cached_data; 
        return Tensor(out_grad_a);
        #return (node.inputs[0] > 0) * out_grad;
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

