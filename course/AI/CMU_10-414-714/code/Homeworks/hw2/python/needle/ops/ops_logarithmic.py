from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def __init__(self, dim: Optional[int] = None):
        if dim == None:
          self.dim = 1
        else:
          self.dim = dim
        
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = array_api.max(Z, axis=self.dim, keepdims=True)
        ez = array_api.exp(Z - maxz)
        sum_exp = array_api.sum(ez, axis=self.dim, keepdims=True)
        log_sum_exp = array_api.log(sum_exp)
        self.softmax = ez / sum_exp
        self.output = Z - maxz - log_sum_exp
        return self.output
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        log_softmax_output = node.numpy()
        softmax = array_api.exp(log_softmax_output)
        grad = out_grad - softmax * array_api.sum(out_grad.numpy(), axis=-1, keepdims=True)
        return grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        # 20241129 add
        #if isinstance(self.axes, int):
        #  self.axes = tuple([self.axes])

    def compute(self, Z):
        maxz = array_api.max(Z, axis=self.axes, keepdims=True)
        maxz_r = array_api.max(Z, axis=self.axes)
        exp_z = array_api.exp(Z - maxz)
        sum_z = array_api.sum(exp_z, axis=self.axes)
        ret_z = array_api.log(sum_z) + maxz_r
        return ret_z

    def gradient(self, out_grad, node):
        Z = node.inputs[0].cached_data
        maxz = array_api.max(Z, axis=self.axes, keepdims=True)
        exp_z = array_api.exp(Z - maxz)
        sum_z = array_api.sum(exp_z, axis=self.axes)
        out_grad = out_grad / sum_z
        input_shape = list(node.inputs[0].shape)
        shrink_dims = range(len(input_shape)) if self.axes is None else self.axes
        for i in shrink_dims:
          input_shape[i] = 1
        #print("input_shape:", input_shape)
        out_grad = out_grad.reshape(tuple(input_shape)).broadcast_to(node.inputs[0].shape)
        # return out_grad
        return out_grad * exp_z

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

