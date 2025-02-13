"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
# import numpy as np
import numpy as array_api


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
        # print("24:", len(params), value.keys())
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        # print("29", len(params))
        return params
    else:
        # print("31")
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
        # print("self.__dict__:", self.__dict__.keys())
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
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        if bias:
          # fan_in需要等于out_features 
          self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=True).reshape((1, out_features)))
        else:
          self.bias = None

        # self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        # if bias == True:
        #   self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose())
        # else:
        #   self.bias = None
        # print("101:", dtype)

    def forward(self, X: Tensor) -> Tensor:
        # Y = X @ self.weight
        # if self.bias is not None:
        #   Y = Y + self.bias
        # return Y

        if X.shape[-1] != self.in_features:
          print(X.shape, self.in_features)
          raise ValueError("Input tensor's last dimension size does not match the linear layer's input features")
        res = X.matmul(self.weight)
        if self.bias != None:
          res += self.bias.broadcast_to(res.shape)
        return res


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        Y = x
        for module in self.modules:
          Y = module(Y)
        return Y


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size = logits.shape[0]
        label_size = logits.shape[1]
        y_one_hot = init.one_hot(label_size, y)
        sum_val = ops.log(ops.summation(ops.exp(logits), axes=(1))) - \
              ops.summation(logits * y_one_hot, axes=(1))
        ret = ops.summation(sum_val / batch_size);
        return ret

        # y_oh = init.one_hot(logits.shape[1], y)
        # ret = (ops.summation(ops.logsumexp(logits, (1,)) / logits.shape[0]) - 
        #      ops.summation(y_oh * logits / logits.shape[0]))
        # return ret

        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

        # self.weight = Parameter(init.ones((dim), requires_grad=True, dtype=dtype, device=device))
        # self.bias = Parameter(init.zeros((dim), requires_grad=True, dtype=dtype, device=device))
        # self.running_mean = init.zeros((dim), dtype=dtype, device=device)
        # self.running_var = init.ones((dim), dtype=dtype, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        weight_broad = ops.broadcast_to(self.weight, x.shape)
        bias_broad = ops.broadcast_to(self.bias, x.shape)
        if self.training is True:
          mean_x = ops.reshape(ops.summation(x, axes = 0) / batch_size, (1, self.dim))
          mean_x_broad = ops.broadcast_to(mean_x, x.shape)
          sub_x_broad = x - mean_x_broad
          var_x = ops.reshape(ops.summation(ops.multiply(sub_x_broad, sub_x_broad), axes = 0) / batch_size, (1, self.dim))
          var_x_broad = ops.broadcast_to(var_x, x.shape)
          div_x = (sub_x_broad / ops.power_scalar(var_x_broad + self.eps, 0.5))
          Y = ops.multiply(div_x, weight_broad) + bias_broad
          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * ops.reshape(mean_x, (self.dim))
          self.running_var = (1 - self.momentum) * self.running_var + self.momentum * ops.reshape(var_x, (self.dim))
        else:
          mean_x_broad = ops.broadcast_to(self.running_mean, x.shape)
          var_x_broad = ops.broadcast_to(self.running_var, x.shape)
          sub_x_broad = x - mean_x_broad
          div_x = (sub_x_broad / ops.power_scalar(var_x_broad + self.eps, 0.5))
          Y = ops.multiply(div_x, weight_broad) + bias_broad
        return Y

        # if self.training:
        #     mean = (x.sum((0,)) / x.shape[0]) #(dim)
        #     var = (((x - mean.broadcast_to(x.shape))**2).sum((0,)) / x.shape[0]) #(dim)
        #     self.running_mean = self.running_mean * (1 - self.momentum) + mean * self.momentum
        #     self.running_var = self.running_var * (1 - self.momentum) + var * self.momentum
        #     x_norm = (x - mean.broadcast_to(x.shape)) / ((
        #         var.broadcast_to(x.shape) + self.eps)**0.5)
        # else:
        #     x_norm = (x - self.running_mean.broadcast_to(x.shape)) / ((
        #         self.running_var.broadcast_to(x.shape) + self.eps)**0.5)
        # return self.weight.broadcast_to(x.shape) * x_norm + self.bias.broadcast_to(x.shape)
        # raise NotImplementedError()
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype, requires_grad=True))
        # self.weight = Parameter(init.ones(dim, requires_grad=True, device=device, dtype=dtype))
        # self.bias = Parameter(init.zeros(dim, requires_grad=True, device=device, dtype=dtype))
        # print("199:", dtype, self.weight.data.dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # X = x.cached_data
        # mean_x = array_api.mean(X, axis = 1).reshape(-1, 1)
        # var_x = array_api.var(X, axis = 1).reshape(-1, 1)
        # # print(X.shape, mean_x.shape, var_x.shape, X, mean_x)
        # Y = self.weight * ((X - mean_x) / array_api.sqrt(var_x + self.eps)) + self.bias;
        # return Tensor(Y);
  
        batch_size = x.shape[0]
        mean_x = ops.broadcast_to(ops.reshape(ops.summation(x, axes = 1) / self.dim, (batch_size, 1)), x.shape)
        sub_x = x - mean_x
        var_x = ops.broadcast_to(ops.reshape(ops.summation(ops.multiply(sub_x, sub_x), axes = 1) / self.dim, (batch_size, 1)), x.shape)
        weight_broad = ops.broadcast_to(self.weight, x.shape)
        bias_broad = ops.broadcast_to(self.bias, x.shape)
        div_x = (sub_x / ops.power_scalar(var_x + self.eps, 0.5))
        Y = ops.multiply(div_x, weight_broad) + bias_broad
        return Y

        # mean = (x.sum((1,)) / self.dim).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        # var = (((x - mean)**2).sum((1,)) / self.dim).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        # deno = (var + self.eps)**0.5
        # return self.weight.broadcast_to(x.shape) * (x - mean) / deno + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.training is True:
          mask = init.randb(*x.shape, p=1-self.p, device=x.device)
          y = x / (1 - self.p) * mask
        else:
          y = x;
        return y
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return x + self.fn(x)
        ### END YOUR SOLUTION
