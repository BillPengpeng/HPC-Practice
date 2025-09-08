"""Optimization module"""
import needle as ndl
# import numpy as np
import numpy as array_api


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
        self.params_size = len(self.params)

    def step(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        for key in range(self.params_size):
          p = self.params[key]
          if self.weight_decay > 0:
            grad = p.grad.data + self.weight_decay * p.data
          else:
            grad = p.grad.data
          if key in self.u.keys():
            self.u[key] = self.momentum * self.u[key] + (1 - self.momentum) * grad
          else:
            self.u[key] = (1 - self.momentum) * grad
          p.data = p.data - self.lr * self.u[key]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
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
        self.params_size = len(self.params)
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for key in range(self.params_size):
          p = self.params[key]
          if self.weight_decay > 0:
            grad = p.grad.data + self.weight_decay * p.data
          else:
            grad = p.grad.data
          if key in self.m.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
          else:
            self.m[key] = (1 - self.beta1) * grad
          if key in self.v.keys():
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad * grad
          else:
            self.v[key] = (1 - self.beta2) * grad * grad
          m_bias_corr = self.m[key] / (1 - self.beta1**self.t)
          v_bias_corr = self.v[key] / (1 - self.beta2**self.t)
          p.data = p.data - self.lr * m_bias_corr / (v_bias_corr**0.5 + self.eps)
        ### END YOUR SOLUTION
