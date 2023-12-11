from functools import partialmethod
from .utils import import_cupy_else_numpy
import numpy as np


class Tensor:
  def __init__(self, data, grad=None, device=None):
    self.grad = grad
    self.device = device
    if device == "cuda":
      global np
      np = import_cupy_else_numpy()
    self.data = np.array(data)
    self._ctx = None
  def __repr__(self):
    return f"Tensor({self.data})"

  def __str__(self):
    return f"Tensor {self.data} with grad {self.grad} on device {self.device}"
  
  def to(self, device):
    return Tensor(self.data, device=device)

  def backward(self, allow_fill=True):
    if self._ctx is None:
      return

    if self.grad is None and allow_fill:
      # fill in the first grad with one
      assert self.data.size == 1
      self.grad = np.ones_like(self.data)

    assert(self.grad is not None)

    grads = self._ctx.backward(self._ctx, self.grad)
    if len(self._ctx.operands) == 1:
      grads = [grads]
    for t,g in zip(self._ctx.operands, grads):
      # if g.shape != t.data.shape:
        # print(f"grad shape must match tensor shape in {self._ctx}, {g.shape} != {t.data.shape}")
        # assert(False)
      t.grad = g
      t.backward(False)

  def mean(self):
    div = Tensor(np.array([1/self.data.size]))
    return self.sum().mul(div)


class Operator:
  """
  Operator class is an abstract class which provides functionality for storing context of each operation for backward propogation
  """
  def __init__(self, *tensors) -> None:
    self.operands = tensors
    self.saved_tensors = []
  
  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  def apply(self, arg, *x):
    ctx = arg(self, *x)
    ret = Tensor(arg.forward(ctx, self.data, *[t.data if isinstance(t, Tensor) else t for t in x]))
    ret._ctx = ctx
    return ret

def register(name, fxn):
  """
  Adds a method to the Tensor class. The new method is a partial application of the `apply` method of the `fxn` object.

  :param name: The name of the new method.
  :type name: str
  :param fxn: The object whose `apply` method will be partially applied.
  :type fxn: object

  Example
  -------
  If `fxn` is an instance of a class `Mul` with an `apply` method that multiplies its arguments:
  
  .. code-block:: python

    register('mul', Mul())
    t = Tensor([2, 3, 4])
    t.mul(5)  # This will output a tensor with data [10, 15, 20]
  """
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))

class Mul(Operator):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x*y

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return y*grad_output, x*grad_output
register('mul', Mul)

class Add(Operator):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x+y

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors

    return grad_output, grad_output
register('add', Add)
    
class ReLU(Operator):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.maximum(input, 0)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.copy()
    grad_input[input < 0] = 0
    return grad_input
register('relu', ReLU)


class Dot(Operator):
  @staticmethod
  def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)
    return input.dot(weight)

  @staticmethod
  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = grad_output.dot(weight.T)
    grad_weight = grad_output.T.dot(input).T
    return grad_input, grad_weight
register('dot', Dot)

class Sum(Operator):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.array([input.sum()])

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return grad_output * np.ones_like(input)
register('sum', Sum)


class LogSoftmax(Operator):
  @staticmethod
  def forward(ctx, input):
    def logsumexp(x):
      c = x.max(axis=1)
      return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
    output = input - logsumexp(input).reshape((-1, 1))
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.saved_tensors
    return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
register('logsoftmax', LogSoftmax)

class MSE(Operator):
  @staticmethod
  def forward(ctx, input, target):
    ctx.save_for_backward(input, target)
    return np.array((input - target)**2)

  @staticmethod
  def backward(ctx, grad_output):
    input, target = ctx.saved_tensors
    return 2*(input - target), -2*(input - target)
register('mse', MSE)

class Conv2d(Operator):
  @staticmethod
  def forward(ctx, input, weight, padding, stride):
    ctx.save_for_backward(input, weight, padding, stride)

    N, C, W, H = input.shape
    F, C, WW, HH = weight.shape

    assert (H + 2 * padding - HH) % stride == 0
    assert (W + 2 * padding - WW) % stride == 0

    out_h = (H + 2 * padding - HH) // stride + 1
    out_w = (W + 2 * padding - WW) // stride + 1

    input_pad = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    output = np.zeros((N, F, out_h, out_w))

    for n in range(N):
      for f in range(F):
        for i in range(out_h):
          for j in range(out_w):
            output[n, f, i, j] = np.sum(input_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] * weight[f, :, :, :])
            
    return output

  @staticmethod
  def backward(ctx, grad_output):
    input, weight, padding, stride = ctx.saved_tensors
    
    N, C, H, W = input.shape 
    F, C, HH, WW = weight.shape
    
    out_h = (H + 2 * padding - HH) // stride + 1
    out_w = (W + 2 * padding - WW) // stride + 1

    grad_weight = np.zeros_like(weight)

    # Pad input tensor
    input_pad = np.pad(input, ((0,0), (0,0), (padding, padding), (padding, padding)), 'constant')
    
    for n in range(N):
      for f in range(F):
        for i in range(out_h):
          for j in range(out_w):
            
            # Get padded input patch
            input_patch = input_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            
            # Accumulate gradient
            grad_weight[f] += input_patch * grad_output[n, f, i, j]

    return None, grad_weight
register('conv2d', Conv2d)