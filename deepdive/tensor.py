from functools import partialmethod
from .utils import import_cupy_else_numpy
import numpy as np


class Tensor:
  """
  A class used to represent a Tensor.

  This class provides methods for tensor operations and also keeps track of gradients for backpropagation.

  Attributes
  ----------
  data : numpy.ndarray or cupy.ndarray
      The actual data of the tensor.
  grad : numpy.ndarray or cupy.ndarray, optional
      The gradient of the tensor, used in backpropagation.
  device : str, optional
      The device where the tensor is stored ("cuda" for GPU, None for CPU).
  _ctx : Context, optional
      The context related to this tensor, used in backpropagation.
  """
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
    """
    Moves the tensor to the specified device.

    Parameters
    ----------
    device : str
      The name of the device to which the tensor should be moved. This should be "cuda" for GPU or "cpu" for CPU.

    Returns
    -------
    Tensor
      A new tensor with the same data as this tensor, but located on the specified device.

    Example
    --------
    >>> x = Tensor([1, 2, 3])
    >>> x = x.to("cuda")  # moves x to GPU
    """
    return Tensor(self.data, device=device)

  def backward(self, allow_fill=True):
    """
    Performs backpropagation from this tensor through the computation graph.

    If this tensor is the result of an operation, this method will compute the gradient of this tensor with respect to the inputs of that operation, and recursively call backward on those inputs.

    :param allow_fill: If True and if this tensor's grad attribute is None, a new gradient tensor will be created with the same shape as this tensor's data, filled with ones. This is useful for starting backpropagation from a scalar loss tensor.
    :type allow_fill: bool, optional

    Example
    --------
    >>> x = Tensor([1, 2, 3])
    >>> y = Tensor([4, 5, 6])
    >>> z = x + y
    >>> z.backward()
    >>> print(x.grad)  # prints: array([1., 1., 1.])
    """
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
    """
    Computes and returns the mean of the tensor.

    The mean is computed by summing all elements in the tensor and dividing by the number of elements.

    :return: A new tensor containing the mean of the data of this tensor.
    :rtype: Tensor

    Example
    -------
    >>> x = Tensor([1, 2, 3, 4])
    >>> print(x.mean())  # prints: 2.5
    """
    div = Tensor(np.array([1/self.data.size]))
    return self.sum().mul(div)


class Operator:
  """
  The Operator class is a base class that provides the functionality to store the context of each operation for backward propagation.

  This class is intended to be subclassed by classes that implement specific tensor operations. The context stored by an Operator instance includes the inputs and outputs of the operation, which are used during the backward pass to compute gradients.

  :note: This class does not implement the actual operations. Subclasses should override the `forward` and `backward` methods to implement the forward and backward passes of the operation.
  
  Attributes
  ----------
  operands : list of Tensor
    The operands of the operation.
  saved_tensors : list of Tensor
    The tensors saved by the `save_for_backward` method.

  """
  def __init__(self, *tensors) -> None:
    self.operands = tensors
    self.saved_tensors = []
  
  def save_for_backward(self, *x):
    """
    Saves the input tensors for later use in the backward pass. This is done by extending the `saved_tensors` list.

    :param x: The tensors to save.
    :type x: Tensor
    """
    self.saved_tensors.extend(x)

  def apply(self, arg, *x):
    """
    Apply the operator.
  
    This method applies the operator by:

    1. Calling ``arg`` to generate a context
    2. Calling ``arg.forward`` with the context and input tensors to compute 
      the output
    3. Creating a new ``Tensor`` instance to hold the result
    4. Saving the context to the new ``Tensor`` instance for later use in 
      backpropagation

    :param arg: The object that will be used to generate the context.
    :type arg: object

    :param x: The input tensors.
    :type x: Tensor

    :return: A new tensor containing the result of the operation and with the context saved in the `_ctx` attribute.
    :rtype: Tensor
    """
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
  """
  The Mul class implements the multiplication operation for tensors.

  :note: The multiplication operation is defined as :math:`f(x, y) = x * y`.

  Example
  -------
  >>> x = Tensor([1, 2, 3])
  >>> y = Tensor([4, 5, 6])
  >>> print(x.mul(y))  # prints: [4, 10, 18]
  """
  @staticmethod
  def forward(ctx, x, y):
    """
    Computes the forward pass of the multiplication operation and saves the context of the operation.

    :param ctx: The context of the operation.
    :type ctx: Operator

    :param x: The first operand of the multiplication.
    :type x: Tensor

    :param y: The second operand of the multiplication.
    :type y: Tensor

    :return: The result of the multiplication.
    :rtype: Tensor
    """
    ctx.save_for_backward(x, y)
    return x*y

  @staticmethod
  def backward(ctx, grad_output):
    """
    Computes the backward pass of the multiplication operation.

    :param ctx: The context of the operation.
    :type ctx: Operator

    :param grad_output: The gradient of the output of the multiplication operation.
    :type grad_output: Tensor

    :return: The gradients of the multiplication operation.
    :rtype: tuple of Tensor
    """
    x,y = ctx.saved_tensors
    return y*grad_output, x*grad_output
register('mul', Mul)

class Add(Operator):
  """
  The Add class implements the addition operation for tensors and saves the context of the operation.

  :note: The addition operation is defined as :math:`f(x, y) = x + y`.

  Example
  -------
  >>> x = Tensor([1, 2, 3])
  >>> y = Tensor([4, 5, 6])
  >>> print(x.add(y))  # prints: [5, 7, 9]
  """
  @staticmethod
  def forward(ctx, x, y):
    """
    Computes the forward pass of the addition operation and saves the context of the operation.

    :param ctx: The context of the operation.
    :type ctx: Operator

    :param x: The first operand of the addition.
    :type x: Tensor

    :param y: The second operand of the addition.
    :type y: Tensor

    :return: The result of the addition.
    :rtype: Tensor
    """
    ctx.save_for_backward(x, y)
    return x+y

  @staticmethod
  def backward(ctx, grad_output):
    """
    Computes the backward pass of the addition operation.

    :param ctx: The context of the operation.
    :type ctx: Operator

    :param grad_output: The gradient of the output of the addition operation.
    :type grad_output: Tensor

    :return: The gradients of the addition operation.
    :rtype: tuple of Tensor
    """
    x,y = ctx.saved_tensors
    return grad_output, grad_output
register('add', Add)
    
class ReLU(Operator):
  """
  The ReLU class implements the ReLU operation for tensors and saves the context of the operation.

  :note: The ReLU operation is defined as :math:`f(x) = max(0, x)`.

  Example
  -------
  >>> x = Tensor([-1, 2, -3])
  >>> print(x.relu())  # prints: [0, 2, 0]
  """
  @staticmethod
  def forward(ctx, input):
    """
    Computes the forward pass of the ReLU operation and saves the context of the operation.

    :param ctx: The context of the operation.
    :type ctx: Operator

    :param input: The input tensor.
    :type input: Tensor

    :return: The result of the ReLU operation.
    :rtype: Tensor
    """
    ctx.save_for_backward(input)
    return np.maximum(input, 0)

  @staticmethod
  def backward(ctx, grad_output):
    """
    Computes the backward pass of the ReLU operation.

    :param ctx: The context of the operation.
    :type ctx: Operator

    :param grad_output: The gradient of the output of the ReLU operation.
    :type grad_output: Tensor

    :return: The gradients of the ReLU operation.
    :rtype: tuple of Tensor
    """
    input, = ctx.saved_tensors
    grad_input = grad_output.copy()
    grad_input[input < 0] = 0
    return grad_input
register('relu', ReLU)


class Dot(Operator):
  """
  The Dot class implements the dot product operation for tensors and saves the context of the operation.

  :note: The dot product operation is defined as :math:`f(x, y) = x \cdot y`.

  Example
  -------
  >>> x = Tensor([1, 2, 3])
  >>> y = Tensor([4, 5, 6])
  >>> print(x.dot(y))  # prints: 32
  """
  @staticmethod
  def forward(ctx, input, weight):
    """
    Computes the forward pass of the dot product operation and saves the context of the operation.

    :param ctx: The context of the operation.
    :type ctx: Operator

    :param input: The first operand of the dot product.
    :type input: Tensor

    :param weight: The second operand of the dot product.
    :type weight: Tensor

    :return: The result of the dot product.
    :rtype: Tensor
    """
    ctx.save_for_backward(input, weight)
    return input.dot(weight)

  @staticmethod
  def backward(ctx, grad_output):
    """
    Computes the backward pass of the dot product operation.

    :param ctx: The context of the operation.
    :type ctx: Operator

    :param grad_output: The gradient of the output of the dot product operation.
    :type grad_output: Tensor

    :return: The gradients of the dot product operation.
    :rtype: tuple of Tensor
    """
    input, weight = ctx.saved_tensors
    grad_input = grad_output.dot(weight.T)
    grad_weight = grad_output.T.dot(input).T
    return grad_input, grad_weight
register('dot', Dot)

class Sum(Operator):
  """
  The Sum class implements the sum operation for tensors and saves the context of the operation.

  :note: The sum operation is defined as :math:`f(x) = \sum_{i=1}^n x_i`.

  Example
  -------
  >>> x = Tensor([1, 2, 3])
  >>> print(x.sum())  # prints: 6
  """
  @staticmethod
  def forward(ctx, input):
    """
    Computes the forward pass of the sum operation and saves the context of the operation.

    :param ctx: The context of the operation.
    :type ctx: Operator

    :param input: The input tensor.
    :type input: Tensor

    :return: The result of the sum operation.
    :rtype: Tensor
    """
    ctx.save_for_backward(input)
    return np.array([input.sum()])

  @staticmethod
  def backward(ctx, grad_output):
    """
    Computes the backward pass of the sum operation.

    :param ctx: The context of the operation.
    :type ctx: Operator

    :param grad_output: The gradient of the output of the sum operation.
    :type grad_output: Tensor

    :return: The gradients of the sum operation.
    :rtype: tuple of Tensor
    """
    input, = ctx.saved_tensors
    return grad_output * np.ones_like(input)
register('sum', Sum)


class LogSoftmax(Operator):
  """
  The LogSoftmax class implements the log softmax operation for tensors and saves the context of the operation.

  :note: The log softmax operation is defined as :math:`f(x) = log(\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}})`.

  Example
  -------
  >>> x = Tensor([1, 2, 3])
  >>> print(x.logsoftmax())
  """
  @staticmethod
  def forward(ctx, input):
    """
    Computes the forward pass of the log softmax operation.

    :param ctx: The context of the operation.
    :type ctx: Operator

    :param input: The input tensor.
    :type input: Tensor

    :return: The result of the log softmax operation.
    :rtype: Tensor
    """
    def logsumexp(x):
      c = x.max(axis=1)
      return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
    output = input - logsumexp(input).reshape((-1, 1))
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    """
    Computes the backward pass of the log softmax operation.

    :param ctx: The context of the operation.
    :type ctx: Operator

    :param grad_output: The gradient of the output of the log softmax operation.
    :type grad_output: Tensor

    :return: The gradients of the log softmax operation.
    :rtype: tuple of Tensor
    """
    output, = ctx.saved_tensors
    return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
register('logsoftmax', LogSoftmax)

class MSE(Operator):
  """
  The MSE class implements the mean squared error operation for tensors and saves the context of the operation.
  Work In Progress
  """
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
  """
  The Conv2d class implements the 2D convolution operation for tensors and saves the context of the operation.

  :note: The 2D convolution operation is defined as :math:`f(x) = x * w`.

  Example
  -------
  
  """
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