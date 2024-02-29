from .utils import import_cupy_else_numpy
import numpy as np
import cupy
from functools import partialmethod
from itertools import product
from typing import Union, Optional, List, Tuple
import graphviz


ndarray = Union[np.ndarray, cupy.ndarray]

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
  def __init__(self, data: Union[ndarray, List[Union[float, int]], int, float] , grad: Optional[ndarray] = None, device: Optional[str] = None):
    self.grad = grad
    self.device = device
    if device == "cuda":
      global np
      np = import_cupy_else_numpy()
    self.data = np.array(data)
    self._ctx: Operator = None
  def __repr__(self):
    return f"Tensor({self.data})"

  def __str__(self):
    return f"Tensor shape: {self.data.shape}\nGradient: {self.grad}\nDevice: {self.device}"

  def __add__(self, other: 'Tensor'):
    return self.add(other)
  
  def __mul__(self, other: 'Tensor'):
    return self.mul(other)
  
  def __sub__(self, other: 'Tensor'):
    return self.add(other.mul(-1))
  
  def to(self, device: str):
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

  def backward(self, allow_fill: bool = True) -> None:
    """
    Performs backpropagation from this tensor through the computation graph using a depth-first search (DFS).

    If this tensor is the result of an operation, this method will compute the gradient of this tensor with respect to the inputs of that operation, and recursively call backward on those inputs. This recursive process is essentially a DFS through the computation graph.

    Parameters
    ----------
    allow_fill : bool, optional
      If True, fills in the first gradient with one if the gradient is None. This is used to fill in the gradient of a scalar with respect to itself.

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

    parent_grads = self._ctx.backward(self._ctx, self.grad) # tuple or np.ndarray
    if len(self._ctx.operands) == 1:
      parent_grads = np.expand_dims(parent_grads, axis=0)

    for tensor, gradient in zip(self._ctx.operands, parent_grads):
      if tensor.data.shape != gradient.shape:
        # take a mean of the gradients of a axis
        gradient = gradient.mean(axis=0)

      if tensor.grad is None:
        tensor.grad = gradient
      else:
        tensor.grad += gradient

      tensor.backward(False)

  def mean(self) -> 'Tensor':
      """
      Computes and returns the mean of the tensor.

      The mean is computed by summing all elements in the tensor and dividing by the number of elements.

      Returns
      -------
      Tensor
        A new tensor containing the mean of this tensor.

      Example
      -------
      >>> x = Tensor([1, 2, 3, 4])
      >>> print(x.mean())  # prints: 2.5
      """
      div = Tensor(np.array([1/self.data.size]))
      return self.sum().mul(div)
  
  def draw_graph(self, namespace: Optional[dict] = None) -> None:
        """
        Draws the computation graph of the tensor and its operands.

        This method uses the graphviz library to draw the computation graph of the tensor and its operands. The graph is saved as a PNG file. The namespace parameter is used to find the variable name of the tensor in the scope where draw_graph is called.

        Parameters
        ----------
        namespace : dict, default=None
            An optional dictionary of the local variables in the scope where draw_graph is called. If not provided, the function will use the local variables in the scope where it is called.

        Example
        -------
        >>> x = Tensor([1, 2, 3])
        >>> y = Tensor([4, 5, 6])
        >>> z = x + y
        >>> z.draw_graph(locals())
        """
        # DFS to build the graph
        def get_label(tensor: Tensor):
              return f'shape: {tensor.data.shape} grad: {True if tensor.grad is not None else False}'
        
        def build_graph(tensor, dot: graphviz.Digraph):
            if tensor in dot:
                return
            # Find the variable name in the calling scope
            name = None
            if namespace is not None:
                for var_name, var_value in namespace.items():
                    if var_value is tensor:
                        name = var_name
                        break

            # Use the variable name as the label if it's found, otherwise use the tensor's id
            label = name if name is not None else str(id(tensor))

            dot.node(str(id(tensor)), f'{label}: {get_label(tensor)}', shape='box')
            if tensor._ctx is not None:
                dot.node(str(id(tensor._ctx)), type(tensor._ctx).__name__)
                dot.edge(str(id(tensor._ctx)), str(id(tensor)))
                for operand in tensor._ctx.operands:
                    if isinstance(operand, Tensor):
                      dot.edge(str(id(operand)), str(id(tensor._ctx)))
                      build_graph(operand, dot)
            else:
                return

        dot = graphviz.Digraph()
        build_graph(self, dot)
        dot.render("graph", format="png", cleanup=True)

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
  def __init__(self, *tensors: Tensor) -> None:
    self.operands = tensors
    self.saved_tensors: List[Tensor] = []
  
  def save_for_backward(self, *x: Tensor):
    """
    Saves the input tensors for later use in the backward pass. This is done by extending the `saved_tensors` list.

    Parameters
    ----------
    x : Tensor
      The input tensors to be saved.
    """
    self.saved_tensors.extend(x)

  def apply(self, arg: 'Operator', *x: tuple) -> Tensor:
    """
    Apply the operator.
  
    This method applies the operator by:

    1. Calling ``arg`` to generate a context
    2. Calling ``arg.forward`` with the context and input tensors to compute 
      the output
    3. Creating a new ``Tensor`` instance to hold the result
    4. Saving the context to the new ``Tensor`` instance for later use in 
      backpropagation

    Parameters
    ----------
    arg : Operator
      The operator to be applied
    x : tuple
      The input tensors to the operator
    
    Returns
    -------
    Tensor
      The result of the operation
    """
    ctx = arg(self, *x)
    ret = Tensor(arg.forward(ctx, self.data, *[t.data if isinstance(t, Tensor) else t for t in x]))
    ret._ctx = ctx
    return ret

def register(name, fxn):
  """
  Adds a method to the Tensor class. The new method is a partial application of the `apply` method of the `fxn` object.

  Parameters
  ----------
  name : str
    The name of the new method
  fxn : Operator
    The operator to be applied

  Example
  -------
  If `fxn` is an instance of a class `Mul` with an `apply` method that multiplies its arguments:
  >>> register('mul', Mul)
  >>> x = Tensor([1, 2, 3])
  >>> y = Tensor([4, 5, 6])
  >>> print(x.mul(y))  # prints: [4, 10, 18]
  """
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))

class Mul(Operator):
  """
  The Mul class implements the hadamard product operation for tensors and saves the context of the operation.

  :note: The multiplication operation is equivalent to a hadamard product.

  Example
  -------
  >>> x = Tensor([1, 2, 3])
  >>> y = Tensor([4, 5, 6])
  >>> print(x.mul(y))  # prints: [4, 10, 18]
  """
  @staticmethod
  def forward(ctx, x, y) -> ndarray:
    """
    Computes the forward pass of the multiplication operation and saves the context of the operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    x : Tensor
      The first operand of the multiplication.
    y : Tensor
      The second operand of the multiplication.
    
    Returns
    -------
    ndarray
      The result of the multiplication operation
    """
    ctx.save_for_backward(x, y)
    return x*y

  @staticmethod
  def backward(ctx, grad_output) -> Tuple[ndarray, ndarray]:
    """
    Computes the backward pass of the multiplication operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    grad_output : Tensor
      The gradient of the loss with respect to the current layer's output. Used to compute gradients for the layer's inputs (if needed) and weights.
    
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
  def forward(ctx, x, y) -> ndarray:
    """
    Computes the forward pass of the addition operation and saves the context of the operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    x : ndarray
      The first operand of the addition.
    y : ndarray
      The second operand of the addition.

    Returns
    -------
    ndarray
      The result of the addition operation
    """
    ctx.save_for_backward(x, y)
    return x+y

  @staticmethod
  def backward(ctx, grad_output) -> Tuple[ndarray, ndarray]:
    """
    Computes the backward pass of the addition operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    grad_output : ndarray
      The gradient of the loss with respect to the current layer's output. Used to compute gradients for the layer's inputs (if needed) and weights.
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
  def forward(ctx: Operator, input: ndarray) -> ndarray:
    """
    Computes the forward pass of the ReLU operation and saves the context of the operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    input : ndarray
      The input ndarray of a tensor.
    """
    ctx.save_for_backward(input)
    return np.maximum(input, 0)

  @staticmethod
  def backward(ctx: Operator, grad_output: ndarray) -> ndarray:
    """
    Computes the backward pass of the ReLU operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    grad_output : ndarray
      The gradient of the loss with respect to the current layer's output. Used to compute gradients for the layer's inputs (if needed) and weights.
    
    Returns
    -------
    ndarray
      The gradients of the ReLU operation.
    """
    input, = ctx.saved_tensors
    grad_input = grad_output.copy()
    grad_input[input < 0] = 0
    return grad_input
register('relu', ReLU)


class Dot(Operator):
  """
  The Dot class implements the matrix multiplication operation for tensors and saves the context of the operation.

  :note: Matrix multiplication is a binary operation that produces a matrix from two matrices. For matrix multiplication, the number of columns in the first matrix must be equal to the number of rows in the second matrix.

  Example
  -------
  >>> x = Tensor([1, 2, 3])
  >>> y = Tensor([4, 5, 6])
  >>> print(x.dot(y))  # prints: 32
  """
  @staticmethod
  def forward(ctx: Operator, input: ndarray, weight: ndarray) -> ndarray:
    """
    Computes the forward pass of the dot product operation and saves the context of the operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    input : ndarray
      The input ndarray of a tensor.
    weight : ndarray
      The weight ndarray of a tensor.
    
    Returns
    -------
    ndarray
      The result of the dot product operation.
    """
    ctx.save_for_backward(input, weight)
    return input.dot(weight)

  @staticmethod
  def backward(ctx: Operator, grad_output: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Computes the backward pass of the dot product operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    grad_output : ndarray
      The gradient of the loss with respect to the current layer's output. Used to compute gradients for the layer's inputs (if needed) and weights.
    
    Returns
    -------
    tuple of ndarray
      The gradients of the dot product operation.
    """
    input, weight = ctx.saved_tensors
    grad_input = grad_output.dot(weight.T)
    grad_weight = input.T.dot(grad_output)
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
  def forward(ctx: Operator, input: ndarray) -> ndarray:
    """
    Computes the forward pass of the sum operation and saves the context of the operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    input : ndarray
      The input ndarray of a tensor.
    
    Returns
    -------
    ndarray
      The result of the sum operation.
    """
    ctx.save_for_backward(input)
    return np.array([input.sum()])

  @staticmethod
  def backward(ctx: Operator, grad_output: ndarray) -> ndarray:
    """
    Computes the backward pass of the sum operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    grad_output : ndarray
      The gradient of the loss with respect to the current layer's output. Used to compute gradients for the layer's inputs (if needed) and weights.
    
    Returns
    -------
    ndarray
      The gradients of the sum operation.
    """
    input, = ctx.saved_tensors
    return grad_output * np.ones_like(input)
register('sum', Sum)

class Softmax(Operator):
  """
  Implements the softmax operation for tensors.
  """
  @staticmethod
  def forward(ctx: Operator, input: ndarray) -> ndarray:
    """
    Computes the forward pass of the softmax operation and saves the context of the operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    input : ndarray
      The input ndarray of a tensor.
    
    Returns
    -------
    ndarray
      The result of the softmax operation.
    """
    exp = np.exp(input - np.max(input, axis=-1, keepdims=True))
    output = exp / exp.sum(axis=-1, keepdims=True)
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx: Operator, grad_output: ndarray) -> ndarray:
      """
      Computes the backward pass of the softmax operation.

      Parameters
      ----------
      ctx : Operator
        The context of the operation.
      grad_output : ndarray
        The gradient of the loss with respect to the current layer's output. Used to compute gradients for the layer's inputs (if needed) and weights.
      
      Returns
      -------
      ndarray
        The gradients of the softmax operation.
      
      """
      output, = ctx.saved_tensors
      d_softmax = output * (1 - output)  # Diagonal part of the Jacobian
      jacobian = -output[..., None] * output[:, None, :]  # Off-diagonal part
      jacobian[:, np.arange(output.shape[1]), np.arange(output.shape[1])] = d_softmax  # Combine diagonal
      return np.einsum('ij,ijk->ik', grad_output, jacobian)
register('softmax', Softmax)

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
  def forward(ctx: Operator, input: ndarray) -> ndarray:
    """
    Computes the forward pass of the log softmax operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    input : ndarray
      The input ndarray of a tensor.

    Returns
    -------
    ndarray
      The result of the log softmax operation.

    """
    def logsumexp(x):
      c = x.max(axis=1)
      return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
    output = input - logsumexp(input).reshape((-1, 1))
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx: Operator, grad_output: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Computes the backward pass of the log softmax operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    grad_output : ndarray
      The gradient of the loss with respect to the current layer's output. Used to compute gradients for the layer's inputs (if needed) and weights.
    
    Returns
    -------
    ndarray
      The gradients of the log softmax operation.
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
  def forward(ctx: Operator, input: ndarray, target: ndarray) -> ndarray:
    """
    Computes the forward pass and saves tensors for backward pass.

    Parameters
    ----------
    ctx : Operator
        The context of the operation.
    input : ndarray
        The input tensor.
    target : ndarray
        The target tensor.
    
    Returns
    -------
    ndarray
        The mean squared error between input and target.
    """

    ctx.save_for_backward(input, target)
    return np.array((input - target)**2)

  @staticmethod
  def backward(ctx: Operator, grad_output: ndarray) -> Tuple[ndarray, ndarray]:
    input, target = ctx.saved_tensors
    grad_input = grad_output * 2*(input - target)
    grad_target = grad_output * -2*(input - target)
    return grad_input, grad_target
register('mse', MSE)

class Conv2d(Operator):
  # https://www.cs.cmu.edu/~aarti/Class/10315_Spring22/315S22_Rec6.pdf
  """
  Implements the 2D convolution operation for tensors.

  This class is a subclass of Operator and defines the forward and backward methods for the 2D convolution operation.

  :note: The 2D convolution operation is defined as :math:`f(x) = \sum_{m} \sum_{n} x[m, n] * w[m, n]`.

  Example
  -------
  >>> input = Tensor(np.random.rand(1, 3, 32, 32))
  >>> weight = Tensor(np.random.rand(16, 3, 5, 5))
  >>> padding = 2
  >>> stride = 1
  >>> out = input.conv2d(weight, padding, stride)
  >>> print(out.shape)  # Example output: (1, 16, 32, 32)
  """
  @staticmethod
  def forward(ctx: Operator, input: ndarray, weight: ndarray, padding: int, stride: int) -> ndarray:
    """
    Computes the forward pass of the 2D convolution operation and saves the context of the operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    input : ndarray
      The input tensor.
    weight : ndarray
      The weight tensor.
    padding : int
      The amount of padding to apply to the input tensor.
    stride : int
      The stride of the convolution operation.
    
    Returns
    -------
    ndarray
      The result of the 2D convolution operation.
    """
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
  def backward(ctx: Operator, grad_output: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Computes the backward pass of the 2D convolution operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    grad_output : ndarray
      The gradient of the loss with respect to the current layer's output. Used to compute gradients for the layer's inputs (if needed) and weights.
    
    Returns
    -------
    tuple of ndarray
      The gradients of the 2D convolution operation.
    """
    input, weight, padding, stride = ctx.saved_tensors
    
    N, C, H, W = input.shape
    F, _, HH, WW = weight.shape
    out_h = (H + 2 * padding - HH) // stride + 1
    out_w = (W + 2 * padding - WW) // stride + 1
    grad_weight = np.zeros_like(weight)
    grad_input = np.zeros_like(input)

    input_pad = np.pad(input, ((0,0), (0,0), (padding, padding), (padding, padding)), 'constant')
    
    for n in range(N):
      for f in range(F):
        for i in range(out_h):
          for j in range(out_w):
            input_patch = input_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            grad_weight[f] += input_patch * grad_output[n,f,i,j]
    
    return grad_input, grad_weight
register('conv2d', Conv2d)

class Reshape(Operator):
  """
  Implements the reshape operation for tensors.

  This class is a subclass of Operator and defines the forward and backward methods for reshaping a tensor.

  Example
  -------
  >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
  >>> y = x.reshape((3, 2))
  >>> print(y)  # Example output: Tensor([[1, 2], [3, 4], [5, 6]])
  """
  @staticmethod
  def forward(ctx: Operator, x: ndarray, shape: tuple) -> ndarray:
    """
    Computes the forward pass of the reshape operation and saves the context of the operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    x : ndarray
      The input tensor.
    shape : tuple
      The new shape of the tensor.
    
    Returns
    -------
    ndarray
      The result of the reshape operation.
    """
    ctx.operands = (ctx.operands[0],)
    ctx.save_for_backward(x.shape)
    return x.reshape(shape)
  @staticmethod
  def backward(ctx: Operator, grad_output: ndarray) -> ndarray:
    """
    Computes the backward pass of the reshape operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    grad_output : ndarray
      The gradient of the loss with respect to the current layer's output. Used to compute gradients for the layer's inputs (if needed) and weights.
    
    Returns
    -------
    ndarray
      The gradients of the reshape operation.
    
    """
    original_shape, = ctx.saved_tensors
    return grad_output.reshape(original_shape)
register('reshape', Reshape)

class MAE(Operator):
  """
  Implements the Mean Absolute Error (MAE) operation for tensors.

  This class is a subclass of Operator and defines the forward and backward methods for computing the MAE between two tensors.

  Example
  -------
  >>> predictions = Tensor(np.array([3.0, -0.5, 2, 7]))
  >>> targets = Tensor(np.array([2.5, 0.0, 2, 8]))
  >>> error = predictions.mae(targets)
  >>> print(error)  # Example output: Tensor([0.5, 0.5, 0, 1])
  """
  @staticmethod
  def forward(ctx: Operator, input: ndarray, target: ndarray) -> ndarray:
    """
    Computes the forward pass of the MAE operation and saves the context of the operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    input : ndarray
      The input tensor.
    target : ndarray
      The target tensor.
    
    Returns
    -------
    ndarray
      The result of the MAE operation.
    """
    ctx.save_for_backward(input, target)
    return np.abs(input - target)

  @staticmethod
  def backward(ctx: Operator, grad_output: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    grad_output : ndarray
      The gradient of the loss with respect to the current layer's output. Used to compute gradients for the layer's inputs (if needed) and weights.
    
    Returns
    -------
    tuple of ndarray
      The gradients of the MAE operation.
    """
    input, target = ctx.saved_tensors
    grad_input = grad_output * np.sign(input - target)
    grad_target = grad_output * -np.sign(input - target)
    return grad_input, grad_target
register('mae', MAE)

class Exp(Operator):
  """
  Implements the exponential operation for tensors.

  This class is a subclass of Operator

  :note: The exponential operation is defined as :math:`f(x) = e^x`.

  Example
  -------
  >>> x = Tensor([1, 2, 3])
  >>> print(x.exp())  # Example output: Tensor([2.71828183, 7.3890561 , 20.08553692])
  """
  @staticmethod
  def forward(ctx: Operator, input: ndarray) -> ndarray:
    """
    Computes the forward pass of the exponential operation and saves the context of the operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    input : ndarray
      The input tensor.
    
    Returns
    -------
    ndarray
      The result of the exponential operation.
    
    """
    ctx.save_for_backward(input)
    return np.exp(input)

  @staticmethod
  def backward(ctx: Operator, grad_output: ndarray) -> ndarray:
    """
    Computes the backward pass of the exponential operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    grad_output : ndarray
      The gradient of the loss with respect to the current layer's output. Used to compute gradients for the layer's inputs (if needed) and weights.
    
    Returns
    -------
    ndarray
      The gradients of the exponential operation.
    """
    input, = ctx.saved_tensors
    return grad_output * np.exp(input)
register('exp', Exp)

class Log(Operator):
  """
  Implements the natural logarithm operation for tensors.

  This class is a subclass of Operator

  :note: The natural logarithm operation is defined as :math:`f(x) = \log(x)`.

  Example
  -------

  >>> x = Tensor([1, 2, 3])
  >>> print(x.log())  # Example output: Tensor([0., 0.69314718, 1.09861229])
  """
  @staticmethod
  def forward(ctx: Operator, input: ndarray) -> ndarray:
    """
    Computes the forward pass of the natural logarithm operation and saves the context of the operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    input : ndarray
      The input tensor.
    
    Returns
    -------
    ndarray
      The result of the natural logarithm operation.
    """
    ctx.save_for_backward(input)
    return np.log(input)

  @staticmethod
  def backward(ctx: Operator, grad_output: ndarray) -> ndarray:
    """
    Computes the backward pass of the natural logarithm operation.

    Parameters
    ----------
    ctx : Operator
      The context of the operation.
    grad_output : ndarray
      The gradient of the loss with respect to the current layer's output. Used to compute gradients for the layer's inputs (if needed) and weights.
    
    Returns
    -------
    ndarray
      The gradients of the natural logarithm operation.
    """
    input, = ctx.saved_tensors
    return grad_output / input
register('log', Log)