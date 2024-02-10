from .tensor import Tensor
from .utils import import_cupy_else_numpy
from abc import ABC, abstractmethod
import numpy as np

# TODO - implement the following:
# 1. Linear Layers
# 2. something like nn.sequential
# 3. Convolutional Layers 
# 4. normalization layers
# 5. Model saving and loading

class Optimizer(ABC):
    """
    Base class for all optimizers

    Attributes
    ----------
    params (list): list of parameters from the model
    lr (float): learning rate
    weight_decay (float): weight decay (L2 penalty)
    """
    def __init__(self, params: dict = None, lr: int = 1e-3, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
    @abstractmethod
    def step(self):
        """
        Performs a single optimization step. Not implemented in base class. This function should be implemented in all subclasses.
        """
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer

    Attributes
    ----------
    params (list): list of parameters to optimize
    lr (float): learning rate
    weight_decay (float): weight decay (L2 penalty)
    """
    def step(self):
        """
        Performs a single optimization step by updating the parameters in the direction of the gradient with the learning rate.
        """
        for p in self.params:
          if self.weight_decay != 0:
            p.grad += self.weight_decay * p.data
          p.data -= self.lr * p.grad


class Adam(Optimizer):
    """
    Adaptive Moment Estimation optimizer

    Attributes
    ----------
    params (list): list of parameters to optimize
    lr (float): learning rate
    betas (tuple): coefficients used for computing running averages of gradient and its square
    eps (float): term added to the denominator to improve numerical stability
    weight_decay (float): weight decay (L2 penalty)
    """
    def __init__(self, params: dict = None, lr: int = 1e-3, betas: tuple = (0.9, 0.999), eps: int = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params.values()]
        self.v = [np.zeros_like(p.data) for p in self.params.values()]
        self.t = 0

    def step(self):
        """
        Performs a single optimization step by updating the parameters in the direction of the gradient with the learning rate. More information can be found in the Analysis algorithms research.
        """
        self.t += 1
        for i, p in enumerate(self.params):
          if self.weight_decay != 0:
            p.grad += self.weight_decay * p.data
          self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad
          self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * p.grad**2
          m_hat = self.m[i] / (1 - self.betas[0]**self.t)
          v_hat = self.v[i] / (1 - self.betas[1]**self.t)
          p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Layer:
  """
  Base class for all layers

  Attributes
  ----------
  params (dict): dictionary of parameters. The keys are the name of the layer followed by the paramter type (W, B) and the values are the parameter tensors.

  Notes
  -----
  Not all layers have both W and B parameters. For example, Conv2d layers only have W parameters.
  """
  def __init__(self):
    self.params = {}
  
  def step(self, lr: int, weight_decay: float, optimizer: Optimizer):
    """
    Calls the step function of the optimizer on the parameters of all the layers.

    :param lr: learning rate
    :type lr: float
    :param weight_decay: weight decay (L2 penalty)
    :type weight_decay: float
    :param optimizer: optimizer to use
    :type optimizer: Optimizer
    """
    for p in self.params.values():
      if p.grad.shape != p.data.shape:
        # print(f"Warning: grad shape {p.grad.shape} != data shape {p.data.shape}, assuming batched data and averaging gradients")
        p.grad = p.grad.mean(axis=0)
    optim = optimizer(self.params.values(), lr, weight_decay)
    optim.step()

class Linear(Layer):
  """
  Linear Neural Network Layer

  Attributes
  ----------
  in_features (int): number of input features
  out_features (int): number of output features
  """
  def __init__(self, in_features: int, out_features: int):
    super().__init__()

    w = np.random.uniform(-1., 1., size=(in_features,out_features))/np.sqrt(in_features*out_features)
    b = np.zeros((out_features,))
    self.params["LinearW"] = Tensor(w)
    self.params["LinearB"] = Tensor(b)

  def __call__(self, x):
    # Todo: combining the weights and biases into a single tensor so forward pass is 1 operation
    """
    Performs a linear transformation on the input tensor and Linear parameters and returns the output tensor.

    :param x: input tensor
    :type x: Tensor

    :return: output tensor
    :rtype: Tensor
    """  
    x = x.dot(self.params["LinearW"])
    x = x.add(self.params["LinearB"])
    return x
  
  def to(self, device: str):
    """
    Moves the parameters of the layer to the specified device.

    :param device: device to move the parameters to
    :type device: str (either "cpu" or "cuda")
    """
    for name, p in self.params.items():
      self.params[name] = p.to(device)
    return self
  
class Conv2d(Layer):
  """
  2D Convolutional Neural Network Layer

  Attributes
  ----------
  in_channels (int): number of input channels
  out_channels (int): number of output channels
  kernel_size (int): size of the kernel
  stride (int): stride of the kernel
  padding (int): padding of the kernel
  params (dict): dictionary of parameters.

  Notes
  -----
  FOLLOW THIS RULE -> O = (W - K + 2P) / S + 1
  Where:
  O = output height/length
  W = input height/length
  K = kernel size
  P = padding
  S = stride
  """
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    w = np.random.uniform(-1., 1., size=(out_channels, in_channels, kernel_size, kernel_size))/np.sqrt(in_channels*out_channels*kernel_size*kernel_size)
    self.params["Conv2dW"] = Tensor(w)

  def __call__(self, x: Tensor):
    """
    Performs a 2D convolution on the input tensor and Conv2d parameters and returns the output tensor.

    :param x: input tensor
    :type x: Tensor

    :return: output tensor
    :rtype: Tensor
    """
    output = x.conv2d(self.params["Conv2dW"], self.padding, self.stride)
    return output

class ReShape():
  """
  Layer for reshaping the input tensor to the specified shape
  """
  def __init__(self, shape: tuple) -> None:
     self.shape = shape

  def __call__(self, x):
      """
      Reshapes the input tensor to the specified shape

      :param x: input tensor
      :type x: Tensor
      """
      return x.reshape(self.shape)
class ReLU():
  """
  Layer for applying the ReLU activation function
  """
  def __call__(self, x: Tensor):
    """
    Applies the ReLU activation function to the input tensor

    :param x: input tensor
    :type x: Tensor
    """
    return x.relu()
  
class LogSoftmax():
  """
  Layer for applying the LogSoftmax activation function
  """
  def __call__(self, x: Tensor):
    """
    Applies the LogSoftmax activation function to the input tensor

    :param x: input tensor
    :type x: Tensor
    """
    return x.logsoftmax()
  
class Sequential:
  """
  Sequential Neural Network Model. This class is used to combine multiple layers into a single model.

  Attributes
  ----------
  layers (list): list of layers to combine

  Example
  -------
  >>> model = nn.Sequential(
  ...     nn.Conv2d(1, 4, 3, 1, 0),
  ...     nn.ReLU(),
  ...     nn.Conv2d(4, 8, 3, 1, 0),
  ...     nn.ReLU(),
  ...     nn.ReShape((-1, 8*24*24)),
  ...     nn.Linear(4608, 1024),
  ...     nn.ReLU(),
  ...     nn.Linear(1024, 10),
  ... )
  >>> input = Tensor(np.random.randn(32, 28, 28))
  >>> output = model.forward(input)
  """
  def __init__(self, *layers: Layer):
    self.layers = layers

  def forward(self, x: Tensor):
    """
    Performs a forward pass on the input tensor through all the layers and returns the output tensor.

    :param x: input tensor
    :type x: Tensor

    :return: output tensor
    :rtype: Tensor
    """
    for l in self.layers:
      x = l(x)
    return x  
  def save(self, path):
    # WIP
    """
    Saves the model parameters to the specified path.

    :param path: path to save the model parameters to
    :type path: str
    """
    model_dict = {}
    for i, l in enumerate(self.layers):
      model_dict[f"layer_{i}"] = l.params
    np.save(path, model_dict)
  def __str__(self) -> str:
    return f"Sequential({', '.join([str(type(l)) for l in self.layers])})"
  def step(self, lr, weight_decay, optimizer: Optimizer):
    """
    Calls the step function of the optimizer on the parameters of all the layers.

    :param lr: learning rate
    :type lr: float

    :param optimizer: optimizer to use
    :type optimizer: Optimizer
    """
    for layer in self.layers:
      if isinstance(layer, Layer):
        layer.step(lr, weight_decay, optimizer)

  def to(self, device: str):
    """
    Moves the parameters of the model to the specified device.

    :param device: device to move the parameters to
    :type device: str (either "cpu" or "cuda")

    :return: self
    :rtype: Sequential
    """
    for layer in self.layers:
      if isinstance(layer, Layer):
        layer.to(device)
    global np
    if device == "cuda":
      np = import_cupy_else_numpy()
    else:
      np = np
    return self