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
    params (list): list of parameters to optimize
    lr (float): learning rate    
    """
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr

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
    """
    def step(self):
        """
        Performs a single optimization step by updating the parameters in the direction of the gradient with the learning rate.
        """
        for p in self.params:
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
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        """
        Performs a single optimization step by updating the parameters in the direction of the gradient with the learning rate. More information can be found in the Analysis algorithms research.
        """
        self.t += 1
        for i, p in enumerate(self.params):
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
  params (dict): dictionary of parameters
  """
  def __init__(self):
    self.params = {}
  
  def step(self, lr, optimizer):
    """
    Calls the step function of the optimizer on the parameters of all the layers.

    :param lr: learning rate
    :type lr: float
    :param optimizer: optimizer to use
    :type optimizer: Optimizer
    """
    for p in self.params.values():
      if p.grad.shape != p.data.shape:
        # print(f"Warning: grad shape {p.grad.shape} != data shape {p.data.shape}, assuming batched data and averaging gradients")
        p.grad = p.grad.mean(axis=0)
    optim = optimizer(self.params.values(), lr=lr)
    optim.step()

class Linear(Layer):
  def __init__(self, in_features, out_features):
    super().__init__()

    w = np.random.uniform(-1., 1., size=(in_features,out_features))/np.sqrt(in_features*out_features)
    b = np.zeros((out_features,))
    self.params["LinearW"] = Tensor(w)
    self.params["LinearB"] = Tensor(b)

  def __call__(self, x):
    # Todo: combining the weights and biases into a single tensor so forward pass is 1 operation
    x = x.dot(self.params["LinearW"])
    x = x.add(self.params["LinearB"])
    return x
  
  def to(self, device):
    for name, p in self.params.items():
      self.params[name] = p.to(device)
    return self
  
class Conv2d(Layer):
  """
  FOLLOW THIS RULE -> O = (W - K + 2P) / S + 1
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    w = np.random.uniform(-1., 1., size=(out_channels, in_channels, kernel_size, kernel_size))/np.sqrt(in_channels*out_channels*kernel_size*kernel_size)
    self.params["Conv2dW"] = Tensor(w)

  def __call__(self, x):
    output = x.conv2d(self.params["Conv2dW"], self.padding, self.stride)
    return output

class ReShape():
  def __init__(self, shape: tuple) -> None:
     self.shape = shape

  def __call__(self, x):
      return x.reshape(self.shape)
  
class Flatten():
  def __call__(self, x):
    return x.flatten()
class ReLU():
  def __call__(self, x):
    return x.relu()
  
class LogSoftmax():
  def __call__(self, x):
    return x.logsoftmax()
  
class Sequential:
  def __init__(self, *layers):
    self.layers = layers

  def forward(self, x):
    for l in self.layers:
      x = l(x)
    return x  
  def save(self, path):
    model_dict = {}
    for i, l in enumerate(self.layers):
      model_dict[f"layer_{i}"] = l.params
    np.save(path, model_dict)
  def __str__(self) -> str:
    return f"Sequential({', '.join([str(type(l)) for l in self.layers])})"
  def step(self, lr, optimizer):
    for layer in self.layers:
      if isinstance(layer, Layer):
        layer.step(lr, optimizer=optimizer)

  def to(self, device):
    for layer in self.layers:
      if isinstance(layer, Layer):
        layer.to(device)
    global np
    if device == "cuda":
      np = import_cupy_else_numpy()
    else:
      np = np
    return self