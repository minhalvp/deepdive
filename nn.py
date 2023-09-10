from tensor import Tensor
import numpy as np
import cupy as cp

# TODO - implement the following:
# 1. Linear Layers
# 2. something like nn.sequential
# 3. Convolutional Layers 
# 4. normalization layers


class Layer:
  def __init__(self):
    self.params = {}
  
  def step(self, lr):
    for p in self.params.values():
      if p.grad.shape != p.data.shape:
        # print(f"Warning: grad shape {p.grad.shape} != data shape {p.data.shape}, assuming batched data and averaging gradients")
        p.grad = p.grad.mean(axis=0)
      p.data -= lr * p.grad

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

  def __str__(self) -> str:
    return f"Sequential({', '.join([str(type(l)) for l in self.layers])})"
  def step(self, lr):
    for layer in self.layers:
      if isinstance(layer, Layer):
        layer.step(lr)

  def to(self, device):
    for layer in self.layers:
      if isinstance(layer, Layer):
        layer.to(device)
    global np
    if device == "cuda":
      np = cp
    else:
      np = np
    return self