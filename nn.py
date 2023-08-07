from tensor import register, Tensor
import numpy as np
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
      p.data -= lr * p.grad


class Linear(Layer):
  def __init__(self, in_features, out_features):
    super().__init__()

    # Register parameters
    self.params["w"] = Tensor((np.random.uniform(-1., 1.,size=(in_features, out_features))/np.sqrt(in_features*out_features)).astype(np.float32)) 
    self.params["b"] = Tensor(np.random.uniform(-1., 1., size=(out_features)))

  def __call__(self, x):
    # Forward pass
    x = x.dot(self.params["w"])
    print(x.data.shape)
    x = x.add(self.params["b"])

    return x