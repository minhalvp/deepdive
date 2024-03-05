from .tensor import Tensor
from .utils import import_cupy_else_numpy
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, TypedDict
import os


class Optimizer(ABC):
    """
    Base class for all optimizers

    Attributes
    ----------
    params (list): list of parameters from the model
    lr (float): learning rate
    weight_decay (float): weight decay (L2 regularization penalty)
    """

    def __init__(
            self,
            params: Optional[dict] = None,
            lr: float = 1e-3,
            weight_decay: Optional[float] = 0.0
        ):
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
    weight_decay (float): weight decay (L2 regularization penalty)
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
    weight_decay (float): weight decay (L2 regularization penalty)
    """

    def __init__(
            self,
            params: Optional[dict] = None,
            lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: Optional[float] = 0.0
        ):
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
            self.m[i] = self.betas[0] * self.m[i] + \
                (1 - self.betas[0]) * p.grad
            self.v[i] = self.betas[1] * self.v[i] + \
                (1 - self.betas[1]) * p.grad**2
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
        self.params: Dict[str, Tensor] = {}

    def step(self, optimizer: Optimizer, lr: float = 1e-3, weight_decay: Optional[float] = 0.0):
        """
        Calls the step function of the optimizer on the parameters of all the layers.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer to use for updating the parameters.
        lr : float, default=1e-3
            The learning rate for the optimizer.
        weight_decay : float, optional, default=0.0
            The weight decay (L2 regularization penalty) for the optimizer.
        """
        for p in self.params.values():
            if p.grad.shape != p.data.shape:
                # print(f"Warning: grad shape {p.grad.shape} != data shape {p.data.shape}, assuming batched data and averaging gradients")
                p.grad = p.grad.mean(axis=0)
        optim = optimizer(self.params.values(), lr, weight_decay)
        optim.step()
        #  set gradients to zero
        self.zero_grad()

    def zero_grad(self):
        """
        Sets the gradient of all the parameters to zero.
        """
        for p in self.params.values():
            p.grad.fill(0)


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

        w = np.random.uniform(
            low=-1.,
            high=1.,
            size=(in_features, out_features)
        ) / np.sqrt(in_features * out_features)

        b = np.zeros(
            shape=(out_features,)
        )
        self.params = {
            "LinearW": Tensor(w),
            "LinearB": Tensor(b)
        }

    def __str__(self) -> str:
        return f"Linear({self.params['LinearW'].data.shape[0]},{self.params['LinearW'].data.shape[1]})"

    def __call__(self, x: Tensor) -> Tensor:
        # Todo: combining the weights and biases into a single tensor so forward pass is 1 operation
        """
        Performs a linear transformation on the input tensor and Linear parameters and returns the output tensor.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The output tensor.
        """
        x = x.dot(self.params["LinearW"])
        x = x.add(self.params["LinearB"])
        return x

    def to(self, device: str):
        """
        Moves the parameters of the layer to the specified device.

        Parameters
        ----------
        device : str
            The device to move the parameters to.

        Returns
        -------
        Linear
            The layer with the parameters moved to the specified device.
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        w = np.random.uniform(
            low=-1.,
            high=1.,
            size=(out_channels, in_channels, kernel_size, kernel_size)
        ) / np.sqrt(in_channels*out_channels*kernel_size*kernel_size)
        self.params["Conv2dW"] = Tensor(w)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Performs a 2D convolution on the input tensor and Conv2d parameters and returns the output tensor.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The output tensor.
        """
        output = x.conv2d(self.params["Conv2dW"], self.padding, self.stride)
        return output

    def __str__(self) -> str:
        return f"Conv2d({self.params['Conv2dW'].data.shape[1]},{self.params['Conv2dW'].data.shape[0]},{self.kernel_size}, {self.stride},{self.padding})"

    def to(self, device: str):
        """
        Moves the parameters of the layer to the specified device.
        Parameters
        ----------
        device : str
            The device to move the parameters to.
        Returns
        -------
        Linear
            The layer with the parameters moved to the specified device.
        """
        for name, p in self.params.items():
            self.params[name] = p.to(device)
        return self


class ReShape():
    """
    Layer for reshaping the input tensor to the specified shape

    Attributes
    ----------
    shape (tuple): shape to reshape the input tensor to
    """

    def __init__(self, shape: tuple) -> None:
        self.shape = shape

    def __call__(self, x: Tensor) -> Tensor:
        """
        Reshapes the input tensor to the specified shape
        Parameters
        ----------
        x : Tensor
            The input tensor.
        Returns
        -------
        Tensor
            The output tensor.
        """
        return x.reshape(self.shape)

    def __str__(self) -> str:
        return f"ReShape{self.shape}"


class ReLU():
    """
    Layer for applying the ReLU activation function
    """

    def __call__(self, x: Tensor) -> Tensor:
        """
        Applies the ReLU activation function to the input tensor
        Parameters
        ----------
        x : Tensor
            The input tensor.
        Returns
        -------
        Tensor
            The output tensor.
        """
        return x.relu()
    
    def __str__(self) -> str:
        return "ReLU"


class LogSoftmax():
    """
    Layer for applying the LogSoftmax activation function
    """

    def __call__(self, x: Tensor) -> Tensor:
        """
        Applies the LogSoftmax activation function to the input tensor
        Parameters
        ----------
        x : Tensor
            The input tensor.
        Returns
        -------
        Tensor
            The output tensor.
        """
        return x.logsoftmax()

    def __str__(self) -> str:
        return "LogSoftmax"
    
class Softmax():
    """
    Layer for applying the Softmax activation function
    """

    def __call__(self, x: Tensor) -> Tensor:
        """
        Applies the Softmax activation function to the input tensor
        Parameters
        ----------
        x : Tensor
            The input tensor.
        Returns
        -------
        Tensor
            The output tensor.
        """
        return x.softmax()

    def __str__(self) -> str:
        return "Softmax"
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
        Parameters
        ----------
        x : Tensor
            The input tensor.
        Returns
        -------
        Tensor
            The output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass on the input tensor through all the layers and returns the output tensor.
        Parameters
        ----------
        x : Tensor
            The input tensor.
        Returns
        -------
        Tensor
            The output tensor.
        """
        return self.forward(x)

    def save(self, dir_path: str, save_arc: bool = True):
        """
        Saves the model parameters to the specified directory.

        Parameters
        ----------
        dir_path : str
            The directory to save the model parameters to.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for name, p in layer.params.items():
                    np.save(os.path.join(
                        dir_path, f"layer_{i}_{name}.npy"), p.data)
        # Export model architecture to a json file
        if save_arc:
            with open(os.path.join(dir_path, "model.arc"), "w") as f:
                f.write(str(self))

    def load(self, dir_path: str):
        """
        Loads the model parameters from the specified directory.

        Parameters
        ----------
        dir_path : str
            The directory to load the model parameters from.
        """
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for name in layer.params.keys():
                    layer.params[name] = np.load(
                        os.path.join(
                            dir_path,
                            f"layer_{i}_{name}.npy"
                        ),
                        allow_pickle=True
                    )

    def __str__(self) -> str:
        return "\n".join([str(l) for l in self.layers])

    def step(self, lr, optimizer: Optimizer, weight_decay: Optional[float] = 0.0):
        """
        Calls the step function of the optimizer on the parameters of all the layers.
        Parameters
        ----------
        lr : float
            The learning rate for the optimizer.
        optimizer : Optimizer
            The optimizer to use for updating the parameters.
        weight_decay : float, optional, default=0.0
            The weight decay (L2 regularization penalty) for the optimizer.
        """
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.step(lr=lr, weight_decay=weight_decay,
                           optimizer=optimizer)

    def to(self, device: str):
        """
        Moves the parameters of the model to the specified device.
        Parameters
        ----------
        device : str
            The device to move the parameters to.
        Returns
        -------
        Sequential
            The model with the parameters moved to the specified device.
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

def load_from_arc(model_dir: str):
    """
    Load model architecture from a model directory

    Parameters
    ----------
    model_dir : str
        The directory to load the model architecture from.

    Returns
    -------
    Sequential
        The model with the architecture loaded from the specified directory.
    """
    layers = []
    with open(os.path.join(model_dir, "model.arc"), "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Linear"):
                in_features, out_features = map(int, line[7:-1].split(","))
                layers.append(Linear(in_features, out_features))
            elif line.startswith("Conv2d"):
                in_channels, out_channels, kernel_size, stride, padding = map(
                    int, line[7:-1].split(","))
                layers.append(Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding))
            elif line.startswith("ReShape"):
                shape = tuple(map(int, line[7:-1].split(",")))
                layers.append(ReShape(shape))
            elif line == "ReLU":
                layers.append(ReLU())
            elif line == "LogSoftmax":
                layers.append(LogSoftmax())
            elif line == "Softmax":
                layers.append(Softmax())
    model = Sequential(*layers)
    model.load(model_dir)
    return model