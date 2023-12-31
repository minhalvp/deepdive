import numpy as np
import torch
from deepdive.tensor import Tensor
from datasets import load_dataset

def test_mul():
    x_init = np.random.randn(3, 4).astype(np.float32)
    y_init = np.random.randn(3, 4).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)
    Y, y = Tensor(y_init), torch.tensor(y_init, requires_grad=True)

    Z = X.mul(Y)
    Z.mean().backward()
    z = x.mul(y)
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(X.grad, x.grad, atol=1e-6)
    np.testing.assert_allclose(Y.grad, y.grad, atol=1e-6)

def test_add():
    x_init = np.random.randn(3, 4).astype(np.float32)
    y_init = np.random.randn(3, 4).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)
    Y, y = Tensor(y_init), torch.tensor(y_init, requires_grad=True)

    Z = X.add(Y)
    Z.mean().backward()
    z = x.add(y)
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(X.grad, x.grad, atol=1e-6)
    np.testing.assert_allclose(Y.grad, y.grad, atol=1e-6)

def test_conv2d():
    x_init = np.random.randn(1, 1, 28, 28).astype(np.float32)
    w_init = np.random.randn(1, 1, 3, 3).astype(np.float32)
    # x_init = np.random.randn(1, 3, 32, 32).astype(np.float32)
    # w_init = np.random.randn(3, 1, 5, 5).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)
    Y, y = Tensor(w_init), torch.tensor(w_init, requires_grad=True)

    Z = X.conv2d(Y, 1, 1)
    Z.mean().backward()
    z = torch.nn.functional.conv2d(x, y, padding=1, stride=1)
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(Y.grad, y.grad, atol=1e-6)

def test_sum():
    x_init = np.random.randn(3, 4).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)

    Z = X.sum()
    Z.mean().backward()
    z = x.sum()
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(X.grad, x.grad, atol=1e-6)