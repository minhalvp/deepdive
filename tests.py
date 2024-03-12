import numpy as np
import torch
from deepdive.tensor import Tensor


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
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)
    Y, y = Tensor(w_init), torch.tensor(w_init, requires_grad=True)

    Z = X.conv2d(Y, 1, 1)
    Z.mean().backward()
    z = torch.nn.functional.conv2d(x, y, padding=1, stride=1)
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(Y.grad, y.grad, atol=1e-5)


def test_sum():
    x_init = np.random.randn(3, 4).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)

    Z = X.sum()
    Z.mean().backward()
    z = x.sum()
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(X.grad, x.grad, atol=1e-6)


def test_relu():
    x_init = np.random.randn(3, 4).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)

    Z = X.relu()
    Z.mean().backward()
    z = torch.nn.functional.relu(x)
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(X.grad, x.grad, atol=1e-6)


def test_dot():
    x_init = np.random.randn(3, 4).astype(np.float32)
    y_init = np.random.randn(4, 5).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)
    Y, y = Tensor(y_init), torch.tensor(y_init, requires_grad=True)

    Z = X.dot(Y)
    Z.mean().backward()
    z = x.matmul(y)
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(X.grad, x.grad, atol=1e-6)
    np.testing.assert_allclose(Y.grad, y.grad, atol=1e-6)


def test_logsoftmax():
    x_init = np.random.randn(3, 4).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)

    Z = X.logsoftmax()
    Z.mean().backward()
    z = torch.nn.functional.log_softmax(x, dim=-1)
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(X.grad, x.grad, atol=1e-6)


def test_softmax():
    x_init = np.random.randn(1, 10).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)

    Z = X.softmax()
    Z.mean().backward()
    z = torch.nn.functional.softmax(x, dim=-1)
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(X.grad, x.grad, atol=1e-6)


def test_exponential():
    x_init = np.random.randn(3, 4).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)

    Z = X.exp()
    Z.mean().backward()
    z = torch.exp(x)
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(X.grad, x.grad, atol=1e-6)


def test_log():
    x_init = np.random.rand(3, 4).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)

    Z = X.log()
    Z.mean().backward()
    z = torch.log(x)
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(X.grad, x.grad, atol=1e-6)

def test_sigmoid():
    x_init = np.random.randn(3, 4).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)

    Z = X.sigmoid()
    Z.mean().backward()
    z = torch.sigmoid(x)
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(X.grad, x.grad, atol=1e-6)

def test_layernorm():
    x_init = np.random.randn(3, 4).astype(np.float32)
    X, x = Tensor(x_init), torch.tensor(x_init, requires_grad=True)

    Z = X.layernorm()
    Z.mean().backward()
    z = torch.nn.functional.layer_norm(x, x.size()[1:])
    z.mean().backward()

    np.testing.assert_allclose(Z.data, z.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(X.grad, x.grad, atol=1e-6)