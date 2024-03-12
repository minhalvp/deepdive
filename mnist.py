import numpy as np
from deepdive.tensor import Tensor
import deepdive.nn as nn
from tqdm import trange
from datasets import load_dataset
from deepdive.dash import Dash

mnist = load_dataset('mnist')


def convert_to_np(example):
    example['np_image'] = np.asarray(example['image'])
    return example


mnist = mnist.map(convert_to_np)

X_train, Y_train = np.asarray(
    mnist['train']['np_image']), np.asarray(mnist['train']['label'])
X_test, Y_test = np.asarray(
    mnist['test']['np_image']), np.asarray(mnist['test']['label'])

lr = 0.01
BS = 64

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.LayerNorm(128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.LayerNorm(10),
    nn.LogSoftmax()
)

# dash = Dash(2)

for i in (t := trange(1000)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    x = Tensor(X_train[samp].reshape((-1, 28*28)))
    Y = Y_train[samp]
    y = np.zeros((len(samp), 10), np.float32)
    y[range(y.shape[0]), Y] = -1.0
    y = Tensor(y)
    output = model.forward(x)
    x = output.mul(y)
    x = x.mean()
    x.backward()

    loss = x.data
    cat = np.argmax(output.data, axis=1)
    accuracy = (cat == Y).mean()
    # dash.update(loss, accuracy)
    t.set_description(f"loss {loss} accuracy {accuracy}")
    # dash.plot()
    # SGD
    model.step(lr=lr, optimizer=nn.SGD)

# Model Saving
model.save('mnist_model')
