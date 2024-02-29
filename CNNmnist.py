from deepdive.tensor import Tensor
import deepdive.nn as nn
from deepdive.datautils import DataLoader
import numpy as np
from datasets import load_dataset, Dataset
from tqdm import trange

mnist: Dataset = load_dataset('mnist')
dataloader = DataLoader(mnist["train"], batch_size=32)

model: nn.Sequential = nn.Sequential(
    nn.Conv2d(1, 4, 3, 1, 0),
    nn.ReLU(),
    nn.Conv2d(4, 8, 3, 1, 0),
    nn.ReLU(),
    nn.ReShape((-1, 8*24*24)),
    nn.Linear(4608, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10),
    nn.LogSoftmax()
)


def train(num_epochs: int = 10):
    for i, (inp, targets) in enumerate(dataloader):
        inp = Tensor(np.expand_dims(inp, axis=1))
        targets = Tensor(np.eye(10)[targets])
        output = model.forward(inp)
        loss = output.mul(targets).mean()
        loss.backward()
        if i == 0:
            loss.draw_graph(locals())
        model.step(lr = 0.01, weight_decay=0.0001, optimizer=nn.SGD)
        print(f"loss {loss.data}")
        if i == num_epochs:
            break

def test_accuracy(num_tests = 100):
    accuracy = []
    for i in range(num_tests):
        inp, targets = mnist["test"][i]['image'], mnist["test"][i]['label']
        inp = np.array(inp)
        inp = Tensor(np.expand_dims(np.expand_dims(inp, axis=0), axis=0))
        output = model.forward(inp)
        prediction = np.argmax(output.data)
        accuracy.append(prediction == targets)
    accuracy = np.array(accuracy)
    accuracy = accuracy.mean()
    return accuracy

train(10)
print(test_accuracy())