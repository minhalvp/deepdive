import numpy as np
from tensor import Tensor
from nn import Linear
from tqdm import trange
from datasets import load_dataset

mnist = load_dataset('mnist')

def convert_to_np(example):
    example['np_image'] = np.array(example['image'])
    return example
mnist = mnist.map(convert_to_np)
# this next line is terribly slow for some reason
X_train, Y_train, X_test, Y_test = np.array(mnist['train']['np_image']), np.array(mnist['train']['label']), np.array(mnist['test']['np_image']), np.array(mnist['test']['label'])

lr = 0.01
BS = 64

l1 = Linear(784, 128)
l2 = Linear(128, 10)

losses, accuracies = [], []
for i in (t := trange(1000)):
  samp = np.random.randint(0, X_train.shape[0], size=(BS))  
  x = Tensor(X_train[samp].reshape((-1, 28*28)))
  Y = Y_train[samp]
  y = np.zeros((len(samp),10), np.float32)
  y[range(y.shape[0]),Y] = -1.0
  y = Tensor(y)
  x = l1(x)
  x = x.relu()
  x = x_l2 = l2(x)
  x = x.logsoftmax()
  x = x.mul(y)
  x = x.mean()
  x.backward()
  
  loss = x.data

  cat = np.argmax(x_l2.data, axis=1)
  accuracy = (cat == Y).mean()
  
  # SGD
  l1.step(lr)
  l2.step(lr)
  accuracies.append(accuracy)
  t.set_description(f"loss {loss} accuracy {accuracy}")

# numpy forward pass
def forward(x):
  x = l1(x)
  x.data = np.maximum(x.data, 0)
  x = l2(x)
  return x.data


def numpy_eval():
  x = Tensor(X_test.reshape((-1, 28*28)))
  print(x.data.shape)
  Y_test_preds_out = forward(x)
  Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
  return (Y_test == Y_test_preds).mean()

print(f"test set accuracy is {numpy_eval()}")