import numpy as np
from tensor import Tensor
from nn import Linear
from tqdm import trange
from datasets import load_dataset

# load the mnist dataset
mnist = load_dataset('mnist')
# convert the images to numpy arrays
def convert_to_np(example):
    example['np_image'] = np.array(example['image'])
    return example
mnist = mnist.map(convert_to_np)
# this next line is terribly slow for some reason
X_train, Y_train, X_test, Y_test = np.array(mnist['train']['np_image']), np.array(mnist['train']['label']), np.array(mnist['test']['np_image']), np.array(mnist['test']['label'])
  
def layer_init(m, h):
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)

def bias_init(h):
  return np.zeros((h,), dtype=np.float32)

# The problem comes when adding the bias 
lr = 0.01
BS = 64


l1 = Tensor(layer_init(28*28, 128))
l2 = Tensor(layer_init(128, 10))
b1 = Tensor(bias_init(128))
b2 = Tensor(bias_init(10))
losses, accuracies = [], []
for i in (t := trange(1000)):
  samp = np.random.randint(0, X_train.shape[0], size=(BS))  
  x = Tensor(X_train[samp].reshape((-1, 28*28)))
  Y = Y_train[samp]
  y = np.zeros((len(samp),10), np.float32)
  y[range(y.shape[0]),Y] = -1.0
  y = Tensor(y)
  x = x.dot(l1)
  x = x.add(b1)
  x = x.relu()
  x = x.dot(l2)
  x = x_l2 = x.add(b2)
  x = x.logsoftmax()
  x = x.mul(y)
  x = x.mean()
  x.backward()
  
  loss = x.data

  cat = np.argmax(x_l2.data, axis=1)
  accuracy = (cat == Y).mean()
  
  # SGD
  l1.data -= lr * l1.grad
  l2.data -= lr * l2.grad
  losses.append(loss)
  accuracies.append(accuracy)
  t.set_description(f"loss {loss} accuracy {accuracy}")

# numpy forward pass
def forward(x):
  x = x.dot(l1.data)
  x = np.maximum(x, 0)
  x = x.dot(l2.data)
  return x

def numpy_eval():
  Y_test_preds_out = forward(X_test.reshape((-1, 28*28)))
  Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
  return (Y_test == Y_test_preds).mean()

print(f"test set accuracy is {numpy_eval()}")

