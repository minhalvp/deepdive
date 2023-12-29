from deepdive.tensor import Tensor
from datasets import DatasetDict, Dataset
import random
import numpy as np
# DataLoader
# Input:
# - dataset = HF_DATASET
# - batch_size = Int
# - Shuffle = Bool

class DataLoader:
    def __init__(self, dataset: DatasetDict, batch_size: int = 1, shuffle: bool = False) -> None:
        self.dataset = dataset
        self.bs = batch_size
        if shuffle:
            self.shuffle()
        self.batches = self.get_batches()
        self.batch_index = 0

    def get_batches(self):
        num_batches, last_bs = divmod(len(self.dataset), self.bs)
        batches = [self.dataset.select(range(i*self.bs, (i+1)*self.bs)) for i in range(num_batches)]
        if last_bs != 0:
            batches.append(self.dataset.select(range(num_batches*self.bs, num_batches*self.bs + last_bs)))
        return batches

    def shuffle(self):
        # Fisher-Yattes Shuffle Algorithm
        arr = self.dataset.to_list()
        for i in range(len(arr) - 1, 0, -1):
            j = random.randint(0, i)
            arr[i], arr[j] = arr[j], arr[i]
            
        self.dataset = Dataset.from_list(arr)
    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.batches)

    def __next__(self):
        if self.batch_index >= len(self.batches):
            raise StopIteration
        batch = self.batches[self.batch_index]
        self.batch_index += 1
        return_values = [batch[feature] for feature in batch.features.keys()]
        return_values = [np.array(return_value) for return_value in return_values]
        return *return_values,