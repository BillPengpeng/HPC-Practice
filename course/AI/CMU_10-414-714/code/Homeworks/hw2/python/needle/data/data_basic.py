import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n = len(dataset)
        if not self.shuffle:
          self.ordering = np.array_split(np.arange(len(dataset)), 
                     range(batch_size, len(dataset), batch_size))
          self.ordering_size = len(self.ordering)

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.index = 0
        if self.shuffle:
          indexes = np.arange(self.n)
          np.random.shuffle(indexes)
          self.ordering = np.array_split(indexes,range(self.batch_size, self.n, self.batch_size))
        self.ordering_size = len(self.ordering)
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        # raise StopIteration
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!:", self.index)
        if self.index == self.ordering_size:
          raise StopIteration

        res = [Tensor(x) for x in self.dataset[self.ordering[self.index]]]
        self.index += 1
        
        return tuple(res)
        ### END YOUR SOLUTION

