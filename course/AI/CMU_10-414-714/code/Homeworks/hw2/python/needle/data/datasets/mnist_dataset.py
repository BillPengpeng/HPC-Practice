from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import os, gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        assert os.path.exists(image_filename)
        assert os.path.exists(label_filename)
        with gzip.open(image_filename, 'rb') as file:
          self.image_array = np.frombuffer(file.read(), dtype=np.uint8)
        with gzip.open(label_filename, 'rb') as file:
          self.label_array = np.frombuffer(file.read(), dtype=np.uint8)
        self.num_examples = self.label_array.shape[0] - 8
        self.transforms = transforms
        self.images = self.image_array[16:].reshape(-1, 784).astype(np.float32)
        self.labels = self.label_array[8:].astype(np.uint8)
        self.images = self.images / 255.0
        ### END YOUR SOLUTION 

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # idx = index % self.num_examples
        # print(index, idx)
        # X = self.image_array[16+idx*784:16+(idx+1)*784].reshape(28, 28, 1).astype(np.float32)
        # X = X / 255.0
        # # print(X.shape)
        # Y = self.label_array[8+idx]
        # if self.transforms is not None:
        #   for proc in self.transforms:
        #     X = proc(X)
        # # print(self.image_array.shape, index, idx, Y, X[0, :5])
        # return X, Y
        # print("index:", index)

        # x = self.X[index]
        # y = self.Y[index]
        # n = len(x.shape)
        # if n == 1:
        #   # 单索引情形
        #   x = x.reshape(28, 28, -1)
        #   x = self.apply_transforms(x)
        #   x = x.reshape(-1, 28*28)
        # else:
        #   # 多索引情形
        #   m = x.shape[0]
        #   x = x.reshape(m, 28, 28, -1)
        #   for i in range(m):
        #     x[i] = self.apply_transforms(x[i])
        #   x = x.reshape(-1, 28*28)
        # return x, y
        # ### END YOUR SOLUTION

        ### BEGIN YOUR SOLUTION
        X, y = self.images[index], self.labels[index]
        if self.transforms:
          X_in = X.reshape((28, 28, -1))
          X_out = self.apply_transforms(X_in)
          return X_out.reshape(-1, 28 * 28), y
        else:
          return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return self.num_examples
        ### END YOUR SOLUTION