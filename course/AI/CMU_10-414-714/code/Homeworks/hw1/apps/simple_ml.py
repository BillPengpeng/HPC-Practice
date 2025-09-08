"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, 'rb') as file:
      image_array = np.frombuffer(file.read(), dtype=np.uint8)
    with gzip.open(label_filename, 'rb') as file:
      label_array = np.frombuffer(file.read(), dtype=np.uint8)
    num_examples = label_array.shape[0] - 8
    X = image_array[16:].reshape(-1, 784).astype(np.float32)
    Y = label_array[8:].astype(np.uint8)
    X = X / 255.0
    return X, Y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size = Z.shape[0]
    # sum_val = 0
    #for idx in range(batch_size):
      #sum_val += (np.log(np.exp(Z[idx,:]).sum()) - np.sum(Z * y))
    sum_val = ndl.ops.log(ndl.ops.summation(ndl.ops.exp(Z), axes=(1))) - ndl.ops.summation(Z * y_one_hot, axes=(1))
    #print(sum_val, ndl.ops.summation(sum_val, axes=0))
    return ndl.ops.summation(sum_val) / batch_size;
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    for start in range(0, num_examples, batch):
      end = min(num_examples, start + batch)
      #print("start:", start)
      X_batch = ndl.Tensor(X[start:end, :])
      y_batch = ndl.Tensor(y[start:end])
      Z1 = ndl.ops.relu(ndl.ops.matmul(X_batch, W1)) 
      Z2 = ndl.ops.matmul(Z1, W2) 
      S = softmax_loss(Z2, ndl.init.one_hot(num_classes, y_batch))
      S.backward()
      #scale = lr / (end - start)
      #W1 = W1 - ndl.ops.mul_scalar(ndl.ops.matmul(ndl.ops.transpose(X_batch), Z1.grad), scale)
      #W2 = W2 - ndl.ops.mul_scalar(ndl.ops.matmul(ndl.ops.transpose(Z1), Z2.grad), scale) 
      W1 = W1 - lr * W1.grad.realize_cached_data()
      W2 = W2 - lr * W2.grad.realize_cached_data()
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
