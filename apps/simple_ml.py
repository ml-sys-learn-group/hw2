import struct
import gzip
import numpy as np
import math

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
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
    # load labels
    with gzip.open(label_filename, 'rb') as label_f:
        magic, n = struct.unpack('>II', label_f.read(8))
        labels = np.fromstring(label_f.read(), dtype=np.uint8)
        
    # load image data
    with gzip.open(image_filesname, 'rb') as image_f:
        magic, num, rows, cols = struct.unpack('>IIII',image_f.read(16))
        # sample num should be equal to label num
        assert num == n
        images = np.fromstring(image_f.read(), dtype=np.uint8).reshape(num, rows*cols).astype(np.float32)
        # max = np.max(images)
        max = 255
        norm_images = images/max
        
    return (norm_images, labels)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
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
    z_exp = ndl.exp(Z)
    z_sum = ndl.summation(z_exp, axes=1).reshape((-1, 1))
    z_softmax = z_exp/ndl.broadcast_to(z_sum, z_exp.shape)
    z_pos = z_softmax * y_one_hot
    loss = -ndl.log(ndl.summation(z_pos, axes=1))
    mean_loss = ndl.summation(loss)/Z.shape[0]
    return mean_loss
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
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
    example_num = y.shape[0]
    batch_num = int(math.ceil(example_num/batch))
    hist_loss = 0.0

    for i in range(batch_num):
        x_batch, y_batch = get_batch(X, y, batch, i)
        x_tensor = ndl.Tensor(x_batch, requires_grad=False)
        logits = ndl.matmul(ndl.relu(ndl.matmul(x_tensor, W1)), W2)
        y_one_hot = np.zeros((y_batch.shape[0], logits.shape[-1]))
        y_one_hot[np.arange(y_batch.size), y_batch] = 1
        y_tensor = ndl.Tensor(y_one_hot)
        loss = softmax_loss(logits, y_tensor)
        hist_loss += loss.numpy()
        loss.backward()

        if i % 10 == 0:
            print("batch step: ", i)
            print("loss: ", hist_loss/(i+1))

        # apply gradient
        W1 = (W1 - W1.grad * lr).data
        W2 = (W2 - W2.grad * lr).data

    return W1, W2
    ### END YOUR SOLUTION


def get_batch(x, y, batch_size, batch_step):
    """
    Args:
        x:
        y:
        batch_size:
        batch_step:
    Returns:
    """
    start = batch_step * batch_size
    end = min(start + batch_size, y.shape[0])
    return x[start:end, ], y[start:end, ]

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)