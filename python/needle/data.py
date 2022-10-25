import numpy as np

import needle
from .autograd import Tensor

from typing import Iterator, Optional, List, Tuple
import struct
import gzip
import random

import numpy as array_api


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return array_api.fliplr(img)
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        pad_img = array_api.pad(img, [(self.padding, ), (self.padding, ), (0, )])
        x_start = self.padding + shift_x
        x_end = img.shape[0] + self.padding + shift_x
        
        y_start = self.padding + shift_y
        y_end = img.shape[1] + self.padding + shift_y
        
        return pad_img[x_start:x_end, y_start:y_end]
        ### END YOUR SOLUTION


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
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            index_range = np.arange(len(self.dataset))
            np.random.shuffle(index_range)
            self.ordering = np.array_split(index_range,
                                           range(self.batch_size, len(self.dataset), self.batch_size))

        self.current_index = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.current_index >= len(self.ordering):
            raise StopIteration
        else:
            batch_ids = self.ordering[self.current_index]
            self.current_index += 1
            batch = self.dataset[batch_ids]
            if not isinstance(batch, Tuple):
                batch = (batch, )
            result = tuple([trans_to_tensor(data) for data in batch])
            return result
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        x, y = parse_mnist(image_filename, label_filename)
        self.x = x
        self.y = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        item_x = self.x[index]
        item_y = self.y[index]
        if self.transforms:
            trans_x = self.apply_transforms(item_x)
        else:
            trans_x = item_x
        return trans_x, item_y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.x.shape[0]
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


def trans_to_tensor(x) -> needle.Tensor:
    if isinstance(x, needle.Tensor):
        return x
    else:
        return needle.Tensor(x)

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
        images = np.fromstring(image_f.read(), dtype=np.uint8).reshape(num, rows, cols, 1).astype(np.float32)
        # max = np.max(images)
        max = 255
        norm_images = images/max
        
    return (norm_images, labels)
    ### END YOUR SOLUTION