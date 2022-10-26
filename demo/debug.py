import sys
import os
os.chdir("../")

import numpy as np
import needle as ndl
import needle.nn as nn
from mlp_resnet import *

def test_mnist_dataset():
    # Test dataset sizing
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz")
    assert len(mnist_train_dataset) == 60000

    sample_norms = np.array([np.linalg.norm(mnist_train_dataset[idx][0]) for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_against = np.array([10.188792, 6.261355, 8.966858, 9.4346485, 9.086626, 9.214664, 10.208544, 10.649756])
    sample_labels = np.array([mnist_train_dataset[idx][1] for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_labels = np.array([0,7,0,5,9,7,7,8])

    np.testing.assert_allclose(sample_norms, compare_against)
    np.testing.assert_allclose(sample_labels, compare_labels)

    mnist_train_dataset = ndl.data.MNISTDataset("data/t10k-images-idx3-ubyte.gz",
                                                "data/t10k-labels-idx1-ubyte.gz")
    assert len(mnist_train_dataset) == 10000

    sample_norms = np.array([np.linalg.norm(mnist_train_dataset[idx][0]) for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_against = np.array([9.857545, 8.980832, 8.57207 , 6.891522, 8.192135, 9.400087, 8.645003, 7.405202])
    sample_labels = np.array([mnist_train_dataset[idx][1] for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_labels = np.array([2, 4, 9, 6, 6, 9, 3, 1])

    np.testing.assert_allclose(sample_norms, compare_against, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sample_labels, compare_labels)

    # test a transform
    np.random.seed(0)
    tforms = [ndl.data.RandomCrop(28), ndl.data.RandomFlipHorizontal()]
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz",
                                                transforms=tforms)

    sample_norms = np.array([np.linalg.norm(mnist_train_dataset[idx][0]) for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_against = np.array([2.0228338 ,0.        ,7.4892044 ,0.,0.,3.8012788,9.583429,4.2152724])
    sample_labels = np.array([mnist_train_dataset[idx][1] for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_labels = np.array([0,7,0,5,9,7,7,8])

    np.testing.assert_allclose(sample_norms, compare_against, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sample_labels, compare_labels)


    # test a transform
    tforms = [ndl.data.RandomCrop(12), ndl.data.RandomFlipHorizontal(0.4)]
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz",
                                                transforms=tforms)
    sample_norms = np.array([np.linalg.norm(mnist_train_dataset[idx][0]) for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_against = np.array([5.369537, 5.5454974, 8.966858, 7.547235, 8.785921, 7.848442, 7.1654058, 9.361828])
    sample_labels = np.array([mnist_train_dataset[idx][1] for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_labels = np.array([0,7,0,5,9,7,7,8])

    np.testing.assert_allclose(sample_norms, compare_against, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sample_labels, compare_labels)


def test_dataloader_mnist():
    batch_size = 1
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz")
    mnist_train_dataloader = ndl.data.DataLoader(dataset=mnist_train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)

    for i, batch in enumerate(mnist_train_dataloader):
        batch_x, batch_y = batch[0].numpy(), batch[1].numpy()
        truth = mnist_train_dataset[i * batch_size:(i + 1) * batch_size]
        truth_x = truth[0] if truth[0].shape[0] > 1 else truth[0].reshape(-1)
        truth_y = truth[1] if truth[1].shape[0] > 1 else truth[1].reshape(-1)

        np.testing.assert_allclose(truth_x, batch_x.flatten())
        np.testing.assert_allclose(batch_y, truth_y)

    batch_size = 5
    mnist_test_dataset = ndl.data.MNISTDataset("data/t10k-images-idx3-ubyte.gz",
                                               "data/t10k-labels-idx1-ubyte.gz")
    mnist_test_dataloader = ndl.data.DataLoader(dataset=mnist_test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)

    for i, batch in enumerate(mnist_test_dataloader):
        batch_x, batch_y = batch[0].numpy(), batch[1].numpy()
        truth = mnist_test_dataset[i * batch_size:(i + 1) * batch_size]
        truth_x = truth[0]
        truth_y = truth[1]

        np.testing.assert_allclose(truth_x, batch_x)
        np.testing.assert_allclose(batch_y, truth_y)



    noshuf = bat9 = ndl.data.DataLoader(dataset=mnist_test_dataset,
                                        batch_size=10,
                                        shuffle=False)
    shuf = bat9 = ndl.data.DataLoader(dataset=mnist_test_dataset,
                                      batch_size=10,
                                      shuffle=True)
    diff = False
    for i, j in zip(shuf, noshuf):
        if i != j:
            diff = True
            break
    assert diff, 'shuffling had no effect on the dataloader.'


def test_dataloader_ndarray():
    for batch_size in [1,10,100]:
        np.random.seed(0)

        train_dataset = ndl.data.NDArrayDataset(np.random.rand(100,10,10))
        train_dataloader = ndl.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

        for i, batch in enumerate(train_dataloader):
            batch_x = batch[0].numpy()
            truth_x = train_dataset[i * batch_size:(i + 1) * batch_size][0].reshape((batch_size,10,10))
            np.testing.assert_allclose(truth_x, batch_x)

    batch_size = 1
    np.random.seed(0)
    train_dataset = ndl.data.NDArrayDataset(np.arange(100,))
    train_dataloader = iter(ndl.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True))

    elements = np.array([next(train_dataloader)[0].numpy().item() for _ in range(10)])
    np.testing.assert_allclose(elements, np.array([26, 86,  2, 55, 75, 93, 16, 73, 54, 95]))

    batch_size = 10
    train_dataset = ndl.data.NDArrayDataset(np.arange(100,))
    train_dataloader = iter(ndl.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True))

    raw_result = [next(train_dataloader)[0].numpy().item() for _ in range(10)]
    expect_raw = [26, 86,  2, 55, 75, 93, 16, 73, 54, 95]
    elements = np.array([np.linalg.norm(data) for data in raw_result])
    expect_norm = np.array([np.linalg.norm(data) for data in expect_raw])
    np.testing.assert_allclose(elements, np.array([164.805946, 173.43875 , 169.841102, 189.050258, 195.880065, 206.387984, 209.909504, 185.776748, 145.948621, 160.252925]))


def mlp_resnet_forward(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob):
    np.random.seed(4)
    input_tensor = ndl.Tensor(np.random.randn(2, dim), dtype=np.float32)
    output_tensor = MLPResNet(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob)(input_tensor)
    return output_tensor.numpy()


def test_mlp_resnet_forward_1():
    np.testing.assert_allclose(
        mlp_resnet_forward(10, 5, 2, 5, nn.LayerNorm1d, 0.5),
        np.array([[3.046162, 1.44972, -1.921363, 0.021816, -0.433953],
                  [3.489114, 1.820994, -2.111306, 0.226388, -1.029428]],
                 dtype=np.float32),
        rtol=1e-5,
        atol=1e-5)


def test_mlp_resnet_forward_2():
    np.testing.assert_allclose(
        mlp_resnet_forward(15, 25, 5, 14, nn.BatchNorm1d, 0.0),
        np.array([[
            0.92448235, -2.745743, -1.5077105, 1.130784, -1.2078242,
            -0.09833566, -0.69301605, 2.8945382, 1.259397, 0.13866742,
            -2.963875, -4.8566914, 1.7062538, -4.846424
        ],
            [
                0.6653336, -2.4708004, 2.0572243, -1.0791507, 4.3489094,
                3.1086435, 0.0304327, -1.9227124, -1.416201, -7.2151937,
                -1.4858506, 7.1039696, -2.1589825, -0.7593413
            ]],
            dtype=np.float32),
        rtol=1e-5,
        atol=1e-5)


def train_epoch_1(hidden_dim, batch_size, optimizer, **kwargs):
    np.random.seed(1)
    train_dataset = ndl.data.MNISTDataset(\
            "./data/train-images-idx3-ubyte.gz",
            "./data/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), **kwargs)
    model.eval()
    return np.array(epoch(train_dataloader, model, opt))


def eval_epoch_1(hidden_dim, batch_size):
    np.random.seed(1)
    test_dataset = ndl.data.MNISTDataset(\
            "./data/t10k-images-idx3-ubyte.gz",
            "./data/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size,
             shuffle=False)

    model = MLPResNet(784, hidden_dim)
    model.train()
    return np.array(epoch(test_dataloader, model))


def train_mnist_1(batch_size, epochs, optimizer, lr, weight_decay, hidden_dim):
    np.random.seed(1)
    out = train_mnist(batch_size, epochs, optimizer, lr, weight_decay, hidden_dim, data_dir="./data")
    return np.array(out)


def test_mlp_train_epoch_1():
    np.testing.assert_allclose(train_epoch_1(5, 250, ndl.optim.Adam, lr=0.01, weight_decay=0.1),
        np.array([0.675267, 1.84043]), rtol=0.0001, atol=0.0001)

def test_mlp_eval_epoch_1():
    np.testing.assert_allclose(eval_epoch_1(10, 150),
        np.array([0.9164 , 4.137814]), rtol=1e-5, atol=1e-5)

def test_mlp_train_mnist_1():
    np.testing.assert_allclose(train_mnist_1(250, 2, ndl.optim.SGD, 0.001, 0.01, 100),
        np.array([0.4875 , 1.462595, 0.3245 , 1.049429]), rtol=0.001, atol=0.001)


if __name__ == "__main__":
    test_mlp_train_mnist_1()
