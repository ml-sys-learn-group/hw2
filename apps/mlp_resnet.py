import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    ### base block = linear + norm + relu + dropout + linear + norm
    base_block = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )

    residual = nn.Residual(base_block)

    return nn.Sequential(residual, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    ### first linear  (dim, hidden_dim)
    seq_modules = [ nn.Flatten() ] # flatten the input data
    fir_linear = nn.Linear(dim, hidden_dim)
    seq_modules.append(fir_linear)
    for i in range(num_blocks):
        residual_block = ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob)
        seq_modules.append(residual_block)

    # last linear
    last_linear = nn.Linear(hidden_dim, num_classes)
    seq_modules.append(last_linear)

    return nn.Sequential(*seq_modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
    loss_func = nn.SoftmaxLoss()
    loss = 0.0
    acc = 0.0
    step = 0
    for x, y in dataloader:
        pred = model(x)
        loss = loss_func(pred, y)
        loss += loss.numpy()
        acc += cal_acc(pred, y)
        step += 1

        if opt:
            loss.backward()
            opt.step()

    mean_loss = loss/step
    mean_acc = acc/step
    return mean_loss, mean_acc
    ### END YOUR SOLUTION

def cal_acc(pred, y):
    """
    Args:
        pred:
        y:

    Returns:
    """
    return np.mean(pred.numpy().argmax(axis=1) == y.numpy())


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    mnist_train_dataset = ndl.data.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz",
                                                f"{data_dir}/train-labels-idx1-ubyte.gz")
    mnist_train_dataloader = ndl.data.DataLoader(dataset=mnist_train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)

    test_batch_size = 5
    mnist_test_dataset = ndl.data.MNISTDataset("data/t10k-images-idx3-ubyte.gz",
                                               "data/t10k-labels-idx1-ubyte.gz")
    mnist_test_dataloader = ndl.data.DataLoader(dataset=mnist_test_dataset,
                                                batch_size=test_batch_size,
                                                shuffle=False)

    dim = 784
    model = MLPResNet(dim=dim, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(epochs):
        print(f"epoch: {i}")
        # training
        train_loss, train_acc = epoch(mnist_train_dataloader, model, opt)
        print(f"train loss: {train_loss}, train acc: {train_acc}")

        # evaluation
        eval_loss, eval_acc = epoch(mnist_test_dataloader, model)
        print(f"eval loss: {eval_loss}, eval acc: {eval_acc}")




    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
