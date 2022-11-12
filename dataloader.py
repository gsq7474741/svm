import numpy as np
import torchvision


def linear_data(data_len):
    data_len = 100
    x = 2 * (np.linspace(0, 10, data_len)) + 2 + np.random.uniform(-0.1, 0.1, (data_len,))
    x[data_len // 2:] += 3

    x = x.reshape(data_len, -1)
    x = np.concatenate((x, np.linspace(0, 10, data_len).reshape(data_len, 1)), axis=1)
    y = np.concatenate([np.ones(data_len // 2), -np.ones(data_len // 2)])
    train_data = np.concatenate([x, y.reshape(data_len, -1)], axis=1)
    np.random.shuffle(train_data)
    x = train_data[:, :-1].reshape(data_len, -1)
    y = train_data[:, -1]

    return x, y


def minst(train_size=200, test_size=50):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_data = train_set.data.numpy().reshape(-1, 28 * 28)
    train_label = train_set.targets.numpy()
    train = np.random.choice(np.arange(0, len(train_data), 1), size=train_size)
    train_data = train_data[train]
    train_label = train_label[train]

    test_data = test_set.data.numpy().reshape(-1, 28 * 28)
    test_label = test_set.targets.numpy()
    test = np.random.choice(np.arange(0, len(test_data), 1), size=test_size)
    test_data = test_data[test]
    test_label = test_label[test]

    # [n,d] [n,] [m,d] [m]
    return train_data, train_label, test_data, test_label


if __name__ == '__main__':
    minst()
