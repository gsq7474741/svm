import matplotlib.pyplot as plt

from psmo import *
import numpy as np
import seaborn as sns

from mertics_np import *

np.random.seed(42)

if __name__ == '__main__':
    data_len = 200
    c = 1
    tol = 1e-3
    max_passes = 5
    std = 1
    # kernel = LinearKernel()

    x = 2 * (np.linspace(0, 10, data_len)) + 2 + np.random.uniform(-std, std, (data_len,))
    x[data_len // 2:] -= 5
    x = x.reshape(data_len, -1)
    x = np.concatenate((x, np.linspace(0, 1, data_len).reshape(data_len, 1)), axis=1)
    x[data_len // 2:, 1] += 1

    y = np.concatenate([np.ones(data_len // 2), -np.ones(data_len // 2)])
    train_data = np.concatenate([x, y.reshape(data_len, -1)], axis=1)

    np.random.shuffle(train_data)

    x = train_data[:, :-1].reshape(data_len, -1)
    y = train_data[:, -1]

    # maps = {'1': 1.0, '9': -1.0}
    # data, label = loadImage("digits/trainingDigits", maps)
    # test, testLabel = loadImage("digits/testDigits", maps)

    # 训练
    smo = SMO(x, y, 1, 0.0001, 5, name='linear', theta=20)
    begin = datetime.datetime.now()
    smo.fit()
    end = datetime.datetime.now()
    time_sub = end - begin
    print("keneral_svm time", time_sub.total_seconds())
    print(f'SV num: {len(smo.SVIndex)}')

    # 预测
    testResult = smo.predict(x)
    ax = sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y)
    plt.scatter(x=x[smo.SVIndex, 0], y=x[smo.SVIndex, 1], c='none', marker='o', edgecolors='g', s=70)
    # plt.show()
    print(f"train acc: {accuracy(testResult, y):.3%}")
    # m = shape(x)[0]
    # count = 0.0
    # for i in range(m):
    #     if y[i] != testResult[i]:
    #         count += 1
    # print("classfied error rate is:", count / m)
    smo.calcw()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    axisx = np.linspace(xlim[0], xlim[1], 100)
    axisy = np.linspace(ylim[0], ylim[1], 100)
    axisy, axisx = np.meshgrid(axisy, axisx)

    xy = np.vstack([axisx.ravel(), axisy.ravel()]).T
    z = smo.forward(xy).reshape(axisx.shape)
    plt.contour(axisx, axisy, z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.show()
