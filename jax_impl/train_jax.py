import datetime
import os
import pickle

import matplotlib.pyplot as plt
import nni
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from tqdm import trange

import jax

from jax import numpy as jnp
from jax import jit, vmap

import mertics_jax as mertics
import model_jax as model
from dataloader import minst
# import psmo as model
from kernels_jax import LinearKernel, RBFKernel

# jax.default_device = 'cpu'
# np.random.seed(42)
# torch.random.manual_seed(42)
# torch.cuda.manual_seed_all(42)


def one_vs_one():
    # ovo dataset
    binary_set = [[[] for j in range(10)] for i in range(10)]
    for i in range(10):
        for j in range(10):
            data = train_data[jnp.where((train_label == i) | (train_label == j))]
            label = train_label[jnp.where((train_label == i) | (train_label == j))]
            label = jnp.where(label == i, 1, -1)
            binary_set[i][j] = [data, label]

    # pca
    if pca_inspect:
        pca_model = [[PCA(n_components=2) for i in range(10)] for j in range(10)]
        pca_data = [[[] for i in range(10)] for j in range(10)]
        for i in range(10):
            for j in range(i):
                # if i == j:
                #     continue
                pca_model[i][j].fit(binary_set[i][j][0])
                pca_data[i][j] = pca_model[i][j].transform(binary_set[i][j][0])
        sns.set_context({'figure.figsize': [50, 50]})
        for i in range(10):
            for j in range(i):
                # if i == j:
                #     continue
                plt.subplot(10, 10, i * 10 + j + 1)
                sns.scatterplot(x=pca_data[i][j][:, 0], y=pca_data[i][j][:, 1], hue=binary_set[i][j][1])
        plt.savefig("binary_ovo_pca.png")
        plt.show()

    # smo
    train_count = jnp.zeros((len(train_data), 10), dtype=jnp.int64)
    train_res = jnp.zeros((10, 10, len(train_data)), dtype=jnp.int64)
    if read_model and os.path.exists(model_path):
        smo_models, train_acc = pickle.load(open(model_path, "rb"))
        for i in trange(10):
            for j in range(i):
                if i == j:
                    continue
                # print(f"Validating on train set {i} vs {j}")
                prediction = smo_models[i][j].predict(train_data)
                train_res[i, j, :] = prediction
                for k in range(len(train_data)):
                    if prediction[k] == 1:
                        train_count[k][i] += 1
                    else:
                        train_count[k][j] += 1
        train_result = jnp.argmax(train_count, axis=1)
        print(f"train accuracy is:{mertics.accuracy(train_label, train_result):.3%}")
    else:
        smo_models = [[[] for _ in range(10)] for __ in range(10)]
        train_acc = jnp.zeros((10, 10))
        # train_count = np.zeros((len(train_data), 10), dtype=np.int64)
        # train_res = np.zeros((10, 10, len(train_data)), dtype=np.int64)
        for i in range(10):
            for j in range(i):
                if i == j:
                    continue
                _model = model.SMO(binary_set[i][j][0],
                                   binary_set[i][j][1],
                                   c, tol, max_passes, rbf_kernel)
                # print(f"Training {i} vs {j}")
                # begin = datetime.datetime.now()
                _model.fit()
                # end = datetime.datetime.now()
                # time_sub = end - begin
                # print("train time", time_sub.total_seconds())
                prediction = _model.predict(binary_set[i][j][0])
                prediction_all = _model.predict(train_data)

                # ovo vote
                train_res[i, j, :] = prediction_all

                for k in range(len(train_data)):
                    train_count[k][i if prediction_all[k] == 1 else j] += 1
                    # if prediction_all[k] == 1:
                    #     train_count[k][i] += 1
                    # else:
                    #     train_count[k][j] += 1

                train_acc[i][j] = mertics.accuracy(prediction, binary_set[i][j][1])
                print(f"Train accuracy {i} vs {j}: {mertics.accuracy(prediction, binary_set[i][j][1]):.3%}")
                smo_models[i][j] = _model
        train_result = jnp.argmax(train_count, axis=1)
        print(f"train accuracy is:{mertics.accuracy(train_label, train_result):.3%}")

    pickle.dump([smo_models, train_acc], open(model_path, "wb"))
    # sns.set_context({'figure.figsize': [10, 10]})
    # sns.heatmap(train_acc, annot=True, fmt=".3%")
    # plt.show()

    # test
    test_count = jnp.zeros((len(test_data), 10), dtype=jnp.int64)
    test_res = jnp.zeros((10, 10, len(test_data)), dtype=jnp.int64)
    for i in trange(10):
        for j in range(i):
            if i == j:
                continue
            # print(f"Validating on test set {i} vs {j}")
            prediction = smo_models[i][j].predict(test_data)
            test_res[i, j, :] = prediction
            for k in range(len(test_data)):
                if prediction[k] == 1:
                    test_count[k][i] += 1
                else:
                    test_count[k][j] += 1
    test_result = jnp.argmax(test_count, axis=1)
    print(f"test accuracy is:{mertics.accuracy(test_label, test_result):.3%}")
    nni.report_final_result({'default': mertics.accuracy(test_label, test_result),
                             'train_acc': mertics.accuracy(train_label, train_result)})
    d = 3


def one_vs_rest():
    # ovr dataset
    binary_set = [[] for _ in range(10)]
    for i in range(10):
        # data = train_data[np.where(train_label == i)]
        # label = train_label[np.where(train_label == i)]
        label = np.where(train_label == i, 1, -1)
        binary_set[i] = [train_data, label]

    # pca
    if pca_inspect:
        pca_model = [PCA(n_components=2) for _ in range(10)]
        pca_data = [[] for _ in range(10)]
        for i in range(10):
            pca_model[i].fit(binary_set[i][0])
            pca_data[i] = pca_model[i].transform(binary_set[i][0])

        sns.set_context({'figure.figsize': [50, 50]})
        for i in range(10):
            plt.subplot(4, 3, i + 1)
            sns.scatterplot(x=pca_data[i][:, 0], y=pca_data[i][:, 1], hue=binary_set[i][1])

        plt.savefig("binary_ovr_pca.png")
        plt.show()

    # smo
    if read_model and os.path.exists(model_path):
        smo_models, train_acc = pickle.load(open(model_path, "rb"))
    else:
        smo_models = [[] for _ in range(10)]
        train_acc = np.zeros((10, 1))
        for i in range(10):
            smo_models[i] = model.SMO(binary_set[i][0],
                                      binary_set[i][1],
                                      c, tol, max_passes, linear_kernel)
            print(f"Training {i} vs rest")
            begin = datetime.datetime.now()
            smo_models[i].fit()
            end = datetime.datetime.now()
            time_sub = end - begin
            print("train time", time_sub.total_seconds())
            train_res = smo_models[i].predict(binary_set[i][0])
            train_acc[i] = mertics.accuracy(train_res, binary_set[i][1])
            print(f"Train accuracy {i} vs rest: {mertics.accuracy(train_res, binary_set[i][1]):.3%}")

    # pickle.dump([smo_models, train_acc], open(model_path, "wb"))
    sns.set_context({'figure.figsize': [10, 10]})
    sns.heatmap(train_acc, annot=True, fmt=".3%")
    plt.show()

    # test
    count = np.zeros((len(test_data), 10), dtype=np.int64)
    for i in range(10):
        print(f"Testing {i} vs rest")
        test_result = smo_models[i].forward(test_data)
        for k in range(len(test_data)):
            count[k][i] += test_result[k]

    # sns.heatmap(count, annot=True, fmt="d")
    # plt.show()
    test_result = np.argmax(count, axis=1)
    print("classfied accuray is:", mertics.accuracy(test_label, test_result))


if __name__ == '__main__':
    # jax.default_device = jax.devices("cpu")[0]
    # c = 1
    tol = 1e-3
    max_passes = 20
    # sigma = 20
    # gamma = 1e-5
    # gamma = 1. / (2 * sigma ** 2)

    # {
    #     "c": 1.9720905169944203,
    #     "gamma": 0.00007371495840890567
    # }

    # params = nni.get_next_parameter()
    params = {'c': 2.88, 'gamma': 6e-5}
    c = params['c']
    gamma = params['gamma']

    # init kernel
    linear_kernel = LinearKernel()
    rbf_kernel = RBFKernel(jnp.sqrt(1 / (2 * gamma)))

    read_model = bool(0)
    pca_inspect = bool(0)
    cls_method = "ovo"
    model_paths = {"ovo": "model/one_vs_one.pkl", "ovr": "model/one_vs_rest.pkl"}
    model_path = model_paths[cls_method]

    train_data, train_label, test_data, test_label = minst(1000, 200)
    train_data, train_label, test_data, test_label = map(jnp.array, [train_data, train_label, test_data, test_label])

    # binary_train_label = [np.where(train_label == i, 1, -1) for i in range(10)]
    # binary_test_label = [np.where(test_label == i, 1, -1) for i in range(10)]

    # i_vs_j()
    if cls_method == "ovo":
        one_vs_one()
    elif cls_method == "ovr":
        one_vs_rest()

    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # axisx = np.linspace(xlim[0], xlim[1], 100)
    # axisy = np.linspace(ylim[0], ylim[1], 100)
    # axisy, axisx = np.meshgrid(axisy, axisx)
    #
    # xy = np.vstack([axisx.ravel(), axisy.ravel()]).T
    # z = smo.forward(xy).reshape(axisx.shape)
    # plt.contour(axisx, axisy, z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # plt.show()
