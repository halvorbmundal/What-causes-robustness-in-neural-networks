import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from datasets.setup_util import download_dataset


def load_images(dataset_name, download_url, file_name):
    home = str(Path.home())
    path = f"{home}/numpy_datasets/{dataset_name}/"

    if not os.path.exists(path):
        download_dataset(path, file_name, download_url)

    X, Y = extract_dataset(from_path=path, file_name=file_name)
    X = X.reshape((-1, 28, 28, 1))
    Y = Y.reshape((-1))
    return preprocess_to_ndarray(X, Y)


def extract_dataset(from_path, file_name):
    data = loadmat(from_path + file_name)
    return data["X"], data["Y"]


def preprocess_to_ndarray(X, Y):
    np.random.seed(1215)

    VAL_FRACTION = 0.1
    TEST_FRACTION = 0.1

    X_train = X
    y_train = Y
    print(y_train)
    num_pts = X_train.shape[0]

    idx = np.random.permutation(num_pts)

    test_idx = idx[:int(TEST_FRACTION * num_pts)]
    train_idxs = idx[int(TEST_FRACTION * num_pts):]
    val_idx = train_idxs[:int(VAL_FRACTION * num_pts)]
    train_idx = train_idxs[int(VAL_FRACTION * num_pts):]

    X_test = X_train[test_idx]
    y_test = y_train[test_idx]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


class Caltech_siluettes():
    def __init__(self):
        dataset = 'caltech_siluettes'
        download_location = "https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28.mat"
        download_name = "caltech101_silhouettes_28.mat"

        # X_train shape: num_train*3*64*64
        X_train, y_train, X_val, y_val, X_test, y_test = load_images(dataset, download_location, download_name)

        # convetion is num_train*size*size*channel, e.g. MNIST: num*28*28*1
        self.train_data = X_train
        self.train_labels = y_train

        self.validation_data = X_val
        self.validation_labels = y_val

        self.test_data = X_test
        self.test_labels = y_test

        self.inp_shape = self.train_data.shape[1:]


if __name__ == "__main__":
    a = Caltech_siluettes()
    print(a.train_labels.shape)