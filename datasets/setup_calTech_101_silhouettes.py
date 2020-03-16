import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import sklearn.model_selection

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

    VAL_FRACTION = 0.2
    TEST_FRACTION = 0.5
    num_classes = 101

    X_train = X
    y_train = Y

    # stratifies the splits
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                test_size=TEST_FRACTION,
                                                                                random_state=1215, stratify=y_train)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=VAL_FRACTION,
                                                                              random_state=1215, stratify=y_train)

    y_test = np.eye(num_classes)[y_test - 1]
    y_val = np.eye(num_classes)[y_val - 1]
    y_train = np.eye(num_classes)[y_train - 1]

    return X_train, y_train, X_val, y_val, X_test, y_test


class CaltechSiluettes():
    def __init__(self):
        self.dataset = 'caltech_siluettes'
        download_location = "https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28.mat"
        download_name = "caltech101_silhouettes_28.mat"

        # X_train shape: num_train*3*64*64
        X_train, y_train, X_val, y_val, X_test, y_test = load_images(self.dataset, download_location, download_name)

        # convetion is num_train*size*size*channel, e.g. MNIST: num*28*28*1
        self.train_data = X_train
        self.train_labels = y_train

        self.validation_data = X_val
        self.validation_labels = y_val

        self.test_data = X_test
        self.test_labels = y_test

        self.inp_shape = self.train_data.shape[1:]


if __name__ == "__main__":
    a = CaltechSiluettes()
    print(a.train_labels.shape)
