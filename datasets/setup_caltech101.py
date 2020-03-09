import os
import shutil

import tensorflow_datasets as tfds
import numpy as np
import pickle as pkl
import time


def load_images(dataset_name, val_fracton=0.1):
    print(f"Loading {dataset_name}")

    start = time.time()

    path = f"./numpy_datasets/{dataset_name}/"
    start = time.time()

    data_dict = {}
    data_dict["X_train"], data_dict["y_train"], data_dict["X_test"], data_dict["y_test"] = None, None, None, None
    if not os.path.exists(path):
        test_generator, train_generator = download_dataset(dataset_name)

        data_dict["X_train"], data_dict["y_train"] = generator_to_array(train_generator)
        data_dict["X_test"], data_dict["y_test"] = generator_to_array(test_generator)
        os.makedirs(path)

        save_ndarrays(data_dict, path)

        shutil.rmtree("./tensorflow_datasets")
    else:
        load_ndarrays(data_dict, path)
    print(f"Done loading {dataset_name}. It took {time.time() - start} seconds.")

    X_train, y_train, X_test, y_test = data_dict["X_train"], data_dict["y_train"], data_dict["X_test"], data_dict["y_test"]

    num_pts = X_train.shape[0]
    idx = np.random.permutation(num_pts)
    validation_idx = idx[:int(val_fracton * num_pts)]
    train_idx = idx[int(val_fracton * num_pts):]
    X_val = X_train[validation_idx]
    y_val = y_train[validation_idx]

    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

    return X_train,y_train, X_val, y_val, X_test, y_test


def load_ndarrays(data_dict, path):
    for i in data_dict.keys():
        fileName = path + i
        fileObject = open(fileName, 'rb')
        data_dict[i] = pkl.load(fileObject)
        fileObject.close()


def save_ndarrays(data_dict, path):
    for i in data_dict.keys():
        fileName = path + i
        fileObject = open(fileName, 'wb')
        pkl.dump(data_dict[i], fileObject)
        fileObject.close()


def download_dataset(dataset_name):
    datasets = tfds.load(dataset_name, data_dir="./tensorflow_datasets")
    train_dataset, test_dataset = datasets['train'], datasets['test']
    train_generator, test_generator = tfds.as_numpy(train_dataset), tfds.as_numpy(test_dataset)
    return test_generator, train_generator


def generator_to_array(train_generator):
    X, y = [], []
    for x in train_generator:
        X.append(x["image"])
        y.append(x["label"])
    X, y = np.array(X), np.array(y)
    return X, y


class Caltech():
    def __init__(self):
        dataset_name = "caltech101"
        if True:
            raise ValueError('Denne skal ikke kjøres.')

        X_train, y_train, X_val, y_val, X_test, y_test = load_images(dataset_name)

        self.train_data = X_train
        self.train_labels = y_train

        self.validation_data = X_val
        self.validation_labels = y_val

        self.test_data = X_test
        self.test_labels = y_test


