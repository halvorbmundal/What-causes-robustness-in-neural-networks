"""
setup_tinyimagenet.py

Tinyimagenet data and model loading code

Copyright (C) 2018, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Luca Daniel <dluca@mit.edu>
"""

import os
import shutil
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import sklearn.model_selection
from PIL import Image

from datasets.setup_util import save_ndarrays, download_dataset


def load_images(dataset_name, download_url, file_name):
    home = str(Path.home())
    path = f"{home}/numpy_datasets/{dataset_name}/"

    start = time.time()
    data_dict = {}
    """
    This takes 10 GB
        
    if not os.path.exists(path):
        download_and_save_to_ndarrays(data_dict, download_url, file_name, path, dataset_name)

    data_dict["X_train"], data_dict["y_train"], data_dict["X_val"], data_dict["y_val"], data_dict["X_test"], \
    data_dict["y_test"] = None, None, None, None, None, None
    
    """

    file_name_zip = file_name + ".zip"
    temp_path = f"{home}/numpy_datasets/temp_data/{dataset_name}/"
    unzipped_path = f"{path}{file_name}/"
    if not os.path.exists(path):
        print("Did not find dataset.", flush=True)
        download_dataset(temp_path, file_name_zip, download_url)
        extract_datazip(to_path=path, from_path=temp_path, file_name=file_name, file_name_zip=file_name_zip)
    if os.path.exists(temp_path):
        print("Deleting temp data", flush=True)
        shutil.rmtree(temp_path)
    data_dict["X_train"], data_dict["y_train"], data_dict["X_val"], data_dict["y_val"], data_dict["X_test"], \
        data_dict["y_test"] = preprocess_to_ndarray(unzipped_path)

    print(f"Done loading {dataset_name}. It took {time.time() - start} seconds.", flush=True)

    return data_dict["X_train"], data_dict["y_train"], data_dict["X_val"], data_dict["y_val"], data_dict["X_test"], \
           data_dict["y_test"]


def download_and_save_to_ndarrays(data_dict, download_url, file_name, path, dataset_name):
    home = str(Path.home())
    print("Home dir:", home)
    file_name_zip = file_name + ".zip"
    temp_path = f"{home}/numpy_datasets/temp_data/{dataset_name}/"
    unzipped_path = f"{temp_path}{file_name}/"

    download_dataset(temp_path, file_name_zip, download_url)
    extract_datazip(temp_path, file_name, file_name_zip)
    data_dict["X_train"], data_dict["y_train"], data_dict["X_val"], data_dict["y_val"], data_dict["X_test"], \
    data_dict["y_test"] = preprocess_to_ndarray(unzipped_path)

    save_ndarrays(data_dict, path)

    os.makedirs(path)
    shutil.rmtree(temp_path)


def extract_datazip(to_path, from_path, file_name, file_name_zip):
    if not os.path.exists(to_path + file_name + "/"):
        print("Unzipping", flush=True)
        with ZipFile(from_path + file_name_zip, 'r') as zipObj:
            zipObj.extractall(path=to_path)
        print("Unzipped", flush=True)


def preprocess_to_ndarray(path):
    print("Converting data to ndarray", flush=True)
    train_path = path + "train"
    np.random.seed(1215)
    num_classes = 200

    X_train, y_train = get_train_data(num_classes, train_path)
    print("Done converting", flush=True)

    VAL_FRACTION = 0.1
    TEST_FRACTION = 0.1

    # stratifies the splits
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                test_size=TEST_FRACTION,
                                                                                random_state=1215, stratify=y_train)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=VAL_FRACTION,
                                                                              random_state=1215, stratify=y_train)

    y_test = np.eye(num_classes)[y_test]
    y_val = np.eye(num_classes)[y_val]
    y_train = np.eye(num_classes)[y_train]

    X_train = np.float32(X_train)
    X_val = np.float32(X_val)
    X_test = np.float32(X_test)

    # X_train shape: num_train*3*64*64
    # convetion is num_train*size*size*channel, e.g. MNIST: num*28*28*1
    return np.swapaxes(X_train, 1, 3), y_train, np.swapaxes(X_val, 1, 3), y_val, np.swapaxes(X_test, 1, 3), y_test


def get_train_data(num_classes, train_path):
    X_train = np.zeros([num_classes * 500, 3, 64, 64], dtype='uint8')
    y_train = np.zeros([num_classes * 500], dtype='uint8')
    i = 0
    j = 0
    annotations = {}
    for sChild in os.listdir(train_path):
        sChildPath = os.path.join(os.path.join(train_path, sChild), 'images')
        annotations[sChild] = j
        for c in os.listdir(sChildPath):
            X = np.array(Image.open(os.path.join(sChildPath, c)))
            if len(np.shape(X)) == 2:
                X_train[i] = np.array([X, X, X])
            else:
                X_train[i] = np.transpose(X, (2, 0, 1))
            y_train[i] = j
            i += 1
        j += 1
        if j >= num_classes:
            break
    return X_train, y_train


class TinyImagenet():
    def __init__(self):
        print("Setting up tinyImagenet", flush=True)

        dataset = 'tiny-imagenet-200'
        download_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        file_name = "tiny-imagenet-200"

        X_train, y_train, X_val, y_val, X_test, y_test = load_images(dataset, download_url, file_name)

        self.train_data = X_train
        self.train_labels = y_train

        self.validation_data = X_val
        self.validation_labels = y_val

        self.test_data = X_test
        self.test_labels = y_test

        self.inp_shape = self.train_data.shape[1:]


if __name__ == "__main__":
    a = TinyImagenet()
    print(a.train_labels.shape)
