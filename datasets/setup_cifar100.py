## setup_cifar.py -- cifar data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import os
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import sklearn.model_selection

from datasets.setup_util import show_progress


def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
    final[:,:,:,0] = data[:,0,:,:]
    final[:,:,:,1] = data[:,1,:,:]
    final[:,:,:,2] = data[:,2,:,:]

    final /= 255
    final -= .5
    labels2 = np.zeros((len(labels), 10))
    labels2[np.arange(len(labels2)), labels] = 1

    return final, labels

def load_batch(fpath):
    f = open(fpath, "rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255)-.5)
    return np.array(images),np.array(labels)
    
def unpickle(file):
    with open(file, 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'latin1'
        p: dict = u.load()
        X = p['data'].reshape(-1,32,32,3)
        y = np.array(p['fine_labels'])
    return X, y

class CIFAR100:
    def __init__(self):
        self.dataset="cifar100"
        home = str(Path.home())
        path = f"{home}/numpy_datasets/cifar-100-python"
        temp_path = f"{home}/numpy_datasets/temp_data/cifar100"
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        print("Setting up cifar100")

        num_classes = 100
        VAL_FRACTION = 0.1
        TEST_FRACTION = 0.1

        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.exists(f"{temp_path}/cifar-100-python.tar.gz") and not os.path.exists(f"{path}/cifar-100-batches-bin"):
            os.mkdir(temp_path)
            urllib.request.urlretrieve(url, f"{temp_path}/cifar-100-python.tar.gz", show_progress)

        if not os.path.exists(f"{path}/cifar-100-python"):
            os.popen(f"tar -xzf {temp_path}/cifar-100-python.tar.gz -C {home}/numpy_datasets").read()

        X_train, y_train = unpickle(f"{path}/train")
        X_test, y_test = unpickle(f"{path}/test")

        # To choose our own train-test split
        X_train = np.concatenate((X_train, X_test), axis=0)
        y_train = np.concatenate((y_train, y_test), axis=0)

        # stratifies the splits
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                    test_size=TEST_FRACTION,
                                                                                    random_state=1215, stratify=y_train)
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=VAL_FRACTION,
                                                                                  random_state=1215, stratify=y_train)

        y_test = np.eye(num_classes)[y_test]
        y_val = np.eye(num_classes)[y_val]
        y_train = np.eye(num_classes)[y_train]

        self.train_data = X_train
        self.train_labels = y_train

        self.validation_data = X_val
        self.validation_labels = y_val

        self.test_data = X_test
        self.test_labels = y_test

        self.inp_shape = self.train_data.shape[1:]

        print("Done setting up cifar100")

if __name__ == "__main__":
    a = CIFAR100()
    print(a.train_labels.shape)