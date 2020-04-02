# set the matplotlib backend so figures can be saved in the background
import pickle

# import the necessary packages
import sklearn.model_selection
from skimage import transform
from skimage import exposure
from skimage import io
import numpy as np
import random
import os
from pathlib import Path
import pandas as pd

# Downloaded from https://www.kaggle.com/datamunge/sign-language-mnist/
from datasets.setup_util import show_progress

def convert_from_csv(path):
    data = []
    labels = []
    for data_set in os.listdir(f"{path}"):
        if os.path.isdir(f"{path}/{data_set}"):
            for csv in os.listdir(f"{path}/{data_set}"):
                if csv != ".DS_Store":
                    df = pd.read_csv(f"{path}/{data_set}/{csv}")
                    y = df["label"].values
                    df.drop('label', axis=1, inplace=True)
                    images = df.values
                    images = images.reshape(-1, 28, 28, 1)
                    data.extend(images)
                    labels.extend(y)
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

def load_data(dataset_name):
    home = str(Path.home())
    path = f"{home}/numpy_datasets/{dataset_name}/"

    X, Y = convert_from_csv(path)

    X = X.astype("float32") / 255.0

    #shift around 0
    X = X - 0.5

    return X, Y

class SignLanguage():
    def __init__(self):
        print("Setting up sign language", flush=True)

        self.dataset = 'sign-language'

        X_train, y_train = load_data(self.dataset)

        TEST_FRACTION = 0.1
        VAL_FRACTION = 0.2
        num_classes = 25
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

        print("Done setting up sign language")

if __name__ == "__main__":
    a = SignLanguage()
    print(a.train_labels.shape)















