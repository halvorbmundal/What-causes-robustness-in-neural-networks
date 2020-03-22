import pathlib
import pickle

from skimage import transform
from skimage import exposure
from skimage import io

import random

import os

import numpy as np
import sklearn.model_selection

# Images augmented as done in here:
# https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/

def load_split(csvPath):
    # initialize the list of data and labels
    # load the contents of the CSV file, remove the first line (since
    # it contains the CSV header), and shuffle the rows (otherwise
    # all examples of a particular class will be in sequential order)
    rows = open(csvPath).read().strip().split("\n")[1:]
    random.shuffle(rows)
    # loop over the rows of the CSV file
    return rows

def resize_images(basePath, rows, save_to="test"):
    data = []
    labels = []
    for (i, row) in enumerate(rows):
        # check to see if we should show a status update
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {} total images".format(i))

        # split the row into components and then grab the class ID
        # and image path
        (label, imagePath) = row.strip().split(",")[-2:]

        # derive the full path to the image file and load it
        imagePath = os.path.sep.join([basePath, imagePath])
        image = io.imread(imagePath)
        # resize the image to be 32x32 pixels, ignoring aspect ratio,
        # and then perform Contrast Limited Adaptive Histogram
        # Equalization (CLAHE)
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)

        # update the list of data and labels, respectively
        data.append(image)
        labels.append(int(label))

        # convert the data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)
    if not os.path.exists('sign_data/'+save_to):
        os.makedirs('sign_data/'+save_to)

    with open(f"{save_to}/np_x_data.pkl", 'wb') as f:
        pickle.dump(data, f)

    with open(f"{save_to}/np_y_data.pkl", 'wb') as f:
        pickle.dump(labels, f)

    # return a tuple of the data and labels
    return (data, labels)

class GTSRB:
    def __init__(self):
        print("Loading GTSRB")

        np.random.seed(1215)
        home = str(pathlib.Path.home())
        path = f"{home}/numpy_datasets/GTSRB"
        self.dataset = "GTSRB"

        VAL_FRACTION = 0.1
        TEST_FRACTION = 0.1
        num_classes = 43

        X_train, y_train = load_gt_sign(path)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X_train,
            y_train,
            test_size=TEST_FRACTION,
            random_state=1215,
            stratify=y_train)

        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                  test_size=VAL_FRACTION,
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

        print("Done loading GTSRB")

def load_gt_sign(path, basePath="", is_resize=False):
    if is_resize:
        data = load_split(path)
        X, Y = resize_images(basePath, data, "test/")
    else:
        with open(path + "/np_x_data.pkl", 'rb') as f:
            X = pickle.load(f)
        with open(path + "/np_y_data.pkl", 'rb') as f:
            Y = pickle.load(f)

    X = X.astype("float32") / 255.0

    return X, Y


if __name__ == "__main__":
    a = GTSRB()
