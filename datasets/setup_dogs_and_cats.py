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

# Downloaded from https://www.kaggle.com/chetankv/dogs-cats-images/
from datasets.setup_util import show_progress



def resize_images(path, imgs_path):
    data = []
    labels = []
    for data_set in os.listdir(f"{path}/{imgs_path}"):
        if data_set != ".DS_Store":
            label_index = 0
            print(f"\nRezising {data_set} images:")
            for label in os.listdir(f"{path}/{imgs_path}/{data_set}"):
                if label != ".DS_Store":
                    index = 0
                    images = os.listdir(f"{path}/{imgs_path}/{data_set}/{label}")
                    for image_path in images:
                        show_progress(index, 1, len(images))
                        index += 1
                        if image_path != ".DS_Store":
                            image = io.imread(f"{path}/{imgs_path}/{data_set}/{label}/{image_path}")
                            image = transform.resize(image, (64, 64))
                            data.append(image)
                            labels.append(label_index)

                    label_index += 1

    data = np.array(data)
    labels = np.array(labels)

    with open(f"{path}/np_x_data.pkl", 'wb') as f:
        pickle.dump(data, f)

    with open(f"{path}/np_y_data.pkl", 'wb') as f:
        pickle.dump(labels, f)

    # return a tuple of the data and labels
    return data, labels

def load_data(dataset_name):
    home = str(Path.home())
    path = f"{home}/numpy_datasets/{dataset_name}/"
    img_path = "dataset"

    if not os.path.exists(path + "/np_x_data.pkl"):
        print("extracting from images")
        X, Y = resize_images(path, img_path)
    else:
        with open(path + "/np_x_data.pkl", 'rb') as f:
            X = pickle.load(f)
        with open(path + "/np_y_data.pkl", 'rb') as f:
            Y = pickle.load(f)

    X = X.astype("float32") / 255.0

    # one-hot encode the training and testing labels
    #numLabels = len(np.unique(Y))
    #Y = to_categorical(Y, numLabels)
    return X, Y

class DogsAndCats():
    def __init__(self):
        print("Setting up dogs and cats", flush=True)

        self.dataset = 'dogs-and-cats'

        X_train, y_train = load_data(self.dataset)

        TEST_FRACTION = 0.1
        VAL_FRACTION = 0.1
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                    test_size=TEST_FRACTION,
                                                                                    random_state=1215, stratify=y_train)
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=VAL_FRACTION,
                                                                                  random_state=1215, stratify=y_train)

        num_classes = 2
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

if __name__ == "__main__":
    a = DogsAndCats()
    print(a.train_labels.shape)















