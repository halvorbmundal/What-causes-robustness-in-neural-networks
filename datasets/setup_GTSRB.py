import pathlib
import pickle

import numpy as np
import sklearn.model_selection


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

def load_gt_sign(path):
    with open(path + "/np_x_data.pkl", 'rb') as f:
        X = pickle.load(f)
    with open(path + "/np_y_data.pkl", 'rb') as f:
        Y = pickle.load(f)

    X = X.astype("float32") / 255.0

    return X, Y


if __name__ == "__main__":
    a = GTSRB()
