import gc
import os
import sys
from tensorflow.contrib.keras.api.keras.models import load_model
import tensorflow as tf
import pandas as pd

from datasets.setup_GTSRB import GTSRB
from datasets.setup_calTech_101_silhouettes import CaltechSiluettes
from datasets.setup_cifar import CIFAR
from datasets.setup_mnist import MNIST
from datasets.setup_rockpaperscissors import RockPaperScissors
from datasets.setup_sign_language import SignLanguage
from datasets.setup_tinyimagenet import TinyImagenet


def loss(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)


def get_data(dataset):
    if dataset == "mnist":
        data = MNIST()
    elif dataset == "cifar":
        data = CIFAR()
    elif dataset == "tinyImagenet":
        data = TinyImagenet()
    elif dataset == "caltechSilhouettes":
        data = CaltechSiluettes()
    elif dataset == "GTSRB":
        data = GTSRB()
    elif dataset == "sign-language":
        data = SignLanguage()
    elif dataset == "rockpaperscissors":
        data = RockPaperScissors()
    else:
        raise NameError(f"{dataset} is not a valid dataset")
    return data


def main():
    _, path = sys.argv
    datasets = ["cifar", "caltechSilhouettes", "GTSRB", "mnist", "rockpaperscissors", "sign-language"]
    for dataset in datasets:
        data_path = f"{path}/{dataset}"
        data = get_data(dataset)
        gc.collect()
        df_data = {"accuracy": [], "file_name": []}
        for file_name in os.listdir(f"{data_path}/adv_models"):
            model_path = f"{data_path}/adv_models/{file_name}"
            model = load_model(model_path, custom_objects={'fn': loss, 'tf': tf, 'atan': tf.math.atan})
            _, accuracy = model.evaluate(data.test_data, data.test_labels, verbose=0)
            df_data["accuracy"].append(accuracy)
            df_data["file_name"].append(file_name)
        df = pd.DataFrame(df_data)
        df.to_csv(f"{data_path}/adv_model_natural_accuracy.csv", index=False)

main()
