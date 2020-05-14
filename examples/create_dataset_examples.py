import numpy as np
import matplotlib.pyplot as plt

from datasets.setup_GTSRB import GTSRB
from datasets.setup_calTech_101_silhouettes import CaltechSiluettes
from datasets.setup_cifar import CIFAR
from datasets.setup_mnist import MNIST
from datasets.setup_rockpaperscissors import RockPaperScissors
from datasets.setup_sign_language import SignLanguage


fig = plt.figure(figsize=(30, 3))


def plot_images(images):
    images = images - np.min(images)
    images = images / np.max(images)
    for i in range(len(images)):
        ax = fig.add_subplot(1, 10, i+1)
        plt.imshow(images[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def main():
    data = RockPaperScissors()
    images = []
    i = 0
    next_label = 0
    while len(images) < 3:
        image = data.test_data[i]
        label = np.argmax(data.test_labels[i])
        if label == next_label:
            images.append(image)
            next_label += 1
        i += 1

    plot_images(np.reshape(images, [
        -1,
        data.train_data.shape[1],
        data.train_data.shape[2],
        data.train_data.shape[3]
    ]))
    plt.show()


if __name__ == "__main__":
    main()
