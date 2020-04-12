from datasets.setup_mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

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
    data = MNIST()
    images = []
    i = 0
    while len(images) < 10:
        image = data.test_data[i]
        label = np.argmax(data.test_labels[i])
        if label == len(images):
            images.append(image)
        i += 1

    plot_images(np.reshape(images, [-1, 28, 28]))
    plt.show()


if __name__ == "__main__":
    main()
