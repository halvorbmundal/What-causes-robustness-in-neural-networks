import multiprocessing

from tensorflow.contrib.keras.api.keras.models import load_model

from Attacks.cw_attack import cw_attack
from Attacks.l0_attack import CarliniL0
from Attacks.l1_attack import EADL1
from Attacks.l2_attack import CarliniL2
from Attacks.li_attack import CarliniLi
from cnn_robustness_tester import gpu_calculations, CnnTestParameters, pool_init
from datasets.setup_GTSRB import GTSRB
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from datasets.setup_mnist import MNIST
from utils import generate_data


def main(model_name, data):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False
    sess = tf.Session(config=config)
    with sess.as_default():
        # cw_attack(model_name, "i", sess, num_image=10, data_set_class=data)

        def loss(correct, predicted):
            return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

        images = []
        true_labels = []
        targets = []
        i = 0
        while len(images) < 5:
            image = data.test_data[i]
            label = np.argmax(data.test_labels[i])
            num_classes = len(data.test_labels[0])
            if label not in true_labels:
                images.append(image)
                true_labels.append(label)
                targets.append(label + 1 % num_classes)
            i += 1

        model = load_model(model_name, custom_objects={'fn': loss, 'tf': tf})

        model.image_size = data.test_data.shape[1]
        model.num_channels = data.test_data.shape[3]
        model.num_labels = data.test_labels.shape[1]
        model.predict = model

        images = np.array(images)
        targets = np.eye(model.num_labels)[targets]

        attack_0 = CarliniL0(sess, model, max_iterations=1000)
        attack_1 = EADL1(sess, model, max_iterations=1000)
        attack_2 = CarliniL2(sess, model, max_iterations=1000)
        attack_inf = CarliniLi(sess, model, max_iterations=1000)

        peturbed_0 = attack_0.attack(images, targets)
        peturbed_1 = attack_1.attack(images, targets)
        peturbed_2 = attack_2.attack(images, targets)
        peturbed_inf = attack_inf.attack(images, targets)

    fig = plt.figure(figsize=(10, 10))

    def plot_images(x_position, images, sets):
        images = images + 0.5
        images = images / np.max(images)
        for i in range(len(images)):
            ax = fig.add_subplot(5, sets, x_position + i * sets)
            plt.imshow(images[i], cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plot_images(1, np.reshape(images, [-1, 28, 28]), 5)
    plot_images(2, np.reshape(peturbed_0, [-1, 28, 28]), 5)
    plot_images(3, np.reshape(peturbed_1, [-1, 28, 28]), 5)
    plot_images(4, np.reshape(peturbed_2, [-1, 28, 28]), 5)
    plot_images(5, np.reshape(peturbed_inf, [-1, 28, 28]), 5)
    plt.show()


if __name__ == "__main__":
    model_name = "test/mnist_type=only_cnn_pool=None_d=5_w=null_f40_k=5_ep=10_ac=ada_strid=1_bias=True_init=glorot__reg=None_bn=True_temp=1_bS=128_es=T_pad=F"
    data = MNIST()
    main(model_name, data)
