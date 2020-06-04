import multiprocessing

from tensorflow.contrib.keras.api.keras.models import load_model

from Attacks.PGD_attack import LinfPGDAttack
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from datasets.setup_GTSRB import GTSRB
from datasets.setup_mnist import MNIST


def main(model_name, data, epsilon=0.111):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False
    sess = tf.Session(config=config)
    with sess.as_default():

        def loss(correct, predicted):
            return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

        images = []
        true_labels = []
        true_labels_one_hot = []
        targets = []
        i = 0
        while len(images) < 5:
            image = data.test_data[i]
            label = np.argmax(data.test_labels[i])
            num_classes = len(data.test_labels[0])
            if label not in true_labels:
                images.append(image)
                true_labels.append(label)
                true_labels_one_hot.append(data.test_labels[i])
                targets.append(label + 1 % num_classes)
            i += 1

        model = load_model(model_name, custom_objects={'fn': loss, 'tf': tf})

        image_size = data.test_data.shape[1]
        num_channels = data.test_data.shape[3]
        num_labels = data.test_labels.shape[1]

        shape = (None, image_size, image_size, num_channels)
        model.x_input = tf.placeholder(tf.float32, shape)
        model.y_input = tf.placeholder(tf.float32, [None, num_labels])

        pre_softmax = model(model.x_input)
        y_loss = tf.nn.softmax_cross_entropy_with_logits(labels=model.y_input, logits=pre_softmax)
        model.xent = tf.reduce_sum(y_loss)

        adv_steps = 40
        attack = LinfPGDAttack(model, epsilon, adv_steps, epsilon * 1.33 / adv_steps, random_start=True)
        advs = attack.perturb(np.array(images), true_labels_one_hot, sess)
        advs2 = LinfPGDAttack(model, 0.01, adv_steps, 0.01 * 1.33 / adv_steps, random_start=True).perturb(np.array(images), true_labels_one_hot, sess)

    fig = plt.figure(figsize=(10, 4))

    def plot_images(x_position, images, sets, gray_scale=False):
        images = images + 0.5
        images = images / np.max(images)
        for i in range(len(images)):
            ax = fig.add_subplot(sets, 5, i + 5*x_position+1)
            if gray_scale:
                plt.imshow(images[i], cmap='gray')
            else:
                plt.imshow(images[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    if num_channels == 1:
        plot_images(0, np.reshape(images, [-1, 28, 28]), 3, gray_scale=True)
        plot_images(1, np.reshape(advs, [-1, 28, 28]), 3, gray_scale=True)
    else:
        plot_images(0, np.reshape(images, [-1, 32, 32, 3]), 3)
        plot_images(1, np.reshape(advs, [-1, 32, 32, 3]), 3)
    plt.show()

GTRSB_model = "../remote/v10/GTSRB_type=only_cnn_pool=None_d=4_w=null_f32_k=3_ep=10_ac=tanh_strid=1_bias=True_init=glorot__reg=None_bn=False_temp=1_bS=128_es=T_pad=F"
mnist_model = "../v10/mnist/output/models/mnist_type=only_cnn_pool=None_d=1_w=null_f8_k=3_ep=10_ac=sigmoid_strid=1_bias=True_init=glorot__reg=None_bn=False_temp=1_bS=128_es=T_pad=F"

if __name__ == "__main__":
    model_name = GTRSB_model
    data = GTSRB()
    main(model_name, data, 0.1)
