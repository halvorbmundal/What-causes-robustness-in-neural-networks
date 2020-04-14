import time

from Attacks.PGD_attack import LinfPGDAttack
from tensorflow.contrib.keras.api.keras.models import load_model
import tensorflow as tf
import numpy as np

from datasets.setup_mnist import MNIST


def loss(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)

def PGD(model, sess, epsilon, num_steps, step_size, data=MNIST()):


    image_size = data.test_data.shape[1]
    num_channels = data.test_data.shape[3]
    num_labels = data.test_labels.shape[1]

    shape = (None, image_size, image_size, num_channels)
    model.x_input = tf.placeholder(tf.float32, shape)
    model.y_input = tf.placeholder(tf.float32, [None, num_labels])

    pre_softmax = model(model.x_input)
    y_loss = tf.nn.softmax_cross_entropy_with_logits(labels=model.y_input, logits=pre_softmax)
    model.xent = tf.reduce_sum(y_loss)

    attack = LinfPGDAttack(model, epsilon, num_steps, step_size, random_start=True)
    return attack.perturb(np.array([data.test_data]), np.array([data.test_labels]), sess)

def get_accuracy(file_name, sess, epsilon, num_steps, step_size, data=MNIST()):
    model = load_model(file_name, custom_objects={'fn': loss, 'tf': tf, 'atan': tf.math.atan})
    start_time = time.time()
    adversaries = PGD(model, sess, epsilon, num_steps, step_size, data)
    predictions = model.predict(adversaries)
    accuracy = np.mean(np.equal(np.argmax(predictions, 1), np.argmax(data.test_labels, 1)))
    time_used = time.time() - start_time

    return time_used, accuracy
