# Interface to run CW/EAD attacks

import numpy as np
from Attacks.HSJA import hsja
from datasets.setup_cifar import CIFAR
from datasets.setup_tinyimagenet import TinyImagenet
from tensorflow.contrib.keras.api.keras.models import load_model
from CNN_Cert.utils import generate_data
import time as timer
import random
from CNN_Cert.train_resnet import *


def loss(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)


# Runs CW/EAD attack with specified norm
def hsja_attack(file_name, norm, sess, num_image=10, data_set_class=MNIST(), targeted=False):
    print("hsja attack", flush=True)
    np.random.seed(1215)
    tf.set_random_seed(1215)
    random.seed(1215)

    data = data_set_class

    model = load_model(file_name,
                       custom_objects={'fn': loss, 'tf': tf, 'ResidualStart': ResidualStart, 'ResidualStart2': ResidualStart2, 'tf': tf, 'atan': tf.math.atan})
    inputs, targets, true_labels, true_ids, img_info = generate_data(data, samples=num_image, targeted=True, random_and_least_likely=True, target_type=0b0001,
                                                                     predictor=model.predict, start=0)
    if len(inputs) == 0:
        return 0, 0

    if targeted:
        target_examples = get_target_examples(data, model, targets)
        targets = np.argmax(targets, axis=1)
    else:
        target_examples = [None for i in range(len(targets))]
        targets = [None for i in range(len(targets))]

    if norm == "2":
        constraint = "l2"
        norm_fn = lambda x: np.sum(x ** 2, axis=(1, 2, 3))
    elif norm == "i":
        constraint = "linf"
        norm_fn = lambda x: np.max(np.abs(x), axis=(1, 2, 3))
    else:
        raise ValueError("norm must be 2 or inf")

    start_time = timer.time()
    perturbed_input = attack_multiple(inputs, targets, target_examples, constraint, model)
    UB = np.average(norm_fn(perturbed_input - inputs))
    time_spent = (timer.time() - start_time) / len(inputs)
    return UB, time_spent


def get_target_examples(data, model, targets):
    target_examples = []
    for target in targets:
        target_example_found = False
        target_example_index = 0
        while not target_example_found:
            target_example_index = find_first(data.train_labels[target_example_index:], np.argmax(target))
            target_example = data.train_data[target_example_index]
            prediction = np.argmax(model.predict(np.array([target_example, ])))
            if prediction == np.argmax(target):
                target_example_found = True
                target_examples.append(target_example)
            target_example_index += 1
    return target_examples


def attack_multiple(imgs, targets, target_examples, constraint, model):
    """
    Perform the L_0 attack on the given images for the given targets.

    If self.targeted is true, then the targets represents the target labels.
    If self.targeted is false, then targets are the original class labels.
    """

    r = []
    print(f"{len(imgs)} images and {len(targets)} targets")
    for img, target, target_example in zip(imgs, targets, target_examples):
        print(f"calclulating robustness for image {len(r) + 1}", flush=True)
        r.append(hsja(model,
                      img,
                      clip_max=0.5,
                      clip_min=-0.5,
                      constraint=constraint,
                      num_iterations=150,
                      gamma=1.0,
                      target_label=target,
                      target_image=target_example,
                      stepsize_search='geometric_progression',
                      max_num_evals=1e4,
                      init_num_evals=100,
                      verbose=False))
    return np.array(r)


def find_first(arr, label):
    for i in range(len(arr)):
        if np.argmax(arr[i]) == label:
            return i
    return -1
