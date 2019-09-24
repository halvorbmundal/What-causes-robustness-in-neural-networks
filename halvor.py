import time

from train_cnn import train as train_cnn
from pymain import run_cnn
from setup_mnist import MNIST
import csv
import os.path
import tensorflow
from Attacks.cw_attack import cw_attack
import time


def get_name(dataset="mnist", only_cnn=True, pooling="None", depth=1, width=8, filters=[0], kernels=[0], epochs=10,
             activation_function="relu",
             stride=1, bias=True, initializer="glorot_uniform", regulizer="None", batch_normalization=False,
             temperature=1,
             batch_size=128):
    return "models/dataset:{}_onlyCnn:{}_pooling:{}_detph:{}_width:{}_filter:{}_kernel:{}_epochs:{}_activationFunction:{}_" \
           "stride:{}_bias:{}_initializer:{}_regualizer:{}_batchNormalization:{}_temperature:{}_batchSize:{}" \
        .format(dataset, only_cnn, pooling, depth, width, filters, kernels, epochs, activation_function, stride, bias,
                initializer, regulizer, batch_normalization, temperature, batch_size)


def make_result_file():
    if not os.path.exists('results/results.csv'):
        print("made new resultfile")
        with open('results/results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["dataset", "is_only_cnn", "pooling", "depth", "width", "filter", "kernel", "epochs",
                             "activation_function", "stride", "has_bias",
                             "initializer", "regulizer", "has_batch_normalization", "temperature", "bach_size",
                             "lower_bound", "upper_bound", "l_norm", "time_elapsed", "accuracy"])

make_result_file()
dataset="mnist"
only_cnn=True
pooling="None"
epochs=2
activation_function="ada"
stride=1
bias=True
initializer="glorot_uniform"
regulizer="None"
batch_normalization=False
temperature = 1
bach_size = 128
num_image=1
l_norm="i"
for depth in range(1, 2):
    for width in range(1, 30, 10):
        for kernel_size in range(1, 5):
            for filter_size in range(1, 5):
                filters = [filter_size for i in range(depth)]
                kernels = [kernel_size for i in range(depth)]
                file_name = get_name(
                    dataset=dataset,
                    only_cnn=True,
                    pooling=pooling,
                    depth=depth,
                    width=width,
                    filters=filters,
                    kernels=kernels,
                    epochs=1,
                    activation_function="relu",
                    stride=1,
                    bias=bias,
                    initializer=initializer,
                    regulizer=regulizer,
                    batch_normalization=batch_normalization,
                    temperature=temperature,
                    batch_size=bach_size)
                history = train_cnn(MNIST(), file_name=file_name, filters=filters, kernels=kernels, num_epochs=epochs)
                accuracy = history.history["val_acc"][-1]
                tensorflow.keras.backend.clear_session()

                lower_bound, total_time = run_cnn(file_name, num_image, l_norm, True, activation_function, False)

                upper_bound = "None"
                with tensorflow.keras.backend.get_session() as sess:
                    upper_bound, time = cw_attack(file_name, l_norm, sess, num_image=num_image)

                with open('results/results.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(dataset, only_cnn, pooling, depth, width, filters, kernels, epochs,
                                    activation_function, stride, bias, initializer, regulizer, batch_normalization,
                                    temperature, bach_size, lower_bound, upper_bound, l_norm, time_elapsed, accuracy)

# hvilke paremetere kan jeg tweake på??
# cnn, ikke cnn, blanding
# pooling
# dybde nn
# bredde nn
# activation function
# stride
# filters
# kernels
# epochs
# dilation
# bias
# Initializers
# Regulizers
# Batch_normalization
# Temperature
# Batch size
