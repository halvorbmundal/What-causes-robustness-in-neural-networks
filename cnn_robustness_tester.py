import sys
import time

from train_cnn import train as train_cnn
from pymain import run_cnn
from setup_mnist import MNIST
import csv
import os.path
from Attacks.cw_attack import cw_attack
import time as timer
import tensorflow as tf
from setup_mnist import MNIST
from enum import Enum
import logging
import multiprocessing
import gc
import time
import os

tf.get_logger().setLevel('WARNING')


class NnArchitecture(Enum):
    ONLY_CNN = "only_cnn"


def get_name(parameter_class):
    directory = "output/models"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory + "/dataset={}_nn_architecture={}_pooling={}_detph={}_width={}_filter={}_kernel={}_epochs={}_activationFunction={}_" \
                       "stride={}_bias={}_initializer={}_regualizer={}_batchNormalization={}_temperature={}_batchSize={}" \
        .format(parameter_class.dataset, parameter_class.nn_architecture, parameter_class.pooling,
                parameter_class.depth,
                parameter_class.width, parameter_class.filter_size, parameter_class.kernel_size, parameter_class.epochs,
                parameter_class.activation_function_string, parameter_class.stride, parameter_class.bias,
                parameter_class.initializer, parameter_class.regulizer, parameter_class.has_batch_normalization,
                parameter_class.temperature, parameter_class.batch_size)


def make_result_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory + file):
        print("made new resultfile")
        with open(directory + file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["dataset", "nn_architecture", "pooling", "depth", "width", "filter", "kernel", "epochs",
                             "activation_function", "stride", "has_bias",
                             "initializer", "regulizer", "has_batch_normalization", "temperature", "bach_size",
                             "lower_bound", "upper_bound", "l_norm", "time_elapsed", "accuracy", "file_name"])


def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)


def setDynamicGPUAllocation():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)

    sess = tf.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras


def train_and_save_network(file_name, filters, kernels, epochs, tf_activation, batch_normalization, use_old_network):
    sess = tf.keras.backend.get_session()
    tf.keras.backend.set_session(sess)
    train_cnn(MNIST(),
              file_name=file_name,
              filters=filters,
              kernels=kernels,
              num_epochs=epochs,
              activation=tf_activation,
              bn=batch_normalization,
              use_old_network=use_old_network)
    tf.keras.backend.clear_session()


def get_lower_bound(file_name, num_image, l_norm, only_cnn, activation_function):
    sess = tf.keras.backend.get_session()
    tf.keras.backend.set_session(sess)
    avg_lower_bound, total_time = run_cnn(file_name, num_image, l_norm, core=only_cnn, activation=activation_function)
    tf.keras.backend.clear_session()
    return avg_lower_bound


def get_upper_bound(file_name, l_norm, num_image):
    sess = tf.keras.backend.get_session()
    tf.keras.backend.set_session(sess)
    upper_bound, time_taken = cw_attack(file_name, l_norm, sess, num_image=num_image)
    tf.keras.backend.clear_session()
    return upper_bound


def csv_contains_file(csv_file, file_name):
    with open(csv_file, "rt") as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')

        first_row = next(csvreader)
        column_number = 0
        for i in range(len(first_row)):
            if first_row[i] == "file_name":
                column_number = i
                break

        for row in csvreader:
            if row[column_number] == file_name:
                return True
    return False


def file_exists(file):
    return os.path.exists(file)


def get_tf_activation_function_from_string(activation_function_string):
    if activation_function_string == "sigmoid":
        return tf.math.sigmoid
    if activation_function_string == "ada":
        return tf.nn.relu
    if activation_function_string == "tanh":
        return tf.math.tanh
    if activation_function_string == "arctan":
        return tf.math.atan


def train_nn(file_name, filters, kernels, tf_activation, has_batch_normalization, use_old_network):
    try:
        train_and_save_network(file_name,
                               filters,
                               kernels,
                               CnnTestParameters.epochs,
                               tf_activation,
                               has_batch_normalization,
                               use_old_network)
    except Exception as e:
        print("An exeption occured")
        logging.exception("This file had an error: \n" + file_name + "\n" + str(e) + "\n\n")


def calculate_lower_bound(file_name, num_image, l_norm, nn_architecture, activation_function_string):
    return get_lower_bound(file_name,
                           num_image,
                           l_norm,
                           nn_architecture == NnArchitecture.ONLY_CNN.value,
                           activation_function_string)


def write_to_file(parameters, lower_bound, accuracy, time_elapsed):
    write_lock.acquire()  # from global variable
    try:
        with open(parameters.result_folder + parameters.result_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                [parameters.dataset, parameters.nn_architecture, parameters.pooling, parameters.depth, parameters.width,
                 parameters.filter_size, parameters.kernel_size, parameters.epochs,
                 parameters.activation_function_string, parameters.stride, parameters.bias, parameters.initializer,
                 parameters.regulizer, parameters.has_batch_normalization,
                 parameters.temperature, parameters.batch_size, lower_bound, parameters.upper_bound,
                 parameters.l_norm, time_elapsed, accuracy
                    , parameters.file_name])
    except Exception as e:
        print("An exeption occured while writing to file")
        logging.exception(str(e) + "\n\n")
    finally:
        write_lock.release()


def reset_keras():
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()

    """
    try:
        del classifier # this is from global space - change this as you need
    except:
        pass"""

    print("clear gc", gc.collect())  # if it's done something you should see a number being outputted


def gpu_calculations(parameters):
    keras_lock.acquire()
    try:
        if not file_exists(parameters.file_name):
            setDynamicGPUAllocation()
            print(f"\ntraining with {parameter_string(parameters)}\n", flush=True)
            train_nn(parameters.file_name,
                     parameters.filters,
                     parameters.kernels,
                     parameters.tf_activation,
                     parameters.has_batch_normalization,
                     parameters.use_old_network)
        else:
            print("Neural network already created - {}".format(parameters.result_file), flush=True)
    finally:
        keras_lock.release()


def get_accuracy_of_nn_from_csv(csv_file, file_name):
    with open(csv_file, "rt") as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')

        first_row = next(csvreader)
        for i in range(len(first_row)):
            if first_row[i] == "accuracy":
                accuracy_column = i

            elif first_row[i] == "file_name":
                name_column = i

        for row in csvreader:
            if row[name_column] == file_name:
                return row[accuracy_column]
    return None


def multithreadded_calculations(parameters):
    if parameters.use_gpu:
        gpu_calculations(parameters)

    if parameters.use_cpu:
        semaphore.acquire()
        try:
            print(f"\nCalculating robustness of {parameter_string(parameters)}\n", flush=True)
            setDynamicGPUAllocation()

            if not file_exists(parameters.file_name):
                print_parameters(parameters)
                print("File does not exist {}".format(parameters.file_name), flush=True)
                return

            start_time = timer.time()

            debugprint(parameters.isDebugging, "reading results csv")
            if csv_contains_file(CnnTestParameters.result_folder + CnnTestParameters.result_file, parameters.file_name):
                print_parameters(parameters)
                print("Bounds already calculated for {}".format(parameters.file_name), flush=True)
                return

            debugprint(parameters.isDebugging, "reading models_meta.csv")
            accuracy = get_accuracy_of_nn_from_csv("output/models_meta.csv", parameters.file_name)

            if float(accuracy) < 0.95:
                print_parameters(parameters)
                print("skiped", parameters.file_name, "as the accuracy was too low", flush=True)
                return

            debugprint(parameters.isDebugging, "calculating lower bound")
            lower_bound = calculate_lower_bound(parameters.file_name,
                                                parameters.num_image,
                                                parameters.l_norm,
                                                parameters.nn_architecture,
                                                parameters.activation_function_string)

            time_elapsed = timer.time() - start_time

            debugprint(parameters.isDebugging, "writing to file")
            write_to_file(parameters, lower_bound, accuracy, time_elapsed)

            print_parameters(parameters)
            print("wrote to file", flush=True)

            reset_keras()
            return

        finally:
            semaphore.release()
            gc.collect()


def pool_init(l1, l2, sema):
    global write_lock
    global keras_lock
    global semaphore
    semaphore = sema
    write_lock = l1
    keras_lock = l2


def parameter_string(parameters):
    return "filter_size={} depth={} kernel_size={} ac={}" \
        .format(parameters.filter_size,
                parameters.depth,
                parameters.kernel_size,
                parameters.activation_function_string)


def print_parameters(parameters):
    print()
    print(
        "========================================================================================")
    print(parameter_string(parameters))
    print("", flush=True)

def debugprint(isDebugging, text):
    if isDebugging:
        print(text)

def main():
    _, arg1, arg2, arg3, arg4 = sys.argv
    cpu = arg1 == "cpu" or arg2 == "cpu"
    gpu = arg1 == "gpu" or arg2 == "gpu"
    debugging = arg3 == "debugging"
    use_old_network = arg4 == "old"

    if use_old_network:
        old_path = "old_network"
        if not os.path.exists(old_path):
            os.makedirs(old_path)
        os.chdir(old_path)


    if not debugging:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')
        tf.logging.set_verbosity(tf.logging.ERROR)

    print("You have {} cores at your disposal.".format(multiprocessing.cpu_count()))
    if debugging:
        print(f"cpu: {cpu}")
        print(f"gpu: {gpu}")
        print(f"old: {use_old_network}")

    if cpu:
        processes = 36

        l1 = multiprocessing.Lock()
        l2 = multiprocessing.Lock()
        sema = multiprocessing.Semaphore(processes)
        pool = multiprocessing.Pool(processes, initializer=pool_init, initargs=(l1, l2, sema), maxtasksperchild=1)

    make_result_file(CnnTestParameters.result_folder, CnnTestParameters.result_file)
    logging.basicConfig(filename='log.log', level="ERROR")
    for kernel_size in range(3, 8, 1):
        for filter_size in range(2, 64, 4):
            for has_batch_normalization in [False]:
                for depth in range(1, 5, 1):
                    for activation_function_string in ["ada", "sigmoid", "arctan", "tanh"]:

                        parameters = CnnTestParameters()
                        parameters.tf_activation = get_tf_activation_function_from_string(activation_function_string)
                        parameters.activation_function_string = activation_function_string
                        parameters.depth = depth
                        parameters.kernel_size = kernel_size
                        parameters.filter_size = filter_size
                        parameters.filters = [filter_size for i in range(depth)]
                        parameters.kernels = [kernel_size for i in range(depth)]
                        parameters.has_batch_normalization = has_batch_normalization
                        parameters.use_old_network = use_old_network
                        parameters.isDebugging = debugging
                        parameters.use_gpu = gpu
                        parameters.use_cpu = cpu

                        parameters.file_name = get_name(parameters)

                        if debugging:
                            pool_init(l1, l2, sema)
                            multithreadded_calculations(parameters)
                        else:
                            pool.apply_async(multithreadded_calculations, (parameters,))

    pool.close()
    pool.join()


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


class CnnTestParameters:
    dataset = "mnist"
    nn_architecture = NnArchitecture.ONLY_CNN.value
    pooling = "None"
    epochs = 100
    stride = 1
    bias = True
    initializer = "glorot_uniform"
    regulizer = "None"
    temperature = 1
    batch_size = 128
    num_image = 10
    l_norm = "i"
    width = "null"
    upper_bound = None
    result_folder = 'output/results/'
    result_file = 'results.csv'


if __name__ == "__main__":
    main()
