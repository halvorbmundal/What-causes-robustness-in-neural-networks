import sys
import traceback

_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
import time

from train_cnn import train as train_cnn
from pymain import run_cnn
from setup_mnist import MNIST
from setup_cifar import CIFAR
import csv
import os.path
from Attacks.cw_attack import cw_attack
import time as timer
import tensorflow as tf
from setup_mnist import MNIST
from enum import Enum
import logging
from datetime import datetime
import multiprocessing
import gc
import time
import os
from numba import cuda

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


def boolToString(b):
    if b:
        return "T"
    else:
        return "F"


def get_name_new_convention(parameter_class):
    directory = "output/models"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory + "/{}_type={}_pool={}_d={}_w={}_f{}_k={}_ep={}_ac={}_" \
                       "strid={}_bias={}_init={}_reg={}_bn={}_temp={}_bS={}_es={}_pad={}" \
        .format(parameter_class.dataset, parameter_class.nn_architecture, parameter_class.pooling,
                parameter_class.depth,
                parameter_class.width, parameter_class.filter_size, parameter_class.kernel_size, parameter_class.epochs,
                parameter_class.activation_function_string, parameter_class.stride, parameter_class.bias,
                parameter_class.initializer[:7], parameter_class.regulizer, parameter_class.has_batch_normalization,
                parameter_class.temperature, parameter_class.batch_size,
                boolToString(parameter_class.use_early_stopping),
                boolToString(parameter_class.use_padding_same))


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
                             "lower_bound", "upper_bound", "l_norm", "time_elapsed", "accuracy", "early_stoppping",
                             "padding_same", "Cnn-cert-core"
                                , "file_name"])


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
                 parameters.l_norm, time_elapsed, accuracy, parameters.use_early_stopping,
                 parameters.use_padding_same, parameters.use_cnnc_core
                    , parameters.file_name])
    except Exception as e:
        print("An exeption occured while writing to file")
        logging.exception(str(traceback.format_exc()) + "\n\n")
    finally:
        write_lock.release()


def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)


def setDynamicGPUAllocation():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)

    sess = tf.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras


def train_and_save_network(file_name, filters, kernels, epochs, tf_activation, batch_normalization, use_padding_same,
                           use_early_stopping, batch_size, dataset="mnist"):
    if dataset == "mnist":
        data = MNIST()
    elif dataset == "cifar":
        data = CIFAR()
    else:
        raise NameError(dataset, "is not a valid dataset")

    train_cnn(data,
              file_name=file_name,
              filters=filters,
              kernels=kernels,
              num_epochs=epochs,
              activation=tf_activation,
              bn=batch_normalization,
              batch_size=batch_size,
              use_padding_same=use_padding_same,
              use_early_stopping=use_early_stopping)


def get_lower_bound(file_name, num_image, l_norm, use_cnnc_core, activation_function, dataset):
    use_cifar = False
    if dataset == "cifar":
        use_cifar = True

    avg_lower_bound, total_time = run_cnn(file_name, num_image, l_norm, core=use_cnnc_core,
                                          activation=activation_function, cifar=use_cifar)
    return avg_lower_bound


def get_upper_bound(file_name, l_norm, num_image):
    sess = tf.keras.backend.get_session()
    tf.keras.backend.set_session(sess)
    upper_bound, time_taken = cw_attack(file_name, l_norm, sess, num_image=num_image)
    tf.keras.backend.clear_session()
    return upper_bound


def csv_contains_file(csv_file, file_name, parameters):
    with open(csv_file, "r") as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')

        first_row = next(csvreader)

        file_name_column = 0
        for i in range(len(first_row)):
            if first_row[i] == "file_name":
                file_name_column = i
                break

        cnnc_column = 0
        for i in range(len(first_row)):
            if first_row[i] == "Cnn-cert-core":
                cnnc_column = i
                break

        for row in csvreader:
            if row[file_name_column] == file_name and row[cnnc_column] == parameters.use_cnnc_core:
                return True
    return False


def file_exists(file, use_cache=True):
    if use_cache:
        start = time.time()
        if file[14:] in model_files:
            print("used {} sec to find file".format(time.time() - start))
            return True
        return False
    else:
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


def calculate_lower_bound(file_name, num_image, l_norm, use_cnnc_core, activation_function_string, dataset):
    return get_lower_bound(file_name,
                           num_image,
                           l_norm,
                           use_cnnc_core,
                           activation_function_string,
                           dataset)


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


def get_dynamic_keras_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    return config


def train_nn(parameters, file_name, filters, kernels, epochs, tf_activation, batch_normalization, use_padding_same,
             use_early_stopping, batch_size, dataset):
    try:
        #reset_cuda()
        print(datetime.now())
        print(f"\ntraining with {parameter_string(parameters)}\n", flush=True)
        sess = tf.Session()
        with sess.as_default(config=get_dynamic_keras_config()):
            train_and_save_network(file_name,
                                   filters,
                                   kernels,
                                   epochs,
                                   tf_activation,
                                   batch_normalization,
                                   use_padding_same,
                                   use_early_stopping,
                                   batch_size,
                                   dataset=dataset)
        #reset_cuda()
        gc.collect()
        sess.close()
    except Exception as e:
        keras_lock.release()
        print("Error: An exeption occured while training network", e)
        date = str(datetime.now())
        logging.exception("\n =================\n\n"
                          + date +
                          "\nThis file had an error: \n"
                          + file_name +
                          "\n" + str(traceback.format_exc()) +
                          "\n\n")


def reset_cuda():
    try:
        print("closing", cuda.get_current_device())
        cuda.close()
    except Exception as e:
        print("Could not reset cuda: ", traceback.format_exc())


def tf_reset():
    tf.reset_default_graph()

def gpu_calculations(parameters):
    try:
        if not file_exists(parameters.file_name):
            train_nn(parameters,
                     parameters.file_name,
                     parameters.filters,
                     parameters.kernels,
                     parameters.epochs,
                     parameters.tf_activation,
                     parameters.has_batch_normalization,
                     parameters.use_padding_same,
                     parameters.use_early_stopping,
                     parameters.batch_size,
                     parameters.dataset)
            print(f"\ndone training with {parameter_string(parameters)}\n", flush=True)
        else:
            print("Neural network already created - {} - {}".format(datetime.now(), parameters.file_name), flush=True)
    finally:
        keras_lock.release()


def get_accuracy_of_nn_from_csv(csv_file, file_name):
    with open(csv_file, "r") as f:
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
    semaphore.acquire()
    try:
        print(f"\nCalculating robustness of {parameter_string(parameters)}\n", flush=True)

        if not file_exists(parameters.file_name, use_cache=False):
            print("File does not exist {}".format(parameters.file_name), flush=True)
            print_parameters(parameters)
            logging.exception("\n =================\n\n"
                              + str(datetime.now()) +
                              "\nThe error was here: \n"
                              + parameter_string(parameters) +
                              "\n" + "File does not exist {}".format(parameters.file_name) +
                              "\n\n")
            return

        start_time = timer.time()

        debugprint(parameters.isDebugging, "reading results csv")
        if csv_contains_file(parameters.result_folder + parameters.result_file, parameters.file_name, parameters):
            print("Bounds already calculated for {}".format(parameters.file_name), flush=True)
            print_parameters(parameters)
            return

        debugprint(parameters.isDebugging, "reading models_meta.csv")
        accuracy = get_accuracy_of_nn_from_csv("output/models_meta.csv", parameters.file_name)

        """
        if float(accuracy) < 0.95:
            print("skiped", parameters.file_name, "as the accuracy was too low", flush=True)
            print_parameters(parameters)
            return
        """

        debugprint(parameters.isDebugging, "calculating lower bound")
        gpu_options = tf.GPUOptions(visible_device_list=_b("").decode('utf-8'))
        session_config = tf.ConfigProto(device_count={'GPU': 0}, gpu_options=gpu_options)
        sess = tf.Session(config=session_config)
        with sess.as_default():
            #may be too global:
            #cpu_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
            #tf.config.experimental.set_visible_devices(devices=cpu_devices, device_type='CPU')
            lower_bound = calculate_lower_bound(parameters.file_name,
                                                parameters.num_image,
                                                parameters.l_norm,
                                                parameters.use_cnnc_core,
                                                parameters.activation_function_string,
                                                parameters.dataset)

        time_elapsed = timer.time() - start_time

        debugprint(parameters.isDebugging, "writing to file")
        write_to_file(parameters, lower_bound, accuracy, time_elapsed)

        print("wrote to file", flush=True)
        print_parameters(parameters)

        return
    except Exception as e:
        print("Error: An exeption occured while calculating robustness", e)
        logging.exception("\n =================\n\n"
                          + str(datetime.now()) +
                          "\nThe error was here: \n"
                          + parameter_string(parameters) +
                          "\n" + str(traceback.format_exc()) +
                          "\n\n")
    finally:
        semaphore.release()
        #reset_cuda()
        gc.collect()


def pool_init(l1, l2, sema):
    global write_lock
    global keras_lock
    global semaphore
    semaphore = sema
    write_lock = l1
    keras_lock = l2


def parameter_string(parameters):
    return "depth={} filter_size={} kernel_size={} ac={} es={} pad={} cnnc={}" \
        .format(
        parameters.depth,
        parameters.filter_size,
        parameters.kernel_size,
        parameters.activation_function_string,
        parameters.use_early_stopping,
        parameters.use_padding_same,
        parameters.use_cnnc_core)


def print_parameters(parameters):
    print()
    print(parameter_string(parameters))
    print(datetime.now())
    print(
        "========================================================================================")
    print("", flush=True)


def debugprint(isDebugging, text):
    if isDebugging:
        print(text)


def main():
    _, arg1, arg2, arg3, arg4, arg5 = sys.argv
    cpu = arg1 == "cpu" or arg2 == "cpu"
    gpu = arg1 == "gpu" or arg2 == "gpu"
    debugging = arg3 == "debugging"
    path = arg4
    dataset = arg5

    set_path(path)

    if dataset != "mnist":
        set_path(dataset)

    global model_files
    try:
        model_files = os.listdir("output/models")
    except:
        model_files = []

    if not debugging:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')
        tf.logging.set_verbosity(tf.logging.ERROR)

    print("You have {} cores at your disposal.".format(multiprocessing.cpu_count()))

    if debugging:
        print(f"cpu: {cpu}")
        print(f"gpu: {gpu}")
        print(f"path: {path}/")

    if multiprocessing.cpu_count() > 36:
        processes = 36
    else:
        processes = multiprocessing.cpu_count()

    l1 = multiprocessing.Lock()
    l2 = multiprocessing.Lock()
    sema = multiprocessing.Semaphore(processes)

    cpu_pool = multiprocessing.Pool(processes, initializer=pool_init, initargs=(l1, l2, sema), maxtasksperchild=1)
    gpu_pool = multiprocessing.Pool(1, initializer=pool_init, initargs=(l1, l2, sema), maxtasksperchild=1)

    pool_init(l1, l2, sema)

    make_result_file(CnnTestParameters.result_folder, CnnTestParameters.result_file)
    logging.basicConfig(filename='log.log', level="ERROR")
    for activation_function_string in ["ada", "sigmoid", "arctan", "tanh"]:
        for kernel_size in range(3, 8, 1):
            for use_cnnc_core in [False, True]:
                for filter_size in range(2, 64, 4):
                    for has_batch_normalization in [False]:
                        for depth in range(1, 5, 1):
                            for use_early_stopping in [True, False]:
                                for use_padding_same in [True, False]:

                                    parameters = CnnTestParameters()
                                    parameters.tf_activation = get_tf_activation_function_from_string(
                                        activation_function_string)
                                    parameters.activation_function_string = activation_function_string
                                    parameters.depth = depth
                                    parameters.kernel_size = kernel_size
                                    parameters.filter_size = filter_size
                                    parameters.filters = [filter_size for i in range(depth)]
                                    parameters.kernels = [kernel_size for i in range(depth)]
                                    parameters.has_batch_normalization = has_batch_normalization
                                    parameters.isDebugging = debugging
                                    parameters.use_early_stopping = use_early_stopping
                                    parameters.use_padding_same = use_padding_same
                                    parameters.use_cnnc_core = use_cnnc_core
                                    parameters.dataset = dataset

                                    parameters.use_gpu = gpu
                                    parameters.use_cpu = cpu

                                    parameters.file_name = get_name_new_convention(parameters)

                                    if debugging:
                                        keras_lock.acquire()
                                        multithreadded_calculations(parameters)
                                    else:
                                        if parameters.use_gpu:
                                            keras_lock.acquire()
                                            gpu_pool.apply_async(gpu_calculations, (parameters,))
                                        if parameters.use_cpu:
                                            keras_lock.acquire()
                                            keras_lock.release()
                                            cpu_pool.apply_async(multithreadded_calculations, (parameters,))

    gpu_pool.close()
    cpu_pool.close()
    gpu_pool.join()
    cpu_pool.join()
    print("program finished")


def set_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)


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
    epochs = 10
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
