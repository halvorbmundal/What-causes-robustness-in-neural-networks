import os
from enum import Enum


class NnArchitecture(Enum):
    ONLY_CNN = "only_cnn"


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
    width = "null"
    upper_bound = None
    result_folder = 'output/results/'
    result_file = 'results.csv'
    upper_bound_result_file = 'upper_bound.csv'


def boolToString(b):
    if b:
        return "T"
    else:
        return "F"


def get_name_new_convention(parameter_class, dir):
    directory = dir
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


def boundary_test_hyperparameters(dataset,
                                  model_files,
                                  debugging=False):
    filter_size_range = range(8, 73, 16)
    depth_range = range(1, 6, 1)
    parameter_list = []



    for filter_size in filter_size_range:
        for depth in depth_range:
            parameters = CnnTestParameters()
            parameters.result_file = 'adversarial_results.csv'
            parameters.result_folder = 'adv_results/'
            parameters.activation_function_string = "ada"
            parameters.depth = depth
            parameters.kernel_size = 3
            parameters.filter_size = filter_size
            parameters.filters = [filter_size for i in range(depth)]
            parameters.kernels = [parameters.kernel_size for i in range(depth)]
            parameters.has_batch_normalization = False
            parameters.is_debugging = debugging
            parameters.use_early_stopping = True
            parameters.use_padding_same = False
            parameters.use_cnnc_core = False
            parameters.dataset = dataset
            parameters.model_files = model_files
            parameters.l_norm = "i"
            parameters.train_on_adversaries = True

            parameters.file_name = get_name_new_convention(parameters, "adv_models")
            parameters.file_name = parameters.file_name + "_adv"
            parameter_list.append(parameters)
    return parameter_list


def hyper_parameters(dataset,
                     model_files,
                     debugging=False,
                     train_on_adversaries=False):
    filter_size_range = range(4, 80, 4)
    depth_range = range(1, 6, 1)
    kernel_size_range = range(3, 8, 1)
    cnnc_choices = [False]
    bn_choices = [False]

    if train_on_adversaries:
        return boundary_test_hyperparameters(dataset,
                                             model_files,
                                             debugging=debugging)

    parameter_list = []
    for filter_size in filter_size_range:
        for kernel_size in kernel_size_range:
            for l_norm in ["i", "2", "1"]:
                for activation_function_string in ["ada", "sigmoid", "tanh", "arctan"]:
                    for use_cnnc_core in cnnc_choices:
                        for use_padding_same in [False]:
                            for use_early_stopping in [True]:
                                for has_batch_normalization in bn_choices:
                                    for depth in depth_range:
                                        parameters = CnnTestParameters()
                                        parameters.activation_function_string = activation_function_string
                                        parameters.depth = depth
                                        parameters.kernel_size = kernel_size
                                        parameters.filter_size = filter_size
                                        parameters.filters = [filter_size for i in range(depth)]
                                        parameters.kernels = [kernel_size for i in range(depth)]
                                        parameters.has_batch_normalization = has_batch_normalization
                                        parameters.is_debugging = debugging
                                        parameters.use_early_stopping = use_early_stopping
                                        parameters.use_padding_same = use_padding_same
                                        parameters.use_cnnc_core = use_cnnc_core
                                        parameters.dataset = dataset
                                        parameters.model_files = model_files
                                        parameters.l_norm = l_norm
                                        parameters.train_on_adversaries = False

                                        parameters.file_name = get_name_new_convention(parameters, "output/models")
                                        parameter_list.append(parameters)
    return parameter_list
