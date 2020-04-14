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

def hyper_parameters(dataset,
                     model_files,
                     debugging=False,
                     gpu=False,
                     cpu=False):
    filter_size_range = range(8, 81, 8)
    depth_range = range(1, 6, 1)
    kernel_size_range = range(3, 8, 1)
    cnnc_choices = [False]
    bn_choices = [False]

    parameter_list = []
    for l_norm in ["2", "1"]:
        for activation_function_string in ["ada", "sigmoid"]:
            for use_cnnc_core in cnnc_choices:
                for use_padding_same in [False]:
                    for use_early_stopping in [True]:
                        for has_batch_normalization in bn_choices:
                            for kernel_size in kernel_size_range:
                                for depth in depth_range:
                                    for filter_size in filter_size_range:
                                        parameters = CnnTestParameters()
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
                                        parameters.model_files = model_files
                                        parameters.l_norm = l_norm

                                        parameters.use_gpu = gpu
                                        parameters.use_cpu = cpu

                                        parameters.file_name = get_name_new_convention(parameters)
                                        parameter_list.append(parameters)
    return parameter_list
