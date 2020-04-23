import csv
import gc
import logging
import os
import sys
from datetime import datetime

import tensorflow as tf

from Attacks.PGD_wrapper import get_accuracy as get_PGD_accuracy
from cnn_robustness_tester import get_tf_activation_function_from_string, debugprint, file_exists, print_parameters, parameter_string, set_path, get_data, \
    make_upper_bound_file, add_l_norm_to_upper_bound_file, upper_bounds_csv_contains_file, get_upper_bound_and_time, write_to_upper_bound_file
from hyper_parameters import hyper_parameters

def upper_bound_calculations(parameters):
    print(f"\nCalculating upper bound of {parameter_string(parameters)}\n", flush=True)

    parameters.tf_activation = get_tf_activation_function_from_string(
        parameters.activation_function_string, tf)

    debugprint(parameters.isDebugging, "checking if model file exists")
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

    csv_name = parameters.upper_bound_result_file
    make_upper_bound_file(csv_name)

    add_l_norm_to_upper_bound_file()
    debugprint(parameters.isDebugging, "reading results csv")
    if upper_bounds_csv_contains_file(csv_name, parameters.file_name, parameters):
        print("Upper bounds already calculated for {}".format(parameters.file_name), flush=True)
        print_parameters(parameters)
        return

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False
    sess = tf.Session(config=config)

    with sess.as_default():
        upper_bound, time_spent = get_upper_bound_and_time(parameters.file_name,
                                                           parameters.l_norm,
                                                           parameters.num_image,
                                                           sess,
                                                           dataset_data)

    debugprint(parameters.isDebugging, "writing upper bound to file")
    write_to_upper_bound_file(parameters, upper_bound, time_spent, csv_name)

    print("wrote upper bound to file", flush=True)
    print_parameters(parameters)

    gc.collect()
    return


def init_globals(data):
    global dataset_data
    dataset_data = data

def main():
    _, path, dataset = sys.argv
    set_path(path)
    set_path(dataset)
    print(path, dataset, flush=True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        model_files = os.listdir("output/models")
    except:
        model_files = []

    data = get_data(dataset)
    init_globals(data)

    params = hyper_parameters(dataset=dataset, model_files=model_files)

    for parameters in params:

        upper_bound_calculations(parameters)

        gc.collect()




if __name__ == "__main__":
    main()
