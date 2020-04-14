

def hyper_parameters(cnnc_choices,
                     bn_choices,
                     kernel_size_range,
                     depth_range,
                     filter_size_range,
                     dataset,
                     debugging,
                     model_files,
                     gpu,
                     cpu,
                     get_name_new_convention,
                     CnnTestParameters):
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
