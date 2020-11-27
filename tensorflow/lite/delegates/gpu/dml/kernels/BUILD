load("//tensorflow:tensorflow.bzl", "if_windows")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

TFLITE_GPU_DEFAULT_COPTS = if_windows([
    "-DTFLITE_GPU_DML"
])

cc_library(
    name = "converter",
    srcs = ["converter.cc"],
    hdrs = ["converter.h"],
    copts = TFLITE_GPU_DEFAULT_COPTS,
    deps = [
        #":util",
        "//tensorflow/lite/delegates/gpu:dml_spi",
        #"//tensorflow/lite/delegates/gpu/cl:cl_command_queue",
        #"//tensorflow/lite/delegates/gpu/cl:cl_errors",
        #"//tensorflow/lite/delegates/gpu/cl:environment",
        #"//tensorflow/lite/delegates/gpu/cl:precision",
        #"//tensorflow/lite/delegates/gpu/cl:tensor",
        #"//tensorflow/lite/delegates/gpu/cl:tensor_type",
        #"//tensorflow/lite/delegates/gpu/cl:cl_tensor_type_util",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@dml_redist//:headers",
    ],
)
