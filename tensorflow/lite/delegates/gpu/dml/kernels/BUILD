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
        "//tensorflow/lite/delegates/gpu:dml_spi",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/dml:d3d_shader",
        "@dml_redist//:headers",
    ],
)
