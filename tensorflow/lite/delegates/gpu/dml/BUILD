load("//tensorflow:tensorflow.bzl", "if_windows")
load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
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
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    copts = TFLITE_GPU_DEFAULT_COPTS,
    deps = [
        #":cl_command_queue",
        #":cl_errors",
        #":cl_event",
        ":environment",
        #":inference_context",
        #":precision",
        #":tensor",
        #":tensor_type",
        #":cl_tensor_type_util",
        "//tensorflow/lite/delegates/gpu:dml_api",
        "//tensorflow/lite/delegates/gpu/dml/kernels:converter",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dml_device",
    srcs = ["dml_device.cc"],
    hdrs = [
		"dml_device.h",
		"dml_common.h",
		"dml_error_handling.h",
    ],
    deps = [
        #":util",
        "//tensorflow/core/platform:logging",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
        "//third_party/dml:dx_helpers",
        "@dml_redist//:headers",
        "//third_party/dml:winadapter",
    ],
	linkopts = [
        "d3d12.lib", 
        "dxgi.lib", 
        "directml.lib",
    ]
)

cc_library(
    name = "environment",
    srcs = ["environment.cc"],
    hdrs = ["environment.h"],
    deps = [
        ":dml_device",
        #":precision",
        #":program_cache",
        #":tensor",
        #":tensor_type",
        #":util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
    ],
)
