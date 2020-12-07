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
        ":environment",
        ":object_manager",
        ":runtime",
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

cc_library(
    name = "runtime",
    srcs = ["runtime.cc"],
    hdrs = ["runtime.h"],
    deps = [
    	"//tensorflow/lite/delegates/gpu/common:model",
    	"//tensorflow/lite/delegates/gpu/common:types",
    	"@com_google_absl//absl/memory",
        "//third_party/dml:dx_helpers",
        "@dml_redist//:headers",
    ],
)

cc_library(
    name = "d3d_resource",
    srcs = ["d3d_resource.cc"],
    hdrs = ["d3d_resource.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:span",
        "//third_party/dml:dx_helpers",
        "@dml_redist//:headers",
        "//third_party/dml:winadapter",
    ],
)

cc_library(
    name = "object_manager",
    srcs = ["object_manager.cc"],
    hdrs = ["object_manager.h"],
    deps = [
        ":d3d_resource",
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)
