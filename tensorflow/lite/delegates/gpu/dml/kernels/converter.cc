/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/dml/kernels/converter.h"

#include <algorithm>
#include <array>
#include <string>

#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace dml {
namespace {

class DirectMllConverterImpl : public TensorObjectConverter {
 public:
  virtual absl::Status Init(const TensorObjectDef& input_def,
                            const TensorObjectDef& output_def,
                            Environment* environment) = 0;

 protected:
};

bool IsSupportedDataType(DataType type) {
  return /* == DataType::FLOAT16 || */type == DataType::FLOAT32;
}

class FromTensorConverter : public DirectMllConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsSupportedDataType(input.data_type) &&
           IsSupportedDataType(output.data_type) &&
           // Output is always BUFFER/BHWC
           output.object_type == ObjectType::DIRECTML_BUFFER &&
           output.data_layout == DataLayout::BHWC &&
           // BUFFER/DHWC4 ->
           input.object_type == ObjectType::DIRECTML_BUFFER &&
           input.data_layout == DataLayout::DHWC4;
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    return absl::OkStatus();
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto output = absl::get_if<DirectMlBuffer>(&output_obj);
    if (!output/* || !output->memobj*/) {
      return absl::InvalidArgumentError(
          "Missing output in from_tensor converter");
    }
    return absl::InvalidArgumentError("Missing input in from_tensor converter");
  }
};

class ToTensorConverter : public DirectMllConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsSupportedDataType(input.data_type) &&
           IsSupportedDataType(output.data_type) &&
           // Input is always BUFFER/BHWC
           input.object_type == ObjectType::DIRECTML_BUFFER &&
           input.data_layout == DataLayout::BHWC &&
           // BUFFER/DHWC4 ->
           output.object_type == ObjectType::DIRECTML_BUFFER &&
           output.data_layout == DataLayout::DHWC4;
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    return absl::OkStatus();
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto input = absl::get_if<DirectMlBuffer>(&input_obj);
    if (!input/* || !input->memobj*/) {
      return absl::InvalidArgumentError("Missing input in to_tensor converter");
    }
    return absl::InvalidArgumentError("Missing input in to_tensor converter");
  }
};

// Copies data from/to CPU into a tensor.
class CpuCopier : public DirectMllConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return input.data_type == output.data_type &&
           input.data_layout == output.data_layout &&
           ((input.object_type == ObjectType::CPU_MEMORY &&
             output.object_type == ObjectType::DIRECTML_BUFFER) ||
            (output.object_type == ObjectType::CPU_MEMORY &&
             input.object_type == ObjectType::DIRECTML_BUFFER));
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    return absl::OkStatus();
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto cpu_input = absl::get_if<CpuMemory>(&input_obj);
    auto cpu_output = absl::get_if<CpuMemory>(&output_obj);
    if (cpu_input) {
      auto buffer_output = absl::get_if<DirectMlBuffer>(&output_obj);
      if (buffer_output) {
          // TODO
      }
    } else if (cpu_output) {
      auto buffer_input = absl::get_if<DirectMlBuffer>(&input_obj);
      if (buffer_input) {
        // TODO
      }
    }
    return absl::InternalError("Unexpected object");
  }
};

class DirectMlTensorConverterBuilder : public TensorObjectConverterBuilder {
 public:
  explicit DirectMlTensorConverterBuilder(Environment* environment)
      : environment_(environment) {}

  bool IsSupported(const TensorObjectDef& input,
                   const TensorObjectDef& output) const final {
    const auto& input_def = input.object_def;
    const auto& output_def = output.object_def;
    return input.dimensions == output.dimensions &&
           (/*TrivialCopier::IsSupported(input_def, output_def) ||*/
            CpuCopier::IsSupported(input_def, output_def) ||
            FromTensorConverter::IsSupported(input_def, output_def) ||
            ToTensorConverter::IsSupported(input_def, output_def));
  }

  absl::Status MakeConverter(
      const TensorObjectDef& input, const TensorObjectDef& output,
      std::unique_ptr<TensorObjectConverter>* converter) final {
    std::unique_ptr<DirectMllConverterImpl> impl;
    const auto& input_def = input.object_def;
    const auto& output_def = output.object_def;
/*    if (TrivialCopier::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<TrivialCopier>();
    } else*/ if (CpuCopier::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<CpuCopier>();
    } else if (FromTensorConverter::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<FromTensorConverter>();
    } else if (ToTensorConverter::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<ToTensorConverter>();
    } else {
      return absl::UnimplementedError("Unsupported conversion");
    }
    RETURN_IF_ERROR(impl->Init(input, output, environment_));
    *converter = std::move(impl);
    return absl::OkStatus();
  }

  Environment* environment_;
};

}  // namespace

std::unique_ptr<TensorObjectConverterBuilder> NewConverterBuilder(
    Environment* environment) {
  return absl::make_unique<DirectMlTensorConverterBuilder>(environment);
}

}  // namespace dml
}  // namespace gpu
}  // namespace tflite
