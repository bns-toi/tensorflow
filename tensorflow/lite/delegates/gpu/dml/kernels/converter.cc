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
#include "tensorflow/lite/delegates/gpu/dml/d3d_resource.h"
#include "tensorflow/lite/delegates/gpu/dml/d3d_shader.h"

namespace tflite {
namespace gpu {
namespace dml {
namespace {

absl::Status WrapResource(DirectMlResource resource,
                          D3DResource* d3d_resource) {
  *d3d_resource = D3DResource(resource.resource, resource.data_type, resource.size_bytes);
  return absl::OkStatus();
}

class DirectMllConverterImpl : public TensorObjectConverter {
 public:
  explicit DirectMllConverterImpl(DMLDevice* device)
      : device_(device) {}
  virtual absl::Status Init(const TensorObjectDef& input_def,
                            const TensorObjectDef& output_def,
                            Environment* environment) = 0;

 protected:
  DMLDevice* device_ = nullptr;
};

class DirectMllShaderConverterImpl : public DirectMllConverterImpl {
 public:
  explicit DirectMllShaderConverterImpl(DMLDevice* device)
      : DirectMllConverterImpl(device) {}

  absl::Status Dispatch(const DirectMlResource* input,
                        const DirectMlResource* output) {
    return shader.Dispatch(device_, shape_.w, shape_.h, shape_.c, input, output);
  }

 protected:
  D3DShader shader;
  BHWC shape_;
};

bool IsSupportedDataType(DataType type) {
  return type == DataType::FLOAT16 || type == DataType::FLOAT32;
}

class FromTensorConverter : public DirectMllShaderConverterImpl {
 public:
  explicit FromTensorConverter(DMLDevice* device)
      : DirectMllShaderConverterImpl(device) {}

  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsSupportedDataType(input.data_type) &&
           IsSupportedDataType(output.data_type) &&
           // Output is always BUFFER/BHWC
           output.object_type == ObjectType::DIRECTML_RESOURCE &&
           output.data_layout == DataLayout::BHWC &&
           // BUFFER/DHWC4 ->
           input.object_type == ObjectType::DIRECTML_RESOURCE &&
           input.data_layout == DataLayout::DHWC4;
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                  input_def.dimensions.w, input_def.dimensions.c);

    return shader.Compile(device_, R"(
    Buffer<half> input : register(t0);
    RWBuffer<half> output : register(u0);

    cbuffer cbCS {
      uint height;
      uint width;
      uint channels;
    };

    [numthreads(32, 16, 1)]
    void main(uint3 blockID : SV_GroupID, uint3 threadID : SV_GroupThreadID) {
      uint x = blockID.x * 32 + threadID.x;
      uint y = blockID.y * 16 + threadID.y;
      if (x < width && y < height) {
        uint index = width * y + x;
        uint planeSize = height * width;
        for (uint c = 0; c < channels; c++) {
          output[index * channels + c] = input[index + planeSize * c];
        }
      }
    })");
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto output = absl::get_if<DirectMlResource>(&output_obj);
    if (!output || !output->resource) {
      return absl::InvalidArgumentError(
          "Missing output in from_tensor converter");
    }
    auto input = absl::get_if<DirectMlResource>(&input_obj);
    if (!input || !input->resource) {
      return absl::InvalidArgumentError("Missing input in converter");
    }
    if (input->resource == output->resource) {
      return absl::InvalidArgumentError("Can not execute inplace conversion");
    }

    return Dispatch(input, output);
  }
};

class ToTensorConverter : public DirectMllShaderConverterImpl {
 public:
  explicit ToTensorConverter(DMLDevice* device)
      : DirectMllShaderConverterImpl(device) {}

  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsSupportedDataType(input.data_type) &&
           IsSupportedDataType(output.data_type) &&
           // Input is always RESOURCE/BHWC
           input.object_type == ObjectType::DIRECTML_RESOURCE &&
           input.data_layout == DataLayout::BHWC &&
           // RESOURCE/DHWC4 ->
           output.object_type == ObjectType::DIRECTML_RESOURCE &&
           output.data_layout == DataLayout::DHWC4;
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                  input_def.dimensions.w, input_def.dimensions.c);

    return shader.Compile(device_, R"(
    Buffer<half> input : register(t0);
    RWBuffer<half> output : register(u0);

    cbuffer cbCS {
      uint height;
      uint width;
      uint channels;
    };

    [numthreads(32, 16, 1)]
    void main(uint3 blockID : SV_GroupID, uint3 threadID : SV_GroupThreadID) {
      uint x = blockID.x * 32 + threadID.x;
      uint y = blockID.y * 16 + threadID.y;
      if (x < width && y < height) {
        uint index = width * y + x;
        uint planeSize = height * width;
        for (uint c = 0; c < channels; c++) {
          output[index + planeSize * c] = input[index * channels + c];
        }
      }
    })");
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto output = absl::get_if<DirectMlResource>(&output_obj);
    if (!output || !output->resource) {
      return absl::InvalidArgumentError(
          "Missing output in from_tensor converter");
    }
    auto input = absl::get_if<DirectMlResource>(&input_obj);
    if (!input || !input->resource) {
      return absl::InvalidArgumentError("Missing input in converter");
    }
    if (input->resource == output->resource) {
      return absl::InvalidArgumentError("Can not execute inplace conversion");
    }

    return Dispatch(input, output);
  }
};

// Copies data from/to CPU into a tensor.
class CpuCopier : public DirectMllConverterImpl {
 public:
  explicit CpuCopier(DMLDevice* device)
      : DirectMllConverterImpl(device) {}

  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return input.data_type == output.data_type &&
           input.data_layout == output.data_layout &&
           ((input.object_type == ObjectType::CPU_MEMORY &&
             output.object_type == ObjectType::DIRECTML_RESOURCE) ||
            (output.object_type == ObjectType::CPU_MEMORY &&
             input.object_type == ObjectType::DIRECTML_RESOURCE));
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
      auto resource_output = absl::get_if<DirectMlResource>(&output_obj);
      if (resource_output) {
        D3DResource d3d_resource;
        RETURN_IF_ERROR(WrapResource(*resource_output, &d3d_resource));
        return d3d_resource.Write(device_,
            absl::MakeConstSpan(static_cast<const uint8_t*>(cpu_input->data),
                                cpu_input->size_bytes));
      }
    } else if (cpu_output) {
      auto resource_input = absl::get_if<DirectMlResource>(&input_obj);
      if (resource_input) {
        D3DResource d3d_resource;
        RETURN_IF_ERROR(WrapResource(*resource_input, &d3d_resource));
        return d3d_resource.Read(device_, absl::MakeSpan(
            static_cast<uint8_t*>(cpu_output->data), cpu_output->size_bytes));
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
    DMLDevice* device = environment_->device();
    /*if (TrivialCopier::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<TrivialCopier>();
    } else*/ if (CpuCopier::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<CpuCopier>(device);
    } else if (FromTensorConverter::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<FromTensorConverter>(device);
    } else if (ToTensorConverter::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<ToTensorConverter>(device);
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
