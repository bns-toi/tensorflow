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

#include "tensorflow/lite/delegates/gpu/dml/api.h"

#include <algorithm>
#include <cstring>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/dml/dml_device.h"
#include "tensorflow/lite/delegates/gpu/dml/environment.h"
#include "tensorflow/lite/delegates/gpu/dml/kernels/converter.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace dml {
namespace {

class InferenceBuilderImpl : public InferenceBuilder {
 public:
  explicit InferenceBuilderImpl(Environment* environment)
      : environment_(environment) {}

  absl::Status Initialize(const InferenceOptions& options,
                          const InferenceEnvironmentOptions& env_options,
                          const GraphFloat32& graph) {
    

    inputs_ = LinkTensors(graph, graph.inputs());
    outputs_ = LinkTensors(graph, graph.outputs());
    return absl::OkStatus();
  }

  std::vector<TensorObjectDef> inputs() const override {
    return GetExternalDefinitions(inputs_);
  }

  std::vector<TensorObjectDef> outputs() const override {
    return GetExternalDefinitions(outputs_);
  }

  absl::Status SetInputShape(int index, const Dimensions& dimensions) override {
    if (index < 0 || index >= inputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    return absl::UnimplementedError("Changing input shapes is not supported");
  }

  absl::Status SetInputObjectDef(int index, ObjectDef new_def) override {
    if (index < 0 || index >= inputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    auto def = inputs_[index];
    def.external_def.object_def = new_def;
/*    if (!tie_factory_->IsSupported(def)) {
      return absl::InvalidArgumentError(
          "New object definition is not supported.");
    }*/
    inputs_[index] = def;
    return absl::OkStatus();
  }

  absl::Status SetOutputObjectDef(int index, ObjectDef new_def) override {
    if (index < 0 || index >= outputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    auto def = outputs_[index];
    def.external_def.object_def = new_def;
/*    if (!tie_factory_->IsSupported(def)) {
      return absl::InvalidArgumentError(
          "New object definition is not supported.");
    }*/
    outputs_[index] = def;
    return absl::OkStatus();
  }

  absl::Status Build(std::unique_ptr<InferenceRunner>* runner) override {
    return absl::OkStatus();
  }

 private:
  // Links internal tensors with external user-facing objects.
  std::vector<TensorTieDef> LinkTensors(const GraphFloat32& graph,
                                        const std::vector<Value*>& values) {
    std::vector<TensorTieDef> links;
    links.reserve(values.size());
    for (const auto& value : values) {
      TensorObjectDef external_def;
      const auto& shape = value->tensor.shape;
      external_def.dimensions = Dimensions(shape.b, shape.h, shape.w, shape.c);
      external_def.object_def.data_type = DataType::FLOAT32;
      external_def.object_def.data_layout = DataLayout::DHWC4;
//      external_def.object_def.object_type = gpu::ObjectType::OPENGL_SSBO;

      TensorObjectDef internal_def = external_def;
      external_def.object_def.user_provided = true;
      internal_def.object_def.user_provided = false;

      AccessType access =
          graph.IsGraphInput(value->id) ? AccessType::READ : AccessType::WRITE;
      links.push_back({value->id, access, internal_def, external_def});
    }
    return links;
  }

  static std::vector<TensorObjectDef> GetExternalDefinitions(
      const std::vector<TensorTieDef>& links) {
    std::vector<TensorObjectDef> defs;
    defs.reserve(links.size());
    for (auto& desc : links) {
      defs.push_back(desc.external_def);
    }
    return defs;
  }

  Environment* environment_;

  std::vector<TensorTieDef> inputs_;
  std::vector<TensorTieDef> outputs_;
};

class InferenceEnvironmentImpl : public InferenceEnvironment {
 public:
  explicit InferenceEnvironmentImpl(const InferenceEnvironmentOptions& options)
      : options_(options) {}

  absl::Status Init() {
    DMLDevice device;
    RETURN_IF_ERROR(CreateDefaultGPUDevice(&device));

    device.Init();

    environment_ = Environment(std::move(device));
    return environment_.Init();
  }

  absl::Status NewInferenceBuilder(
      const InferenceOptions& options, GraphFloat32 model,
      std::unique_ptr<InferenceBuilder>* builder) final {
    if (!IsValid(options)) {
      return absl::InvalidArgumentError("InferenceOptions are invalid.");
    }
    InferenceOptions resolved_options = options;
    ResolveAutoPriority(&resolved_options);
#if 0
    if (environment_.program_cache() &&
        !options_.serialized_binary_cache.empty()) {
      // Ignore returned error. Cache is discarded.
      environment_.program_cache()
          ->AddSerializedCache(environment_.context(), environment_.device(),
                               options_.serialized_binary_cache)
          .IgnoreError();
    }
#endif

    auto builder_impl = absl::make_unique<InferenceBuilderImpl>(&environment_);
    RETURN_IF_ERROR(
        builder_impl->Initialize(resolved_options, options_, model));
    *builder = std::move(builder_impl);
    return absl::OkStatus();
  }

  const InferenceEnvironmentProperties& properties() const {
    return properties_;
  }

 private:
  const InferenceEnvironmentOptions options_;
  Environment environment_;
  InferenceEnvironmentProperties properties_;
};

}  // namespace

absl::Status NewInferenceEnvironment(
    const InferenceEnvironmentOptions& options,
    std::unique_ptr<InferenceEnvironment>* environment,
    InferenceEnvironmentProperties* properties) {
  auto env_impl = absl::make_unique<InferenceEnvironmentImpl>(options);
  absl::Status status = env_impl->Init();
  if (properties) {
    *properties = env_impl->properties();
  }
  RETURN_IF_ERROR(status);
  *environment = std::move(env_impl);
  return absl::OkStatus();
}

}  // namespace dml
}  // namespace gpu
}  // namespace tflite