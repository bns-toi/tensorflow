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
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/dml/dml_device.h"
#include "tensorflow/lite/delegates/gpu/dml/environment.h"
#include "tensorflow/lite/delegates/gpu/dml/runtime.h"
#include "tensorflow/lite/delegates/gpu/dml/kernels/converter.h"

using Microsoft::WRL::ComPtr;

namespace tflite {
namespace gpu {
namespace dml {
namespace {

absl::Status MaybeAllocateD3D12Resource(DMLDevice* device, const TensorObjectDef& def, ComPtr<ID3D12Resource>& resource) {
  if (def.object_def.object_type != gpu::ObjectType::DIRECTML_BUFFER) {
    return absl::InvalidArgumentError("Tensor object is not DirectML Buffer");
  }

  auto& dims = def.dimensions;
  UINT tensor_sizes[4] = {dims.b, dims.h, dims.w, dims.c};
  ::dml::TensorDesc::Dimensions dimensions(std::begin(tensor_sizes), std::end(tensor_sizes));
  ::dml::TensorDesc desc;
  switch (def.object_def.data_type) {
    case DataType::FLOAT32:
      desc = ::dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, dimensions);
      break;
    case DataType::FLOAT16:
      desc = ::dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT16, dimensions);
      break;
    default:
      return absl::InternalError(
          "Unable to create new GL SSBO. Unsupported data type.");
  };

  UINT64 tensor_buffer_size = desc.totalTensorSizeInBytes;

  DML_CHECK_SUCCEEDED(device->d3d_device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
    D3D12_HEAP_FLAG_NONE,
    &CD3DX12_RESOURCE_DESC::Buffer(tensor_buffer_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
     D3D12_RESOURCE_STATE_COPY_DEST,
     nullptr,
     IID_PPV_ARGS(&resource)));

  return absl::OkStatus();
}

// Does one-step conversion between internal and external objects.
// It may also allocate external objects if requested.
class DefaultTensorTie : public TensorTie {
 public:
  DefaultTensorTie(const TensorTieDef& def, TensorObject internal_obj)
      : TensorTie(def), internal_obj_(internal_obj) {}

  static bool IsSupported(
      const TensorTieDef& def,
      const TensorObjectConverterBuilder& converter_builder) {
    auto object_type = def.external_def.object_def.object_type;
    return (object_type == ObjectType::DIRECTML_BUFFER ||
            object_type == ObjectType::DIRECTML_TEXTURE ||
            object_type == ObjectType::CPU_MEMORY) &&
           converter_builder.IsSupported(def.internal_def, def.external_def) &&
           converter_builder.IsSupported(def.external_def, def.internal_def);
  }

  static absl::Status New(const TensorTieDef& def,
                          TensorObjectConverterBuilder* converter_builder,
                          Environment* env, std::unique_ptr<TensorTie>* tie) {
    auto tie_impl = absl::make_unique<DefaultTensorTie>(def, TensorObject{});
    RETURN_IF_ERROR(tie_impl->Init(converter_builder, env));
    *tie = std::move(tie_impl);
    return absl::OkStatus();
  }

  absl::Status CopyToExternalObject() final {
    if (!converter_to_) {
      return absl::UnavailableError("Conversion is not available");
    }
    return converter_to_->Convert(internal_obj_, GetExternalObject());
  }

  absl::Status CopyFromExternalObject() final {
    if (!converter_from_) {
      return absl::UnavailableError("Conversion is not available");
    }
    return converter_from_->Convert(GetExternalObject(), internal_obj_);
  }

  absl::Status SetExternalObject(TensorObject obj) final {
    if (!def().external_def.object_def.user_provided) {
      return absl::InvalidArgumentError("External object is read-only");
    }
    if (!IsValid(def().external_def, obj)) {
      return absl::InvalidArgumentError("Given object is not valid");
    }
    external_obj_ = obj;
    return absl::OkStatus();
  }

  TensorObject GetExternalObject() final { return external_obj_; }

 private:
  absl::Status Init(TensorObjectConverterBuilder* converter_builder,
                    Environment* env) {

    RETURN_IF_ERROR(converter_builder->MakeConverter(
        def().external_def, def().internal_def, &converter_from_));
    RETURN_IF_ERROR(converter_builder->MakeConverter(
        def().internal_def, def().external_def, &converter_to_));

    return MaybeAllocateExternalObject(env);
  }

  absl::Status MaybeAllocateExternalObject(Environment* env) {
    const TensorObjectDef& d = def().external_def;
    if (d.object_def.user_provided) {
      return absl::OkStatus();
    }

    switch (d.object_def.object_type) {
      case ObjectType::CPU_MEMORY: {
        size_t bytes_size = NumElements(d) * SizeOf(d.object_def.data_type);
        cpu_memory_.resize(bytes_size);
        external_obj_ = CpuMemory{cpu_memory_.data(), cpu_memory_.size()};
        break;
      }
      case ObjectType::DIRECTML_BUFFER: {
        ComPtr<ID3D12Resource> resource;
        RETURN_IF_ERROR(MaybeAllocateD3D12Resource(env->GetDevicePtr(), d, resource));
        external_obj_ = DirectMlBuffer{resource};
        break;
      }
      default:
        return absl::InternalError("Unexpected object type");
    }
    return absl::OkStatus();
  }

  const TensorObject internal_obj_;
  TensorObject external_obj_;
  std::vector<uint8_t> cpu_memory_;
  std::unique_ptr<TensorObjectConverter> converter_to_;
  std::unique_ptr<TensorObjectConverter> converter_from_;
};

// Copies data to intermediate DirectML buffer and then does two step conversion.
// It drives the following cases were one-step conversion is not supported:
//   - CPU BHWC -> CL buffer BHWC -> CL texture DHWC4.
class TwoStepTensorTie : public TensorTie {
 public:
  explicit TwoStepTensorTie(const TensorTieDef& def) : TensorTie(def) {}

  static bool IsSupported(
      const TensorTieDef& def,
      const TensorObjectConverterBuilder& converter_builder) {
    auto defs = MakeOuterInnerDefs(def);
    return DefaultTensorTie::IsSupported(defs.first, converter_builder) &&
           DefaultTensorTie::IsSupported(defs.second, converter_builder);
  }

  static absl::Status New(const TensorTieDef& def,
                          TensorObjectConverterBuilder* converter_builder,
                          Environment* env, std::unique_ptr<TensorTie>* tie) {
    auto tie_impl = absl::make_unique<TwoStepTensorTie>(def);
    RETURN_IF_ERROR(tie_impl->Init(converter_builder, env));
    *tie = std::move(tie_impl);
    return absl::OkStatus();
  }

  absl::Status CopyToExternalObject() final {
    RETURN_IF_ERROR(inner_tie_->CopyToExternalObject());
    return outer_tie_->CopyToExternalObject();
  }

  absl::Status CopyFromExternalObject() final {
    RETURN_IF_ERROR(outer_tie_->CopyFromExternalObject());
    return inner_tie_->CopyFromExternalObject();
  }

  absl::Status SetExternalObject(TensorObject obj) final {
    return outer_tie_->SetExternalObject(obj);
  }

  TensorObject GetExternalObject() final {
    return outer_tie_->GetExternalObject();
  }

 private:
  static std::pair<TensorTieDef, TensorTieDef> MakeOuterInnerDefs(
      const TensorTieDef& def) {
    TensorTieDef outer_def;
    outer_def.external_def = def.external_def;
    outer_def.internal_def = def.external_def;
    outer_def.internal_def.object_def.object_type = ObjectType::DIRECTML_BUFFER;
    outer_def.internal_def.object_def.user_provided = true;

    TensorTieDef inner_def;
    inner_def.external_def = outer_def.internal_def;
    inner_def.external_def.object_def.user_provided = false;
    inner_def.internal_def = def.internal_def;
    return std::make_pair(outer_def, inner_def);
  }

  absl::Status Init(TensorObjectConverterBuilder* converter_builder,
                    Environment* env) {
    auto defs = MakeOuterInnerDefs(def());
    RETURN_IF_ERROR(DefaultTensorTie::New(defs.second,
                                          converter_builder, env, &inner_tie_));
    return DefaultTensorTie::New(defs.first,
                                 converter_builder, env, &outer_tie_);
  }

  std::unique_ptr<TensorTie> inner_tie_;
  std::unique_ptr<TensorTie> outer_tie_;
};

class TensorTieFactory {
 public:
  TensorTieFactory(Environment* env)
      : env_(*env),
        converter_builder_(NewConverterBuilder(env)) {}

  bool IsSupported(const TensorTieDef& def) const {
    return IsValid(def.external_def.object_def) &&
           (/*NoopTensorTie::IsSupported(def) ||*/
            DefaultTensorTie::IsSupported(def, *converter_builder_) ||
            TwoStepTensorTie::IsSupported(def, *converter_builder_));
  }
  
  absl::Status NewTensorTie(const TensorTieDef& def,
                            std::unique_ptr<TensorTie>* tie) {
    auto converter = converter_builder_.get();
    if (DefaultTensorTie::IsSupported(def, *converter)) {
      return DefaultTensorTie::New(def, converter, &env_, tie);
    }
    if (TwoStepTensorTie::IsSupported(def, *converter)) {
      return TwoStepTensorTie::New(def, converter, &env_, tie);
    }
    return absl::UnimplementedError("Unsupported tensor tie definition.");
  }

 private:
  Environment& env_;
  std::unique_ptr<TensorObjectConverterBuilder> converter_builder_;
};

class InferenceRunnerImpl : public InferenceRunner {
 public:
  InferenceRunnerImpl(Environment* environment,
                      std::unique_ptr<Runtime> runtime)
      : runtime_(std::move(runtime)) {}

  absl::Status Initialize(const std::vector<TensorTieDef>& inputs,
                          const std::vector<TensorTieDef>& outputs,
                          TensorTieFactory* factory) {
    RETURN_IF_ERROR(LinkTensors(inputs, factory, &inputs_));
    return LinkTensors(outputs, factory, &outputs_);
  }

  std::vector<TensorObjectDef> inputs() const override {
    return GetExternalDefinitions(inputs_);
  }

  std::vector<TensorObjectDef> outputs() const override {
    return GetExternalDefinitions(outputs_);
  }

  absl::Status GetInputObject(int index, TensorObject* object) override {
    if (index < 0 || index >= inputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    *object = inputs_[index]->GetExternalObject();
    return absl::OkStatus();
  }

  absl::Status GetOutputObject(int index, TensorObject* object) override {
    if (index < 0 || index >= outputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    *object = outputs_[index]->GetExternalObject();
    return absl::OkStatus();
  }

  absl::Status SetInputObject(int index, TensorObject object) override {
    if (index < 0 || index >= inputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    return inputs_[index]->SetExternalObject(object);
  }

  absl::Status SetOutputObject(int index, TensorObject object) override {
    if (index < 0 || index >= outputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    return outputs_[index]->SetExternalObject(object);
  }

  absl::Status Run() override {
    for (auto& obj : inputs_) {
      RETURN_IF_ERROR(obj->CopyFromExternalObject());
    }
//    RETURN_IF_ERROR(context_->AddToQueue(queue_));
//    clFlush(queue_->queue());
    for (auto& obj : outputs_) {
      RETURN_IF_ERROR(obj->CopyToExternalObject());
    }
    return absl::OkStatus();
  }

 private:
  static absl::Status LinkTensors(
      const std::vector<TensorTieDef>& defs, TensorTieFactory* factory,
      std::vector<std::unique_ptr<TensorTie>>* objects) {
    objects->reserve(defs.size());
    for (auto& def : defs) {
      std::unique_ptr<TensorTie> object;
      RETURN_IF_ERROR(factory->NewTensorTie(def, &object));
      objects->push_back(std::move(object));
    }
    return absl::OkStatus();
  }

  static std::vector<TensorObjectDef> GetExternalDefinitions(
      const std::vector<std::unique_ptr<TensorTie>>& objects) {
    std::vector<TensorObjectDef> defs;
    defs.reserve(objects.size());
    for (auto& obj : objects) {
      defs.push_back(obj->def().external_def);
    }
    return defs;
  }

  std::unique_ptr<Runtime> runtime_;
  std::vector<std::unique_ptr<TensorTie>> inputs_;
  std::vector<std::unique_ptr<TensorTie>> outputs_;
};

class InferenceBuilderImpl : public InferenceBuilder {
 public:
  explicit InferenceBuilderImpl(Environment* environment, GraphFloat32 graph)
      : environment_(environment),
        graph_(std::move(graph)) {}

  absl::Status Initialize(const InferenceOptions& options,
                          const InferenceEnvironmentOptions& env_options) {
    
    tie_factory_ = absl::make_unique<TensorTieFactory>(environment_);

    inputs_ = LinkTensors(graph_.inputs());
    outputs_ = LinkTensors(graph_.outputs());
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
    if (!tie_factory_->IsSupported(def)) {
      return absl::InvalidArgumentError(
          "New object definition is not supported.");
    }
    inputs_[index] = def;
    return absl::OkStatus();
  }

  absl::Status SetOutputObjectDef(int index, ObjectDef new_def) override {
    if (index < 0 || index >= outputs_.size()) {
      return absl::OutOfRangeError("Index is out of range");
    }
    auto def = outputs_[index];
    def.external_def.object_def = new_def;
    if (!tie_factory_->IsSupported(def)) {
      return absl::InvalidArgumentError(
          "New object definition is not supported.");
    }
    outputs_[index] = def;
    return absl::OkStatus();
  }

  absl::Status Build(std::unique_ptr<InferenceRunner>* runner) override {
    auto runtime =
        absl::make_unique<Runtime>();
    auto runner_impl = absl::make_unique<InferenceRunnerImpl>(
        environment_, std::move(runtime));
    RETURN_IF_ERROR(
        runner_impl->Initialize(inputs_, outputs_, tie_factory_.get()));
    *runner = std::move(runner_impl);
    return absl::OkStatus();
  }

 private:
  // Links internal tensors with external user-facing objects.
  std::vector<TensorTieDef> LinkTensors(const std::vector<Value*>& values) {
    std::vector<TensorTieDef> links;
    links.reserve(values.size());
    for (const auto& value : values) {
      TensorObjectDef external_def;
      const auto& shape = value->tensor.shape;
      external_def.dimensions = Dimensions(shape.b, shape.h, shape.w, shape.c);
      external_def.object_def.data_type = DataType::FLOAT32;
      external_def.object_def.data_layout = DataLayout::DHWC4;
      external_def.object_def.object_type = gpu::ObjectType::DIRECTML_BUFFER;

      TensorObjectDef internal_def = external_def;
      external_def.object_def.user_provided = true;
      internal_def.object_def.user_provided = false;
      AccessType access =
          graph_.IsGraphInput(value->id) ? AccessType::READ : AccessType::WRITE;
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
  GraphFloat32 graph_;

  std::vector<TensorTieDef> inputs_;
  std::vector<TensorTieDef> outputs_;
  std::unique_ptr<TensorTieFactory> tie_factory_;
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

    auto builder_impl = absl::make_unique<InferenceBuilderImpl>(
        &environment_, std::move(model));
    RETURN_IF_ERROR(
        builder_impl->Initialize(resolved_options, options_));
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