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
#ifdef _DEBUG
#include <chrono>
#endif // _DEBUG

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/dml/environment.h"
#include "tensorflow/lite/delegates/gpu/dml/runtime.h"
#include "tensorflow/lite/delegates/gpu/dml/kernels/converter.h"
#include "tensorflow/lite/delegates/gpu/dml/object_manager.h"

using Microsoft::WRL::ComPtr;

namespace tflite {
namespace gpu {
namespace dml {
namespace {
absl::Status WrapResource(DirectMlResource resource,
                          D3DResource* d3d_resource) {
  *d3d_resource =
      D3DResource(resource.resource, resource.data_type, resource.size_bytes);
  return absl::OkStatus();
}

absl::Status MaybeAllocateD3D12Resource(DMLDevice* device,
                                        const TensorObjectDef& def,
                                        bool external, AccessType access_type,
                                        D3DResource* resource, wchar_t* name) {                                        
  if (def.object_def.object_type != gpu::ObjectType::DIRECTML_RESOURCE) {
    return absl::InvalidArgumentError("Tensor object is not D3D Resource");
  }

  auto& dims = def.dimensions;
  UINT tensor_sizes[4] = {dims.b, dims.c, dims.h, dims.w};
  ::dml::TensorDesc::Dimensions dimensions(std::begin(tensor_sizes), std::end(tensor_sizes));
  DML_TENSOR_DATA_TYPE data_type = def.object_def.data_type == DataType::FLOAT32
                                       ? DML_TENSOR_DATA_TYPE_FLOAT32
                                       : DML_TENSOR_DATA_TYPE_FLOAT16;
  ::dml::TensorDesc desc = ::dml::TensorDesc(data_type, dimensions);

  UINT64 tensor_buffer_size = desc.totalTensorSizeInBytes;
  return CreateResource(device, external, access_type, data_type,
                        tensor_buffer_size,
                        resource, name);
}

// Does one-step conversion between internal and external objects.
// It may also allocate external objects if requested.
class DefaultTensorTie : public TensorTie {
 public:
  DefaultTensorTie(const TensorTieDef& def, TensorObject internal_obj,
                   ObjectManager* objects)
      : TensorTie(def), objects_(objects), internal_obj_(internal_obj) {}

  static bool IsSupported(
      const TensorTieDef& def,
      const TensorObjectConverterBuilder& converter_builder) {
    auto object_type = def.external_def.object_def.object_type;
    return (object_type == ObjectType::DIRECTML_RESOURCE ||
            object_type == ObjectType::DIRECTML_TEXTURE ||
            object_type == ObjectType::CPU_MEMORY) &&
           converter_builder.IsSupported(def.internal_def, def.external_def) &&
           converter_builder.IsSupported(def.external_def, def.internal_def);
  }

  static absl::Status New(const TensorTieDef& def,
                          TensorObjectConverterBuilder* converter_builder,
                          ObjectManager* objects,
                          Environment* env, std::unique_ptr<TensorTie>* tie) {
    auto tie_impl =
        absl::make_unique<DefaultTensorTie>(def, TensorObject{}, objects);
    RETURN_IF_ERROR(tie_impl->Init(converter_builder, env));
    *tie = std::move(tie_impl);
    return absl::OkStatus();
  }

  static absl::Status New(const TensorTieDef& def,
                          TensorObjectConverterBuilder* converter_builder,
                          TensorObject internal_object, Environment* env,
                          std::unique_ptr<TensorTie>* tie) {
    if (!IsValid(def.internal_def, internal_object)) {
      return absl::InternalError("Internal object does not match definition.");
    }

    auto tie_impl =
        absl::make_unique<DefaultTensorTie>(def, internal_object, nullptr);
    RETURN_IF_ERROR(tie_impl->Init(converter_builder, env));
    *tie = std::move(tie_impl);
    return absl::OkStatus();
  }

  absl::Status CopyToExternalObject() final {
    if (!converter_to_) {
      return absl::OkStatus();
    }
    return converter_to_->Convert(internal_obj_, GetExternalObject());
  }

  absl::Status CopyFromExternalObject() final {
    if (!converter_from_) {
      return absl::OkStatus();
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
  bool IsSameDef() const {
    const auto& external_def = def().external_def.object_def;
    const auto& internal_def = def().internal_def.object_def;
    return (external_def.object_type == internal_def.object_type &&
            external_def.data_type == internal_def.data_type &&
            external_def.data_layout == internal_def.data_layout)/* ||
           // Check for equivalent layouts that have the same size.
           (external_def.object_type == internal_def.object_type &&
            external_def.data_type == internal_def.data_type &&
            external_def.data_layout == DataLayout::BHWC &&
            internal_def.data_layout == DataLayout::DHWC4 &&
            def().external_def.dimensions.c == 4)*/;
  }

  absl::Status Init(TensorObjectConverterBuilder* converter_builder,
                    Environment* env) {
    // First check is an object is user provided.
    const auto& external_def = def().external_def.object_def;

    const bool is_same_def = IsSameDef();

    if (!is_same_def) {
      RETURN_IF_ERROR(converter_builder->MakeConverter(
          def().external_def, def().internal_def, &converter_from_));
      RETURN_IF_ERROR(converter_builder->MakeConverter(
          def().internal_def, def().external_def, &converter_to_));
    }

    if (external_def.user_provided) {
      if (is_same_def) {
        return absl::OkStatus();
      }
      // Object is provided by a user, but runtime expects different object
      // type. Therefore, we have to allocate internal object and convert.
      return MaybeAllocateInternalObject(env);
    } else {
      RETURN_IF_ERROR(MaybeAllocateInternalObject(env));

      if (is_same_def) {
        // Object is NOT provided by a user, but it matches definition expected
        // by runtime. Conversion is not needed.
        external_obj_ = internal_obj_;
        return absl::OkStatus();
      }

      // Object is NOT provided by a user.
      return MaybeAllocateExternalObject(env);
    }
    return absl::OkStatus();
  }

  absl::Status MaybeAllocateInternalObject(Environment* env) {
    const TensorObjectDef& d = def().internal_def;
    if (d.object_def.user_provided) {
      return absl::OkStatus();
    }
    switch (d.object_def.object_type) {
      case gpu::ObjectType::DIRECTML_RESOURCE: {
        D3DResource resource;
        RETURN_IF_ERROR(MaybeAllocateD3D12Resource(
            env->device(), d, false, def().access_type, &resource,
            def().access_type == AccessType::READ ? L"InternalReadTensor"
                                                  : L"InternalWriteTensor"));
        internal_obj_ = DirectMlResource{resource.Get(), resource.data_type(),
                                         resource.bytes_size()};
        RETURN_IF_ERROR(
            objects_->RegisterResource(def().id, std::move(resource)));
        break;
      }
      // TODO(akulik): support textures as internal object when compiler permits
      default:
        return absl::InternalError("Unexpected object type");
    }
    return absl::OkStatus();
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
      case ObjectType::DIRECTML_RESOURCE: {
        RETURN_IF_ERROR(MaybeAllocateD3D12Resource(
            env->device(), d, true, def().access_type, &external_resource_,
            def().access_type == AccessType::READ ? L"ExternalReadTensor"
                                                  : L"ExternalWriteTensor"));
        external_obj_ = DirectMlResource{external_resource_.Get(),
                                         external_resource_.data_type(),
                                         external_resource_.bytes_size()};
        D3DResource bbb;
        RETURN_IF_ERROR(
            WrapResource(DirectMlResource{external_resource_.Get(),
                                          external_resource_.data_type(),
                                          external_resource_.bytes_size()},
                         &bbb));
        break;
      }
      default:
        return absl::InternalError("Unexpected object type");
    }
    return absl::OkStatus();
  }

  ObjectManager* objects_;

  // hold references to objects.
  TensorObject internal_obj_;
  TensorObject external_obj_;

  // Hold actual objects.
  D3DResource external_resource_;
  std::vector<uint8_t> cpu_memory_;

  std::unique_ptr<TensorObjectConverter> converter_to_;
  std::unique_ptr<TensorObjectConverter> converter_from_;
};

// Copies data to intermediate DirectML resource and then does two step conversion.
// It drives the following cases were one-step conversion is not supported:
//   - CPU BHWC -> DML resource BHWC -> DML resource DHWC4.
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
                          ObjectManager* objects, Environment* env,
                          std::unique_ptr<TensorTie>* tie) {
    auto tie_impl = absl::make_unique<TwoStepTensorTie>(def);
    RETURN_IF_ERROR(tie_impl->Init(converter_builder, objects, env));
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
    outer_def.id = def.id;
    outer_def.access_type = def.access_type;
    outer_def.external_def = def.external_def;
    outer_def.internal_def = def.external_def;
    outer_def.internal_def.object_def.object_type = ObjectType::DIRECTML_RESOURCE;
    // Will not allocate new DirectML Resource
    outer_def.internal_def.object_def.user_provided = true;

    TensorTieDef inner_def;
    inner_def.id = def.id;
    inner_def.access_type = def.access_type;
    inner_def.external_def = outer_def.internal_def;
    // Should not allocate external object.
    inner_def.external_def.object_def.user_provided = false;
#if 1
    inner_def.internal_def = def.internal_def;
#else
    // Reflects what is actually supported by compiler.
    inner_def.internal_def.dimensions = inner_def.external_def.dimensions;
    inner_def.internal_def.object_def.data_type =
        DataType::FLOAT16; // DataType::FLOAT32;
    inner_def.internal_def.object_def.data_layout = DataLayout::DHWC4;
    inner_def.internal_def.object_def.object_type = ObjectType::DIRECTML_RESOURCE;
#endif
    // It may allocate another internal object and should register it to
    // ObjectManager.
    inner_def.internal_def.object_def.user_provided = false;
    return std::make_pair(outer_def, inner_def);
  }

  absl::Status Init(TensorObjectConverterBuilder* converter_builder,
                    ObjectManager* objects, Environment* env) {
    auto defs = MakeOuterInnerDefs(def());
    RETURN_IF_ERROR(DefaultTensorTie::New(defs.second, converter_builder,
                                          objects, env, &inner_tie_));
    return DefaultTensorTie::New(defs.first, converter_builder,
                                 inner_tie_->GetExternalObject(), env, &outer_tie_);
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
  
  absl::Status NewTensorTie(const TensorTieDef& def, ObjectManager* objects,
                            std::unique_ptr<TensorTie>* tie) {
    auto converter = converter_builder_.get();
    if (DefaultTensorTie::IsSupported(def, *converter)) {
      return DefaultTensorTie::New(def, converter, objects, & env_, tie);
    }
    if (TwoStepTensorTie::IsSupported(def, *converter)) {
      return TwoStepTensorTie::New(def, converter, objects, & env_, tie);
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
                      std::unique_ptr<Runtime> runtime,
                      std::unique_ptr<ObjectManager> objects)
      : environment_(environment), runtime_(std::move(runtime)),
        objects_(std::move(objects)) {}

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

//#ifdef _DEBUG
#if 1
    #define DUMP_TIME
#endif  // _DEBUG

  absl::Status Run() override {
#ifdef DUMP_TIME
    auto start = std::chrono::system_clock::now();
#endif // DUMP_TIME
    for (auto& obj : inputs_) {
      RETURN_IF_ERROR(obj->CopyFromExternalObject());
    }
#ifdef DUMP_TIME
    auto end = std::chrono::system_clock::now();
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CopyFrom : " << msec << " ms \n";
    start = std::chrono::system_clock::now();
#endif // DUMP_TIME
    RETURN_IF_ERROR(runtime_->Execute());
#ifdef DUMP_TIME
    end = std::chrono::system_clock::now();
    msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Execute : " << msec << " ms \n";
    start = std::chrono::system_clock::now();
#endif // DUMP_TIME
    for (auto& obj : outputs_) {
      RETURN_IF_ERROR(obj->CopyToExternalObject());
    }
#ifdef DUMP_TIME
    end = std::chrono::system_clock::now();
    msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CopyTo : " << msec << " ms \n";
    start = std::chrono::system_clock::now();
#endif // DUMP_TIME
    return absl::OkStatus();
  }

 private:
  absl::Status LinkTensors(
      const std::vector<TensorTieDef>& defs, TensorTieFactory* factory,
      std::vector<std::unique_ptr<TensorTie>>* objects) {
    objects->reserve(defs.size());
    for (auto& def : defs) {
      std::unique_ptr<TensorTie> object;
      RETURN_IF_ERROR(factory->NewTensorTie(def, objects_.get(), &object));
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

  Environment* environment_;
  std::unique_ptr<Runtime> runtime_;
  std::unique_ptr<ObjectManager> objects_;
  std::vector<std::unique_ptr<TensorTie>> inputs_;
  std::vector<std::unique_ptr<TensorTie>> outputs_;
};

class InferenceBuilderImpl : public InferenceBuilder {
 public:
  explicit InferenceBuilderImpl(const InferenceOptions& options,
                                Environment* environment, GraphFloat32 graph)
    : options_(options),
      environment_(environment),
      graph_(std::move(graph)),
      allow_precision_loss_(false) {}

  absl::Status Initialize(const InferenceEnvironmentOptions& env_options) {
    allow_precision_loss_ =
        environment_->device()->is_fp16_supported &&
        GetPosition(options_, InferencePriority::MAX_PRECISION) > 1;

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
//    def.access_type = AccessType::WRITE;
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
//    def.access_type = AccessType::READ;
    outputs_[index] = def;
    return absl::OkStatus();
  }

  absl::Status Build(std::unique_ptr<InferenceRunner>* runner) override {
#ifdef DUMP_TIME
    auto start = std::chrono::system_clock::now();
#endif  // DUMP_TIME
    auto external_objects = absl::make_unique<ObjectManager>();
    auto runtime = absl::make_unique<Runtime>(
        environment_->device(), external_objects.get(), allow_precision_loss_);
    Runtime* runtime_ptr = runtime.get();
    auto runner_impl = absl::make_unique<InferenceRunnerImpl>(
        environment_, std::move(runtime), std::move(external_objects));
    RETURN_IF_ERROR(
        runner_impl->Initialize(inputs_, outputs_, tie_factory_.get()));

    RETURN_IF_ERROR(runtime_ptr->Compile(graph_));

    *runner = std::move(runner_impl);
#ifdef DUMP_TIME
    auto end = std::chrono::system_clock::now();
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Build : " << msec << " ms \n";
#endif // DUMP_TIME
    return absl::OkStatus();
  }

 private:
  // Links internal tensors with external user-facing objects.
  std::vector<TensorTieDef> LinkTensors(const std::vector<Value*>& values) {
    std::vector<TensorTieDef> links;
    links.reserve(values.size());
    for (const auto& value : values) {
      TensorObjectDef external_def;
      // So far the compiler always forces inputs and outputs to be in the fixed
      // format.
      const auto& shape = value->tensor.shape;
      external_def.dimensions = Dimensions(shape.b, shape.h, shape.w, shape.c);
      external_def.object_def.data_type =
          allow_precision_loss_ ? DataType::FLOAT16 : DataType::FLOAT32;
      external_def.object_def.data_layout = DataLayout::DHWC4;
      external_def.object_def.object_type = gpu::ObjectType::DIRECTML_RESOURCE;

      // Internal object is not expected to be provided by user because: if
      // external and internal objects have same defs, the external object is
      // propagated and just used as an internal one; otherwise, if they have
      // different defs, internal object will be created, because it is not
      // provided by user.
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

  const InferenceOptions options_;
  Environment* environment_;
  GraphFloat32 graph_;
  bool allow_precision_loss_;

  std::vector<TensorTieDef> inputs_;
  std::vector<TensorTieDef> outputs_;
  std::unique_ptr<TensorTieFactory> tie_factory_;
};

class InferenceEnvironmentImpl : public InferenceEnvironment {
 public:
  explicit InferenceEnvironmentImpl(const InferenceEnvironmentOptions& options)
      : options_(options) {}

  absl::Status Init() {
    properties_.is_directml_available = true;

    DMLDevice* device = options_.dml_device;
    if (!device) {
      if (options_.d3d_device) {
        device_.reset(new DMLDevice(options_.d3d_device));
      } else {
        device_.reset(new DMLDevice());
        RETURN_IF_ERROR(CreateDefaultGPUDevice(device_.get()));
      }
      device_->Init();
      device = device_.get();
    }

    environment_ = Environment(device);
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
        resolved_options, & environment_, std::move(model));
    RETURN_IF_ERROR(
        builder_impl->Initialize(options_));
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
  std::unique_ptr<DMLDevice> device_;
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