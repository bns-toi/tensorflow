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

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/dml/runtime.h"

using Microsoft::WRL::ComPtr;

namespace tflite {
namespace gpu {
namespace dml {

namespace {

bool Is4Aligned(const SliceAttributes& attr) {
  return attr.strides.c == 1 && attr.starts.c % 4 == 0;
}

int4 GetOffset(const SliceAttributes& attr, int src_width, int src_height,
               int src_channels, int src_batch) {
  int4 offset;
  if (attr.strides.w > 0) {
    offset.x = attr.starts.w;
  } else {
    if (attr.ends.w > 0) {
      offset.x = attr.ends.w;
    } else {
      offset.x = src_width + attr.ends.w;
    }
  }
  if (attr.strides.h > 0) {
    offset.y = attr.starts.h;
  } else {
    if (attr.ends.h > 0) {
      offset.y = attr.ends.h;
    } else {
      offset.y = src_height + attr.ends.h;
    }
  }
  if (attr.strides.c > 0) {
    offset.z = attr.starts.c;
  } else {
    if (attr.ends.c > 0) {
      offset.z = attr.ends.c;
    } else {
      offset.z = src_channels + attr.ends.c;
    }
  }
  if (Is4Aligned(attr)) {
    offset.z /= 4;
  }
  if (attr.strides.b > 0) {
    offset.w = attr.starts.b;
  } else {
    if (attr.ends.b > 0) {
      offset.w = attr.ends.b;
    } else {
      offset.w = src_batch + attr.ends.b;
    }
  }
  return offset;
}

}  // namespace

Runtime::Runtime(DMLDevice* device_, const ObjectManager* external_objects)
    : device(device_), external_objects_(external_objects) {}

absl::Status Runtime::Compile(const GraphFloat32& graph) {

  DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;
//  flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
  ::dml::TensorPolicy policy = ::dml::TensorPolicy::Default();
//  ::dml::TensorPolicy policy = ::dml::TensorPolicy::InterleavedChannel();
  ::dml::Scope scope(device->dml_device.Get(), policy);

  std::map<ValueId, ::dml::Expression> expressions;
  ValueId last_output = 0;

  for (auto value : graph.values()) {
    if (graph.IsGraphInput(value->id)) {
      last_output =
          CreateInputTensorExpression(scope, flags, policy, value, expressions);
    } else if (graph.IsGraphOutput(value->id)) {
      output_resources.push_back(
          external_objects_->FindResource(value->id));
    }
  }

  std::vector<Node*> graph_nodes = graph.nodes();
  for (int i = 0; i < graph_nodes.size(); ++i) {
    const Node& node = *graph_nodes[i];
    auto inputs = graph.FindInputs(node.id);
    auto outputs = graph.FindOutputs(node.id);

    auto op_type = OperationTypeFromString(node.operation.type);
    switch (op_type) {
      case OperationType::CONCAT: {
        last_output = CreateConcatExpression(graph, node, expressions);
      } break;
      case OperationType::CONVOLUTION_2D: {
        last_output = CreateConvolution2DExpression(scope, flags, policy, graph,
                                                    node, expressions);
      } break;
      case OperationType::PAD: {
        last_output = CreatePadExpression(graph, node, expressions);
      } break;
      case OperationType::RELU: {
        last_output = CreateReLUExpression(graph, node, expressions);
      } break;
      case OperationType::SLICE: {
        last_output = CreateSliceExpression(graph, node, expressions);
      } break;
      case OperationType::MAXIMUM: {
        last_output = CreateMaximumExpression(graph, node, expressions);
      } break;
      case OperationType::MINIMUM: {
        last_output = CreateMinimumExpression(graph, node, expressions);
      } break;
      default:
        return absl::UnimplementedError(absl::StrCat(
            "No support of ", node.operation.type, " with this parameters"));
    }
  }

  auto output = expressions[last_output];

//  DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
  DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_NONE;
  compiled_operator = scope.Compile(execution_flags, {output});

  IDMLCompiledOperator* compiled_operators[] = {compiled_operator.Get()};
  DML_CHECK_SUCCEEDED(device->dml_device->CreateOperatorInitializer(
      1, compiled_operator.GetAddressOf(),
      IID_PPV_ARGS(&operator_initializer)));

  DML_BINDING_PROPERTIES initialize_binding_properties = operator_initializer->GetBindingProperties();
  DML_BINDING_PROPERTIES execute_binding_properties = compiled_operator->GetBindingProperties();
  descriptor_count = std::max(initialize_binding_properties.RequiredDescriptorCount, execute_binding_properties.RequiredDescriptorCount);

  // Create descriptor heaps.
  D3D12_DESCRIPTOR_HEAP_DESC descriptor_heap_desc{};
  descriptor_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  descriptor_heap_desc.NumDescriptors = descriptor_count;
  descriptor_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  DML_CHECK_SUCCEEDED(device->d3d_device->CreateDescriptorHeap(
      &descriptor_heap_desc, IID_PPV_ARGS(&descriptor_heap)));

  // Set the descriptor heap(s).
  ID3D12DescriptorHeap* descriptor_heaps[] = {descriptor_heap.Get()};
  device->command_list->SetDescriptorHeaps(ARRAYSIZE(descriptor_heaps), descriptor_heaps);

  // Create a binding table over the descriptor heap we just created.
  DML_BINDING_TABLE_DESC binding_table_desc{};
  binding_table_desc.Dispatchable = operator_initializer.Get();
  binding_table_desc.CPUDescriptorHandle = descriptor_heap->GetCPUDescriptorHandleForHeapStart();
  binding_table_desc.GPUDescriptorHandle = descriptor_heap->GetGPUDescriptorHandleForHeapStart();
  binding_table_desc.SizeInDescriptors = descriptor_count;

  DML_CHECK_SUCCEEDED(device->dml_device->CreateBindingTable(
      &binding_table_desc, IID_PPV_ARGS(&binding_table)));

  // Initialize and bind the operator on the GPU.
  if (execute_binding_properties.TemporaryResourceSize > 0) {
    DML_CHECK_SUCCEEDED(device->d3d_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(
            execute_binding_properties.TemporaryResourceSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&temporary_buffer)));
  }
  Microsoft::WRL::ComPtr<ID3D12Resource> initialize_temporary_buffer;
  if (initialize_binding_properties.TemporaryResourceSize >
      execute_binding_properties.TemporaryResourceSize) {
    DML_CHECK_SUCCEEDED(device->d3d_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(
            initialize_binding_properties.TemporaryResourceSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&temporary_buffer)));
  } else if (initialize_binding_properties.TemporaryResourceSize > 0) {
    initialize_temporary_buffer = temporary_buffer;
  }

  if (initialize_temporary_buffer) {
    DML_BUFFER_BINDING buffer_binding{
        temporary_buffer.Get(), 0,
        execute_binding_properties.TemporaryResourceSize};
    binding_table->BindTemporaryResource(
        &DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &buffer_binding});
  }

  if (execute_binding_properties.PersistentResourceSize > 0) {
    DML_CHECK_SUCCEEDED(device->d3d_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(
            execute_binding_properties.PersistentResourceSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&persistent_buffer)));
  }

  if (persistent_buffer) {
    DML_BUFFER_BINDING buffer_binding{
        persistent_buffer.Get(), 0,
        execute_binding_properties.PersistentResourceSize};
    DML_BINDING_DESC binding_desc{DML_BINDING_TYPE_BUFFER, &buffer_binding};
    binding_table->BindOutputs(1, &binding_desc);
  }

  // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
  DML_CHECK_SUCCEEDED(device->dml_device->CreateCommandRecorder(IID_PPV_ARGS(&command_recorder)));

  // Record execution of the operator initializer.
  command_recorder->RecordDispatch(device->command_list,
                                   operator_initializer.Get(),
                                   binding_table.Get());

  device->CloseExecuteResetWait();

  return absl::OkStatus();
}

D3DResource* Runtime::AllocateConstObject(const uint8_t* data, uint32_t size) {
  D3DResource d3d_resource;
  CreateResource(device, AccessType::WRITE, size, &d3d_resource);

  const_objects_.RegisterResource(next_const_id_, d3d_resource);
  D3DResource* resource = const_objects_.FindResource(next_const_id_);
  next_const_id_++;

  resource->Write(device, absl::MakeConstSpan(data, size));

  return resource;
}

ValueId Runtime::CreateInputTensorExpression(
    ::dml::Scope& scope, DML_TENSOR_FLAGS flags,
    const ::dml::TensorPolicy& policy, const Value* value,
    std::map<ValueId, ::dml::Expression>& expressions) {
  uint32_t index = input_resources.size();
  input_resources.push_back(external_objects_->FindResource(value->id));

  const auto& shape = value->tensor.shape;
  UINT tensor_sizes[4] = {shape.b, shape.c, shape.h, shape.w};
  ::dml::TensorDesc::Dimensions dimensions(std::begin(tensor_sizes),
                                           std::end(tensor_sizes));
  auto output =
      ::dml::InputTensor(scope, index,
                         ::dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, flags,
                                           dimensions, policy));

  expressions[value->id] = output;
  return value->id;
}

::dml::Expression Runtime::CreateConstInputTensorExpression(
    ::dml::Scope& scope, DML_TENSOR_FLAGS flags,
    const ::dml::TensorPolicy& policy, const uint8_t* data, uint32_t size,
    ::dml::TensorDesc::Dimensions& dimensions)
{
  uint32_t index = input_resources.size();
  D3DResource* resource = AllocateConstObject(data, size);
  input_resources.push_back(resource);

  return ::dml::InputTensor(scope, index,
                            ::dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32,
                                              flags, dimensions, policy));
}

ValueId Runtime::CreateConcatExpression(
    const GraphFloat32& graph, const Node& node,
    std::map<ValueId, ::dml::Expression>& expressions) {
  auto input_values = graph.FindInputs(node.id);
  auto output_values = graph.FindOutputs(node.id);
  auto attr = absl::any_cast<ConcatAttributes>(node.operation.attributes);

  const ::dml::Expression inputs[2] = {expressions[input_values[0]->id],
                                       expressions[input_values[1]->id]};
      
  auto output = ::dml::Join(::dml::Span<const ::dml::Expression>(inputs),
                            static_cast<uint32_t>(attr.axis));

  expressions[output_values[0]->id] = output;
  return output_values[0]->id;
}

ValueId Runtime::CreateConvolution2DExpression(
    ::dml::Scope& scope, DML_TENSOR_FLAGS flags,
    const ::dml::TensorPolicy& policy, const GraphFloat32& graph,
    const Node& node,
    std::map<ValueId, ::dml::Expression>& expressions) {
  auto inputs = graph.FindInputs(node.id);
  auto outputs = graph.FindOutputs(node.id);
  auto attr = absl::any_cast<Convolution2DAttributes>(node.operation.attributes);

  // Input Expression
  auto input = expressions[inputs[0]->id];

#if 1
  // Weights Expression
  const auto& weights_shape = attr.weights.shape;

  std::vector<float> gpu_data;
  gpu_data.resize(weights_shape.o * weights_shape.h * weights_shape.w *
                   weights_shape.i);
  for (uint32_t o = 0; o < weights_shape.o; o++) {
    for (uint32_t h = 0; h < weights_shape.h; h++) {
      for (uint32_t w = 0; w < weights_shape.w; w++) {
        for (uint32_t i = 0; i < weights_shape.i; i++) {
          uint32_t offset = o * weights_shape.i * weights_shape.h * weights_shape.w;
          uint32_t idx = w + h * weights_shape.w;

          gpu_data[offset + idx + i * weights_shape.h * weights_shape.w] =
              attr.weights.data[offset + idx * weights_shape.i + i];
        }
      }
    }
  }

  UINT weights_tensor_sizes[4] = {weights_shape.o, weights_shape.i,
                                  weights_shape.h, weights_shape.w};
  ::dml::TensorDesc::Dimensions weights_dimensions(
      std::begin(weights_tensor_sizes), std::end(weights_tensor_sizes));
  auto filter = CreateConstInputTensorExpression(
      scope, flags, policy, reinterpret_cast<const uint8_t*>(gpu_data.data()),
      gpu_data.size() * sizeof(float), weights_dimensions);
  //      reinterpret_cast<const uint8_t*>(attr.weights.data.data()),
//      attr.weights.data.size() * sizeof(float), weights_dimensions);

  // Bias Expression
  const auto& bias_shape = attr.bias.shape;
  UINT bias_tensor_sizes[4] = {1, bias_shape.v, 1, 1};
  ::dml::TensorDesc::Dimensions bias_dimensions(std::begin(bias_tensor_sizes),
                                                std::end(bias_tensor_sizes));
  auto bias = CreateConstInputTensorExpression(
      scope, flags, policy,
      reinterpret_cast<const uint8_t*>(attr.bias.data.data()),
      attr.bias.data.size() * sizeof(float), bias_dimensions);

  // Parameters
  const uint32_t strides[2] = {attr.strides.h, attr.strides.w};
  const uint32_t dilations[2] = {attr.dilations.h, attr.dilations.w};
  const uint32_t start_padding[2] = {attr.padding.prepended.h,
                                     attr.padding.prepended.w};
  const uint32_t end_padding[2] = {attr.padding.appended.h,
                                   attr.padding.appended.w};

  // Convolution 2D Expression
  auto output = ::dml::ConvolutionBuilder(input, filter, bias)
                    .Strides(::dml::Span<const uint32_t>{strides})
                    .Dilations(::dml::Span<const uint32_t>{dilations})
                    .StartPadding(::dml::Span<const uint32_t>{start_padding})
                    .EndPadding(::dml::Span<const uint32_t>{end_padding})
                    .Build();
#else
  auto output = ::dml::Identity(input);
#endif

  expressions[outputs[0]->id] = output;
  return outputs[0]->id;
}

ValueId Runtime::CreatePadExpression(
    const GraphFloat32& graph, const Node& node,
    std::map<ValueId, ::dml::Expression>& expressions) {
  auto inputs = graph.FindInputs(node.id);
  auto outputs = graph.FindOutputs(node.id);
  auto attr = absl::any_cast<PadAttributes>(node.operation.attributes);
  DML_PADDING_MODE padding_mode = attr.type == PaddingContentType::ZEROS
                                      ? DML_PADDING_MODE_CONSTANT
                                      : attr.type == PaddingContentType::REFLECT
                                            ? DML_PADDING_MODE_REFLECTION
                                            : DML_PADDING_MODE_EDGE;
  const uint32_t start_padding[4] = {attr.prepended.b, attr.prepended.c,
                                     attr.prepended.h, attr.prepended.w};
  const uint32_t end_padding[4] = {attr.appended.b, attr.appended.c,
                                   attr.appended.h, attr.appended.w};

  auto input = expressions[inputs[0]->id];
#if 1
  auto output = ::dml::Padding(input, padding_mode, 0.0f,
                               ::dml::Span<const uint32_t>{start_padding},
                               ::dml::Span<const uint32_t>{end_padding});
#else
  auto output = ::dml::Identity(input);
#endif

  expressions[outputs[0]->id] = output;
  return outputs[0]->id;
}

ValueId Runtime::CreateReLUExpression(
    const GraphFloat32& graph, const Node& node,
    std::map<ValueId, ::dml::Expression>& expressions) {
  auto inputs = graph.FindInputs(node.id);
  auto outputs = graph.FindOutputs(node.id);
  auto attr = absl::any_cast<ReLUAttributes>(node.operation.attributes);

  auto input = expressions[inputs[0]->id];
#if 1
  auto output = ::dml::ActivationRelu(input);
#else
  auto output = ::dml::Identity(input);
#endif

  expressions[outputs[0]->id] = output;
  return outputs[0]->id;
}

ValueId Runtime::CreateSliceExpression(
    const GraphFloat32& graph, const Node& node,
    std::map<ValueId, ::dml::Expression>& expressions) {
  auto inputs = graph.FindInputs(node.id);
  auto outputs = graph.FindOutputs(node.id);
  auto attr = absl::any_cast<SliceAttributes>(node.operation.attributes);

  const auto& input_shape = inputs[0]->tensor.shape;
  const auto& output_shape = outputs[0]->tensor.shape;
  int4 offset = GetOffset(attr, input_shape.w, input_shape.h, input_shape.c,
                          input_shape.b);

  const uint32_t offsets[4] = {offset.w, offset.z, offset.y, offset.x};
  const uint32_t sizes[4] = {output_shape.b, output_shape.c, output_shape.h,
                             output_shape.w};
  const uint32_t strides[4] = {attr.strides.b, attr.strides.c, attr.strides.h,
                               attr.strides.w};

  auto input = expressions[inputs[0]->id];
  auto output = ::dml::Slice(input, ::dml::Span<const uint32_t>{offsets},
                             ::dml::Span<const uint32_t>{sizes},
                             ::dml::Span<const uint32_t>{strides});

  expressions[outputs[0]->id] = output;
  return outputs[0]->id;
}

ValueId Runtime::CreateClipExpression(
    const GraphFloat32& graph, const Node& node,
    std::map<ValueId, ::dml::Expression>& expressions,
    float min_value, float max_value) {
  auto input_values = graph.FindInputs(node.id);
  auto output_values = graph.FindOutputs(node.id);

  auto input = expressions[input_values[0]->id];
  auto output = ::dml::Clip(input, min_value, max_value);

  expressions[output_values[0]->id] = output;
  return output_values[0]->id;
}

ValueId Runtime::CreateMaximumExpression(
    const GraphFloat32& graph, const Node& node,
    std::map<ValueId, ::dml::Expression>& expressions) {
  auto attr = absl::any_cast<ElementwiseAttributes>(node.operation.attributes);
  const float* scalar = absl::get_if<float>(&attr.param);
  return CreateClipExpression(graph, node, expressions, *scalar, FLT_MAX);
}

ValueId Runtime::CreateMinimumExpression(
    const GraphFloat32& graph, const Node& node,
    std::map<ValueId, ::dml::Expression>& expressions) {
  auto attr = absl::any_cast<ElementwiseAttributes>(node.operation.attributes);
  const float* scalar = absl::get_if<float>(&attr.param);
  return CreateClipExpression(graph, node, expressions, FLT_MIN, *scalar);
}

absl::Status Runtime::Execute() {
  // Bind and execute the operator on the GPU.
  ID3D12DescriptorHeap* descriptor_heaps[] = {descriptor_heap.Get()};
  device->command_list->SetDescriptorHeaps(ARRAYSIZE(descriptor_heaps),
                                           descriptor_heaps);

  // Reset the binding table to bind for the operator we want to execute
  DML_BINDING_TABLE_DESC binding_table_desc{};
  binding_table_desc.Dispatchable = compiled_operator.Get();
  binding_table_desc.CPUDescriptorHandle =
      descriptor_heap->GetCPUDescriptorHandleForHeapStart();
  binding_table_desc.GPUDescriptorHandle =
      descriptor_heap->GetGPUDescriptorHandleForHeapStart();
  binding_table_desc.SizeInDescriptors = descriptor_count;
  DML_CHECK_SUCCEEDED(binding_table->Reset(&binding_table_desc));

  if (temporary_buffer) {
    DML_BUFFER_BINDING buffer_binding{temporary_buffer.Get(), 0,
                                      temporary_buffer->GetDesc().Width};
    DML_BINDING_DESC binding_desc{DML_BINDING_TYPE_BUFFER, &buffer_binding};
    binding_table->BindTemporaryResource(&binding_desc);
  }

  if (persistent_buffer) {
    DML_BUFFER_BINDING buffer_binding{persistent_buffer.Get(), 0,
                                      persistent_buffer->GetDesc().Width};
    DML_BINDING_DESC binding_desc{DML_BINDING_TYPE_BUFFER, &buffer_binding};
    binding_table->BindPersistentResource(&binding_desc);
  }

  std::vector<DML_BUFFER_BINDING> buffer_bindings;
  std::vector<DML_BINDING_DESC> input_bindings;
  buffer_bindings.reserve(input_resources.size());
  input_bindings.reserve(input_resources.size());
  for (auto resource : input_resources) {
    DML_BUFFER_BINDING input_buffer_binding{resource->Get(), 0, resource->bytes_size()};
    buffer_bindings.push_back(input_buffer_binding);

    DML_BINDING_DESC input_binding_desc{DML_BINDING_TYPE_BUFFER, &buffer_bindings[buffer_bindings.size() - 1]};
    input_bindings.push_back(input_binding_desc);
  }
  binding_table->BindInputs(input_bindings.size(), input_bindings.data());

  for (auto resource : output_resources) {
    DML_BUFFER_BINDING output_buffer_binding{resource->Get(), 0,
                                             resource->bytes_size()};
    DML_BINDING_DESC output_binding_desc{DML_BINDING_TYPE_BUFFER,
                                       &output_buffer_binding};
    binding_table->BindOutputs(1, &output_binding_desc);
  }

  // Record execution of the compiled operator.
  command_recorder->RecordDispatch(device->command_list,
                                   compiled_operator.Get(),
                                   binding_table.Get());

  device->CloseExecuteResetWait();

  return absl::OkStatus();
}

}  // namespace dml
}  // namespace gpu
}  // namespace tflite
