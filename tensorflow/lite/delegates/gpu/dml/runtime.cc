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

::dml::Expression& FindInputExpression(
    ::dml::Scope& scope, std::map<ValueId, ::dml::Expression>& expressions,
    const Value* value) {

  auto itr = expressions.find(value->id);
  if (itr != expressions.end()) {
    return itr->second;
  }

  const auto& shape = value->tensor.shape;
  UINT tensor_sizes[4] = {shape.b, shape.w, shape.h, shape.c};
  ::dml::TensorDesc::Dimensions dimensions(std::begin(tensor_sizes),
                                           std::end(tensor_sizes));
  ::dml::TensorDesc desc = {DML_TENSOR_DATA_TYPE_FLOAT32, dimensions};
  ::dml::Expression input = ::dml::InputTensor(scope, 0, desc);

  expressions[value->id] = input;
  return expressions[value->id];
}

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
  ::dml::Scope scope(device->dml_device.Get());
  std::map<ValueId, ::dml::Expression> expressions;
  ValueId last_output = 0;

#if 1
  auto inputs = graph.FindInputs(0);
  auto input = FindInputExpression(scope, expressions, inputs[0]);
  auto output = ::dml::Identity(input);

  DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
  compiled_operator = scope.Compile(execution_flags, {output});

  IDMLCompiledOperator* compiled_operators[] = {compiled_operator.Get()};
  DML_CHECK_SUCCEEDED(device->dml_device->CreateOperatorInitializer(
      ARRAYSIZE(compiled_operators), compiled_operators,
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

  // Create the temporary and persistent resources that are necessary for executing an operator.
  temporary_resource_size =
      std::max(initialize_binding_properties.TemporaryResourceSize, execute_binding_properties.TemporaryResourceSize);
  persistent_resource_size = execute_binding_properties.PersistentResourceSize;

  // Bind and initialize the operator on the GPU.
  if (temporary_resource_size != 0) {
    DML_CHECK_SUCCEEDED(device->d3d_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(temporary_resource_size),
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&temporary_buffer)));

    DML_BUFFER_BINDING buffer_binding{temporary_buffer.Get(), 0, temporary_resource_size};
    DML_BINDING_DESC binding_desc{DML_BINDING_TYPE_BUFFER, &buffer_binding};
    binding_table->BindTemporaryResource(&binding_desc);
  }

  if (persistent_resource_size != 0) {
    DML_CHECK_SUCCEEDED(device->d3d_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(persistent_resource_size),
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&persistent_buffer)));

    // The persistent resource should be bound as the output to the
    // IDMLOperatorInitializer.
    DML_BUFFER_BINDING buffer_binding{persistent_buffer.Get(), 0, persistent_resource_size};
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

  for (auto value : graph.values()) {
    if (graph.IsGraphInput(value->id)) {
      input_ids.push_back(value->id);
    } else if (graph.IsGraphOutput(value->id)) {
      output_ids.push_back(value->id);
    }
  }

#else
  std::vector<Node*> graph_nodes = graph.nodes();
  for (int i = 0; i < graph_nodes.size(); ++i) {
    const Node& node = *graph_nodes[i];
    auto inputs = graph.FindInputs(node.id);
    auto outputs = graph.FindOutputs(node.id);

    auto op_type = OperationTypeFromString(node.operation.type);
    switch (op_type) {
      case OperationType::CONVOLUTION_2D: {
        auto attr = absl::any_cast<Convolution2DAttributes>(node.operation.attributes);

        auto input = FindInputExpression(scope, expressions, inputs[0]);
//        auto output = ::dml::Convolution(input,

      } break;
      case OperationType::PAD: {
        auto attr = absl::any_cast<PadAttributes>(node.operation.attributes);
        DML_PADDING_MODE padding_mode =
            attr.type == PaddingContentType::ZEROS
                ? DML_PADDING_MODE_CONSTANT
                : attr.type == PaddingContentType::REFLECT
                      ? DML_PADDING_MODE_REFLECTION
                      : DML_PADDING_MODE_EDGE;
        uint32_t start_padding[] = {0, 0, 0, 0};
        uint32_t end_padding[] = {attr.prepended.b, attr.prepended.w,
                                  attr.prepended.h, attr.prepended.c};

        auto input = FindInputExpression(scope, expressions, inputs[0]);
        auto output = ::dml::Padding(input, padding_mode, 0.0f,
                                     ::dml::Span<uint32_t>(start_padding),
                                     ::dml::Span<uint32_t>(end_padding));

        expressions[outputs[0]->id] = output;
        last_output = outputs[0]->id;
      } break;
      case OperationType::SLICE: {
        auto attr = absl::any_cast<SliceAttributes>(node.operation.attributes);
        const auto& shape = inputs[0]->tensor.shape;
        int4 offset = GetOffset(attr, shape.w, shape.h, shape.c, shape.b);

        uint32_t offsets[] = {offset.w, offset.x, offset.y, offset.z};
        uint32_t sizes[] = {shape.b, shape.w, shape.h, shape.c};
        uint32_t strides[] = {attr.strides.b, attr.strides.w,
                                         attr.strides.h, attr.strides.c};

        auto input = FindInputExpression(scope, expressions, inputs[0]);
        auto output = ::dml::Slice(input,
          ::dml::Span<uint32_t>(offsets),
          ::dml::Span<uint32_t>(sizes),
          ::dml::Span<uint32_t>(strides));

        expressions[outputs[0]->id] = output;
        last_output = outputs[0]->id;
      } break;
      default:
        return absl::UnimplementedError(absl::StrCat(
            "No support of ", node.operation.type, " with this parameters"));
    }
  }
#endif

  return absl::OkStatus();
}

absl::Status Runtime::Execute() {
  // Bind and execute the operator on the GPU.
  ID3D12DescriptorHeap* descriptor_heaps[] = {descriptor_heap.Get()};
  device->command_list->SetDescriptorHeaps(ARRAYSIZE(descriptor_heaps), descriptor_heaps);

  // Reset the binding table to bind for the operator we want to execute 
  DML_BINDING_TABLE_DESC binding_table_desc{};
  binding_table_desc.Dispatchable = compiled_operator.Get();
  binding_table_desc.CPUDescriptorHandle = descriptor_heap->GetCPUDescriptorHandleForHeapStart();
  binding_table_desc.GPUDescriptorHandle = descriptor_heap->GetGPUDescriptorHandleForHeapStart();
  binding_table_desc.SizeInDescriptors = descriptor_count;
  DML_CHECK_SUCCEEDED(binding_table->Reset(&binding_table_desc));

  if (temporary_resource_size != 0) {
    DML_BUFFER_BINDING buffer_binding{temporary_buffer.Get(), 0, temporary_resource_size};
    DML_BINDING_DESC binding_desc{DML_BINDING_TYPE_BUFFER, &buffer_binding};
    binding_table->BindTemporaryResource(&binding_desc);
  }

  if (persistent_resource_size != 0) {
    DML_BUFFER_BINDING buffer_binding{persistent_buffer.Get(), 0, persistent_resource_size};
    DML_BINDING_DESC binding_desc{DML_BINDING_TYPE_BUFFER, &buffer_binding};
    binding_table->BindPersistentResource(&binding_desc);
  }

  for (auto id : input_ids) {
    auto resource = external_objects_->FindResource(id);
    DML_BUFFER_BINDING input_buffer_binding{resource->Get(), 0,
                                            resource->bytes_size()};
    DML_BINDING_DESC input_binding_desc{DML_BINDING_TYPE_BUFFER,
                                        &input_buffer_binding};
    binding_table->BindInputs(1, &input_binding_desc);
  }

  for (auto id : output_ids) {
    auto resource = external_objects_->FindResource(id);
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
