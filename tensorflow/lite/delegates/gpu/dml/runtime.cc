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

#include "tensorflow/lite/delegates/gpu/dml/runtime.h"

using Microsoft::WRL::ComPtr;

namespace tflite {
namespace gpu {
namespace dml {

Runtime::Runtime(DMLDevice* device_, const ObjectManager* external_objects,
                 bool allow_precision_loss)
  : device(device_),
    external_objects_(external_objects),
    allow_precision_loss_(allow_precision_loss) {}

absl::Status Runtime::Compile(const GraphFloat32& graph) {
  // Compile operator
  RETURN_IF_ERROR(CreateOperator(graph));

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
        D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&initialize_temporary_buffer)));
  } else if (initialize_binding_properties.TemporaryResourceSize > 0) {
    initialize_temporary_buffer = temporary_buffer;
  }

#if DML_MANAGED_WEIGHTS
  std::vector<DML_BUFFER_BINDING> buffer_bindings;
  const uint32_t num_input = input_resources.size();
  buffer_bindings.resize(num_input);
  for (uint32_t i = 0; i < num_input; i++) {
    auto resource = input_resources[i];
    if (external_objects_->IsRegistered(resource)) {
      buffer_bindings[i] = {nullptr, 0, 0};
    } else {
      buffer_bindings[i] = {resource->Get(), 0, resource->bytes_size()};
    }
  }
  DML_BUFFER_ARRAY_BINDING input_binding = {num_input,
                                            buffer_bindings.data()};
  binding_table->BindInputs(
      1, &DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER_ARRAY, &input_binding});
#else // DML_MANAGED_WEIGHTS
  binding_table->BindInputs(0, nullptr);
#endif // DML_MANAGED_WEIGHTS

  if (initialize_temporary_buffer) {
    DML_BUFFER_BINDING buffer_binding{
        initialize_temporary_buffer.Get(), 0,
        initialize_binding_properties.TemporaryResourceSize};        
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
    binding_table->BindOutputs(
        1, &DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &buffer_binding});
  }

  // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
  DML_CHECK_SUCCEEDED(device->dml_device->CreateCommandRecorder(IID_PPV_ARGS(&command_recorder)));

  // Record execution of the operator initializer.
  command_recorder->RecordDispatch(device->command_list.Get(),
                                   operator_initializer.Get(),
                                   binding_table.Get());

  device->CloseExecuteResetWait();

#if DML_MANAGED_WEIGHTS
  for (uint32_t i = 0; i < num_input; i++) {
    if (buffer_bindings[i].Buffer) {
      input_resources[i] = nullptr;
    }
  }
  const_objects_.RemoveAllResource();
#endif // DML_MANAGED_WEIGHTS

  return absl::OkStatus();
}

D3DResource* Runtime::AllocateConstObject(const uint8_t* data,
                                          DML_TENSOR_DATA_TYPE data_type,
                                          uint32_t size) {
  D3DResource d3d_resource;
  CreateResource(device, AccessType::WRITE, data_type, size, &d3d_resource);

  const_objects_.RegisterResource(next_const_id_, d3d_resource);
  D3DResource* resource = const_objects_.FindResource(next_const_id_);
  next_const_id_++;

  resource->Write(device, absl::MakeConstSpan(data, size));

  return resource;
}

absl::Status Runtime::Execute() {
  DML_BINDING_PROPERTIES execute_binding_properties =
      compiled_operator->GetBindingProperties();

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
    DML_BUFFER_BINDING buffer_binding{
        temporary_buffer.Get(), 0,
        execute_binding_properties.TemporaryResourceSize};
    binding_table->BindTemporaryResource(
        &DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &buffer_binding});
  }

  if (persistent_buffer) {
    DML_BUFFER_BINDING buffer_binding{persistent_buffer.Get(), 0,
                                      persistent_buffer->GetDesc().Width};
    binding_table->BindPersistentResource(
        &DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &buffer_binding});
  }

  std::vector<DML_BUFFER_BINDING> buffer_bindings;
  std::vector<DML_BINDING_DESC> input_bindings;
  const uint32_t num_input = input_resources.size();
  buffer_bindings.resize(num_input);
  input_bindings.resize(num_input);
  for (uint32_t i = 0; i < num_input; i++) {
#if DML_MANAGED_WEIGHTS
    auto resource = input_resources[i];
    if (resource) {
      buffer_bindings[i] = {resource->Get(), 0, resource->bytes_size()};
      input_bindings[i] = {DML_BINDING_TYPE_BUFFER, &buffer_bindings[i]};
    } else {
      input_bindings[i] = {DML_BINDING_TYPE_NONE, nullptr};
    }
#else  // DML_MANAGED_WEIGHTS
    auto resource = input_resources[i];
    buffer_bindings[i] = {resource->Get(), 0, resource->bytes_size()};
    input_bindings[i] = {DML_BINDING_TYPE_BUFFER, &buffer_bindings[i]};
#endif  // DML_MANAGED_WEIGHTS
  }
  binding_table->BindInputs(num_input, input_bindings.data());

  DML_BUFFER_BINDING output_buffer_binding = {output_resources[0]->Get(), 0,
                                              output_resources[0]->bytes_size()};
  binding_table->BindOutputs(
      1, &DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &output_buffer_binding});

  // Record execution of the compiled operator.
  command_recorder->RecordDispatch(device->command_list.Get(),
                                   compiled_operator.Get(),
                                   binding_table.Get());

  device->CloseExecuteResetWait();

  return absl::OkStatus();
}

}  // namespace dml
}  // namespace gpu
}  // namespace tflite
