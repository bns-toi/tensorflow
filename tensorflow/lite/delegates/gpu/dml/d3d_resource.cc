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

#include "tensorflow/lite/delegates/gpu/dml/d3d_resource.h"

using Microsoft::WRL::ComPtr;

namespace tflite {
namespace gpu {
namespace dml {

D3DResource::~D3DResource() {

}

absl::Status D3DResource::ReadResource(DMLDevice* device, void* data) const {
  ComPtr<ID3D12Resource> readback_buffer;
  DML_CHECK_SUCCEEDED(device->d3d_device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(bytes_size_),
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_buffer)));

  device->command_list->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             resource_ptr, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COPY_SOURCE));

  device->command_list->CopyResource(readback_buffer.Get(), resource_ptr);

  device->CloseExecuteResetWait();

  D3D12_RANGE buffer_range{0, static_cast<SIZE_T>(bytes_size_)};
  FLOAT* buffer_data{};
  DML_CHECK_SUCCEEDED(readback_buffer->Map(
      0, &buffer_range, reinterpret_cast<void**>(&buffer_data)));

  std::memcpy(data, buffer_data, bytes_size_);
  
  D3D12_RANGE empty_range{0, 0};
  readback_buffer->Unmap(0, &empty_range);

  return absl::OkStatus();
}

absl::Status D3DResource::WriteResource(DMLDevice* device, const void* data) {
  ComPtr<ID3D12Resource> upload_buffer;
  DML_CHECK_SUCCEEDED(device->d3d_device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
      D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(bytes_size_),
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload_buffer)));

  D3D12_SUBRESOURCE_DATA subresource_data{};
  subresource_data.pData = data;
  subresource_data.RowPitch = static_cast<LONG_PTR>(bytes_size_);
  subresource_data.SlicePitch = subresource_data.RowPitch;

  // Upload the input tensor to the GPU.
  ::UpdateSubresources(device->command_list.Get(), resource_ptr,
                       upload_buffer.Get(),
                       0, 0, 1, &subresource_data);

  device->command_list->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             resource_ptr, D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

  device->CloseExecuteResetWait();

  return absl::OkStatus();
}

absl::Status D3DResource::CopyResource(DMLDevice* device,
                                     const D3DResource& src_resource) {
  D3D12_RESOURCE_BARRIER barrier[2] = {
      CD3DX12_RESOURCE_BARRIER::Transition(
        src_resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE),
      CD3DX12_RESOURCE_BARRIER::Transition(
        resource_ptr, D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
  };
  device->command_list->ResourceBarrier(2, barrier);

  device->command_list->CopyResource(resource_ptr, src_resource.Get());

  device->CloseExecuteResetWait();

  return absl::OkStatus();
}


absl::Status D3DResource::Copy(DMLDevice* device,
                               const D3DResource& src_resource) {
  if (src_resource.bytes_size() > bytes_size_) {
    return absl::InvalidArgumentError(
        "Copy to buffer failed. Source data is larger than buffer.");
  }
  return CopyResource(device, src_resource);
}

absl::Status CreateResource(DMLDevice* device, AccessType access_type,
                            DML_TENSOR_DATA_TYPE data_type, UINT64 size,
                            D3DResource* d3d_resource) {
  ComPtr<ID3D12Resource> resource;

  DML_CHECK_SUCCEEDED(device->d3d_device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
      D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
      access_type == AccessType::WRITE ? D3D12_RESOURCE_STATE_COPY_DEST
                                       : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      nullptr,
      IID_PPV_ARGS(&resource)));

  *d3d_resource = D3DResource(resource, data_type, size);
  return absl::OkStatus();
}

}  // namespace dml
}  // namespace gpu
}  // namespace tflite
