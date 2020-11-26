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

#include "tensorflow/lite/delegates/gpu/dml/dml_device.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"

using Microsoft::WRL::ComPtr;

namespace tflite {
namespace gpu {
namespace dml {

DMLDevice::DMLDevice(Microsoft::WRL::ComPtr<ID3D12Device>& device, Microsoft::WRL::ComPtr<ID3D12CommandQueue>& command_queue)
    : d3d_device_ptr(device),
      command_queue_ptr(command_queue),
      d3d_device(d3d_device_ptr.Get()),
      command_queue_(command_queue_ptr.Get()) {}

DMLDevice::DMLDevice(ID3D12Device* device, ID3D12CommandQueue* command_queue)
    : d3d_device(device), command_queue_(command_queue) {}

void DMLDevice::Init() {

  // create DirectML device
  DML_CREATE_DEVICE_FLAGS dml_flags = DML_CREATE_DEVICE_FLAG_NONE;
#if defined(_DEBUG)
//  dml_flags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif

  DML_CHECK_SUCCEEDED(
      DMLCreateDevice(d3d_device, dml_flags, IID_PPV_ARGS(&dml_device)));
}

absl::Status CreateDefaultGPUDevice(DMLDevice* result) {
  ComPtr<IDXGIFactory6> dxgi_factory;
  DML_CHECK_SUCCEEDED(CreateDXGIFactory(IID_PPV_ARGS(&dxgi_factory)));

  // create device
  const D3D_FEATURE_LEVEL feature_level = D3D_FEATURE_LEVEL_11_0;

  uint32_t adapter_index = 0;
  ComPtr<IDXGIAdapter1> adapter;
  while (dxgi_factory->EnumAdapterByGpuPreference(
             adapter_index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
             IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND) {
    DXGI_ADAPTER_DESC1 desc = {};
    DML_CHECK_SUCCEEDED(adapter->GetDesc1(&desc));

    HRESULT hr = D3D12CreateDevice(adapter.Get(), feature_level,
                                   IID_ID3D12Device, nullptr);
    if (SUCCEEDED(hr)) {
      break;
    }

    ++adapter_index;
    adapter = nullptr;
  }

  ComPtr<ID3D12Device> d3d_device;
  DML_CHECK_SUCCEEDED(D3D12CreateDevice(adapter.Get(), feature_level,
                                        IID_PPV_ARGS(&d3d_device)));

  // create command queue
  D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
  command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  command_queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
  command_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
  command_queue_desc.NodeMask = 0;

  ComPtr<ID3D12CommandQueue> command_queue;
  DML_CHECK_SUCCEEDED(d3d_device->CreateCommandQueue(
      &command_queue_desc, IID_PPV_ARGS(&command_queue)));

  // construct
  *result = DMLDevice(d3d_device, command_queue);
  return absl::OkStatus();
}

}  // namespace dml
}  // namespace gpu
}  // namespace tflite
