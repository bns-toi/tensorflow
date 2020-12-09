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

class UniqueHandle {
 public:
  explicit UniqueHandle(HANDLE handle) : m_handle(handle) {}
  UniqueHandle(const UniqueHandle&) = delete;
  UniqueHandle(UniqueHandle&& other) {
    m_handle = std::move(other.m_handle);
    other.m_handle = nullptr;
  }
  ~UniqueHandle() {
    if (m_handle) {
      CloseHandle(m_handle);
      m_handle = nullptr;
    }
  }
  UniqueHandle& operator=(UniqueHandle& other) = delete;
  UniqueHandle& operator=(UniqueHandle&& other) {
    m_handle = std::move(other.m_handle);
    other.m_handle = nullptr;
    return *this;
  }
  HANDLE get() { return m_handle; }
  operator bool() const { return m_handle; }

 private:
  HANDLE m_handle = nullptr;
};

DMLDevice::DMLDevice(
    Microsoft::WRL::ComPtr<IDXGIFactory4>& factory,
    Microsoft::WRL::ComPtr<ID3D12Device>& device,
    Microsoft::WRL::ComPtr<ID3D12CommandQueue>& command_queue_,
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator>& command_allocator_,
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>& command_list_)
    : dxgi_factory(factory),
      d3d_device_ptr(device),
      command_queue_ptr(command_queue_),
      command_allocator_ptr(command_allocator_),
      command_list_ptr(command_list_),
      d3d_device(d3d_device_ptr.Get()),
      command_queue(command_queue_ptr.Get()),
      command_allocator(command_allocator_ptr.Get()),
      command_list(command_list_ptr.Get()) {}

DMLDevice::DMLDevice(ID3D12Device* device, ID3D12CommandQueue* command_queue_,
                     ID3D12CommandAllocator* command_allocator_,
                     ID3D12GraphicsCommandList* command_list_)
    : d3d_device(device),
      command_queue(command_queue_),
      command_allocator(command_allocator_),
      command_list(command_list_) {}

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
  // create device
  const D3D_FEATURE_LEVEL feature_level = D3D_FEATURE_LEVEL_11_0;
  ComPtr<IDXGIAdapter1> adapter;

  DWORD dxgi_factory_flags = 0;
  Microsoft::WRL::ComPtr<IDXGIFactory4> dxgi_factory;
  DML_CHECK_SUCCEEDED(CreateDXGIFactory2(dxgi_factory_flags, IID_PPV_ARGS(dxgi_factory.ReleaseAndGetAddressOf())));

  ComPtr<IDXGIFactory6> dxgi_factory6;
  //DML_CHECK_SUCCEEDED(CreateDXGIFactory(IID_PPV_ARGS(&dxgi_factory)));
  HRESULT hr = dxgi_factory.As(&dxgi_factory6);
  if (SUCCEEDED(hr)) {
    uint32_t adapter_index = 0;
    while (dxgi_factory6->EnumAdapterByGpuPreference(
               adapter_index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
               IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND) {
      ++adapter_index;

      DXGI_ADAPTER_DESC1 desc = {};
      DML_CHECK_SUCCEEDED(adapter->GetDesc1(&desc));
      if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
        continue;
      }

      HRESULT hr = D3D12CreateDevice(adapter.Get(), feature_level,
                                     IID_ID3D12Device, nullptr);
      if (SUCCEEDED(hr)) {
        break;
      }

      adapter = nullptr;
    }
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

  ComPtr<ID3D12CommandAllocator> command_allocator;
  DML_CHECK_SUCCEEDED(d3d_device->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&command_allocator)));

  ComPtr<ID3D12GraphicsCommandList> command_list;
  DML_CHECK_SUCCEEDED(d3d_device->CreateCommandList(
      0, D3D12_COMMAND_LIST_TYPE_DIRECT, command_allocator.Get(), nullptr,
      IID_PPV_ARGS(&command_list)));

  // construct
  *result = DMLDevice(dxgi_factory, d3d_device, command_queue,
                      command_allocator,
                      command_list);
  return absl::OkStatus();
}

void DMLDevice::CloseExecuteResetWait() {
  DML_CHECK_SUCCEEDED(command_list->Close());

  ID3D12CommandList* command_lists[] = {command_list};
  command_queue->ExecuteCommandLists(ARRAYSIZE(command_lists), command_lists);

  DML_CHECK_SUCCEEDED(command_list->Reset(command_allocator, nullptr));

  ComPtr<ID3D12Fence> fence;
  DML_CHECK_SUCCEEDED(
      d3d_device->CreateFence(
      0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

  UniqueHandle fence_event_handle(CreateEvent(nullptr, TRUE, FALSE, nullptr));

  DML_CHECK_SUCCEEDED(fence->SetEventOnCompletion(1, fence_event_handle.get()));

  DML_CHECK_SUCCEEDED(command_queue->Signal(fence.Get(), 1));
  ::WaitForSingleObjectEx(fence_event_handle.get(), INFINITE, FALSE);
}

}  // namespace dml
}  // namespace gpu
}  // namespace tflite
