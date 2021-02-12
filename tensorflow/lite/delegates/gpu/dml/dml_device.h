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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_DML_DML_DEVICE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_DML_DML_DEVICE_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/dml/dml_common.h"

namespace tflite {
namespace gpu {
namespace dml {

// A wrapper around directml device
class DMLDevice {
 public:
  DMLDevice() = default;
  DMLDevice(Microsoft::WRL::ComPtr<ID3D12Device>& device);
  DMLDevice(ID3D12Device* device);

  ~DMLDevice() {}

  void Init();
  void CloseExecuteResetWait();

  Microsoft::WRL::ComPtr<ID3D12Device> d3d_device_ptr;
  ID3D12Device* d3d_device;

  Microsoft::WRL::ComPtr<ID3D12CommandQueue> command_queue;
  Microsoft::WRL::ComPtr<ID3D12CommandAllocator> command_allocator;
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> command_list;

  Microsoft::WRL::ComPtr<IDMLDevice> dml_device;
};

absl::Status CreateDefaultGPUDevice(DMLDevice* result);

}  // namespace dml
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_DML_DML_DEVICE_H_
