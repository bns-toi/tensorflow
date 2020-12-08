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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_DML_RUNTIME_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_DML_RUNTIME_H_

#include <vector>
#include <memory>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/dml/environment.h"
#include "tensorflow/lite/delegates/gpu/dml/object_manager.h"

namespace tflite {
namespace gpu {
namespace dml {
class Runtime {
 public:
  Runtime(DMLDevice* device, const ObjectManager* external_objects);

  absl::Status Compile(const GraphFloat32& graph);
  absl::Status Execute();

 private:
  DMLDevice* device;
  const ObjectManager* external_objects_;

  Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_operator;
  Microsoft::WRL::ComPtr<IDMLOperatorInitializer> operator_initializer;
  UINT descriptor_count;
  Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptor_heap;
  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  UINT64 temporary_resource_size;
  UINT64 persistent_resource_size;
  Microsoft::WRL::ComPtr<ID3D12Resource> temporary_buffer;
  Microsoft::WRL::ComPtr<ID3D12Resource> persistent_buffer;
  Microsoft::WRL::ComPtr<IDMLCommandRecorder> command_recorder;
  std::vector<uint32_t> input_ids;
  std::vector<uint32_t> output_ids;
};
}  // namespace dml
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_DML_RUNTIME_H_
