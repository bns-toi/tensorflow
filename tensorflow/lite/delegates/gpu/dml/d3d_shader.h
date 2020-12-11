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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_DML_D3D_SHADER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_DML_D3D_SHADER_H_

#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/dml/dml_common.h"
#include "tensorflow/lite/delegates/gpu/dml/dml_device.h"

namespace tflite {
namespace gpu {
namespace dml {

class D3DShader {
 public:
  D3DShader();
  ~D3DShader();

  absl::Status Compile(DMLDevice* device, const std::string& shader_source);
  void Release();
  absl::Status Dispatch(DMLDevice* device, UINT width, UINT height,
                        const DirectMlResource* input,
                        const DirectMlResource* output);

private:
  ID3DBlob* shader;
  Microsoft::WRL::ComPtr<ID3D12RootSignature> root_signature;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline_stat;
  Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptor_heap;
  bool init_uav;
};

}  // namespace dml
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_DML_D3D_SHADER_H_
