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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_DML_D3D_RESOURCE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_DML_D3D_RESOURCE_H_

#include <cstring>
#include <functional>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/dml/dml_common.h"
#include "tensorflow/lite/delegates/gpu/dml/dml_device.h"

namespace tflite {
namespace gpu {
namespace dml {

class D3DResource {
 public:
  D3DResource() {}
  D3DResource(Microsoft::WRL::ComPtr<ID3D12Resource>& resource, UINT64 size)
      : resource_(resource), size_(size) {}

  ~D3DResource();

  ID3D12Resource* Get() const { return resource_ .Get(); }
  UINT64 size() const { return size_; }

 private:
  Microsoft::WRL::ComPtr<ID3D12Resource> resource_;
  UINT64 size_;
};

absl::Status CreateResource(DMLDevice* device,
                            AccessType access_type, UINT64 size,
                            D3DResource* d3d_resource);

}  // namespace dml
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_DML_D3D_RESOURCE_H_
