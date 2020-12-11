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
  D3DResource(Microsoft::WRL::ComPtr<ID3D12Resource>& resource,
              size_t bytes_size)
      : resource_(resource), resource_ptr(resource_.Get()), bytes_size_(bytes_size) {}
  D3DResource(ID3D12Resource* resource, size_t bytes_size)
      : resource_ptr(resource), bytes_size_(bytes_size) {}

  ~D3DResource();

  // Reads data from buffer into CPU memory. Data should point to a region that
  // has at least bytes_size available.
  template <typename T>
  absl::Status Read(DMLDevice* device, absl::Span<T> data) const;

  // Writes data to a buffer.
  template <typename T>
  absl::Status Write(DMLDevice* device, absl::Span<const T> data);

  absl::Status Copy(DMLDevice* device, const D3DResource& src_resource);

  ID3D12Resource* Get() const { return resource_ptr; }
  size_t bytes_size() const { return bytes_size_; }

 private:
  Microsoft::WRL::ComPtr<ID3D12Resource> resource_;
  ID3D12Resource* resource_ptr;
  size_t bytes_size_;

  absl::Status ReadResource(DMLDevice* device, void* data) const;
  absl::Status WriteResource(DMLDevice* device, const void* data);
  absl::Status CopyResource(DMLDevice* device, const D3DResource& src_resource);
};

absl::Status GetResourceSize(ID3D12Resource* resource, int64_t* size_bytes);

absl::Status CreateResource(DMLDevice* device,
                            AccessType access_type, UINT64 size,
                            D3DResource* d3d_resource);

template <typename T>
absl::Status D3DResource::Read(DMLDevice* device, absl::Span<T> data) const {
  if (data.size() * sizeof(T) < bytes_size()) {
    return absl::InvalidArgumentError(
        "Read from buffer failed. Destination data is shorter than buffer.");
  }
  return ReadResource(device, data.data());
}

template <typename T>
absl::Status D3DResource::Write(DMLDevice* device, absl::Span<const T> data) {
  if (data.size() * sizeof(T) > bytes_size_) {
    return absl::InvalidArgumentError(
        "Write to buffer failed. Source data is larger than buffer.");
  }
  return WriteResource(device, data.data());
}

}  // namespace dml
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_DML_D3D_RESOURCE_H_
