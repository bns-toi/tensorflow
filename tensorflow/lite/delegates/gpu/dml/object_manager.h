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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_DML_OBJECT_MANAGER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_DML_OBJECT_MANAGER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/dml/d3d_resource.h"
#include "tensorflow/lite/delegates/gpu/gl/stats.h"

namespace tflite {
namespace gpu {
namespace dml {
class ObjectManager {
 public:
  // Moves ownership over the given resource to the manager.
  absl::Status RegisterResource(uint32_t id, D3DResource resource);

  void RemoveResource(uint32_t id);
  void RemoveAllResource();

  // Return a permanent pointer to a resource for the given id or nullptr.
  D3DResource* FindResource(uint32_t id) const;

  bool IsRegistered(D3DResource* resource) const;

 private:
  std::vector<std::unique_ptr<D3DResource>> resources_;
};

}  // namespace dml
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_DML_OBJECT_MANAGER_H_