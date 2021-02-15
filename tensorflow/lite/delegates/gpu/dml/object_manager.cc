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

#include "tensorflow/lite/delegates/gpu/dml/object_manager.h"

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace dml {

absl::Status ObjectManager::RegisterResource(uint32_t id, D3DResource resource) {
  if (id >= resources_.size()) {
    resources_.resize(id + 1);
  }
  resources_[id] = absl::make_unique<D3DResource>(std::move(resource));
  return absl::OkStatus();
}

void ObjectManager::RemoveResource(uint32_t id) {
  if (id < resources_.size()) {
    resources_[id].reset(nullptr);
  }
}

void ObjectManager::RemoveAllResource() {
  resources_.clear();
}

D3DResource* ObjectManager::FindResource(uint32_t id) const {
  return id >= resources_.size() ? nullptr : resources_[id].get();
}

bool ObjectManager::IsRegistered(D3DResource* resource) const {
  for (int i = 0; i < resources_.size(); i++) {
    if (resources_[i].get() == resource) {
      return true;
    }
  }
  return false;
}

}  // namespace dml
}  // namespace gpu
}  // namespace tflite
