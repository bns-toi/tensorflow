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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_DML_ENVIRONMENT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_DML_ENVIRONMENT_H_

#include "tensorflow/lite/delegates/gpu/dml/dml_device.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace dml {

class Environment {
 public:
  Environment() = default;
  explicit Environment(DMLDevice&& device);
  // Move only
  Environment(Environment&& environment);
  Environment& operator=(Environment&& environment);
  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;

  const DMLDevice& device() const { return device_; }
  DMLDevice* GetDevicePtr() { return &device_; }
  const DMLDevice* GetDevicePtr() const { return &device_; }

  absl::Status Init();

 private:
  DMLDevice device_;
};

}  // namespace dml
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_DML_ENVIRONMENT_H_
