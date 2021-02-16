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

#ifdef CONST
#undef CONST
#endif // CONST

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/dml/environment.h"
#include "tensorflow/lite/delegates/gpu/dml/object_manager.h"

// Let DirectML manage the data in the weight tensors. This can be faster on
// some hardware.
#define DML_MANAGED_WEIGHTS 1

namespace tflite {
namespace gpu {
namespace dml {

class Runtime {
 public:
  Runtime(DMLDevice* device, const ObjectManager* external_objects,
          bool allow_precision_loss);

  absl::Status Compile(const GraphFloat32& graph);
  absl::Status Execute();

  absl::Status CreateConstInputTensorExpression(
      ::dml::Scope& scope, DML_TENSOR_FLAGS flags,
      const ::dml::TensorPolicy& policy, const uint8_t* data, const UINT* sizes,
      ::dml::Expression& output);

 private:
  DMLDevice* device;
  const ObjectManager* external_objects_;
  const bool allow_precision_loss_;
  ObjectManager const_objects_;
  uint32_t next_const_id_ = 0;  // id for const objects

  Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_operator;
  Microsoft::WRL::ComPtr<IDMLOperatorInitializer> operator_initializer;
  UINT descriptor_count;
  Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptor_heap;
  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  Microsoft::WRL::ComPtr<ID3D12Resource> temporary_buffer;
  Microsoft::WRL::ComPtr<ID3D12Resource> persistent_buffer;
  Microsoft::WRL::ComPtr<IDMLCommandRecorder> command_recorder;
  std::vector<D3DResource*> input_resources;
  std::vector<D3DResource*> output_resources;

  D3DResource* AllocateConstObject(const uint8_t* data,
                                   DML_TENSOR_DATA_TYPE data_type,
                                   uint32_t size);

  absl::Status CreateOperator(const GraphFloat32& graph);

  absl::Status CreateInputTensorExpression(
      ::dml::Scope& scope, DML_TENSOR_FLAGS flags,
      const ::dml::TensorPolicy& policy, const Value* value,
      ::dml::Expression& output);


  absl::Status CreateAddExpression(const GraphFloat32& graph, const Node* node,
                                   const std::vector<::dml::Expression>& inputs,
                                   std::vector<::dml::Expression>& outputs);
  absl::Status CreateConcatExpression(const GraphFloat32& graph, const Node* node,
      const std::vector<::dml::Expression>& inputs,
      std::vector<::dml::Expression>& outputs);
  absl::Status CreateConvolution2DExpression(
      ::dml::Scope& scope, DML_TENSOR_FLAGS flags,
      const ::dml::TensorPolicy& policy, const GraphFloat32& graph,
      const Node* node, OperationType activation_type,
      const std::vector<::dml::Expression>& inputs,
      std::vector<::dml::Expression>& outputs);
  absl::Status CreatePadExpression(const GraphFloat32& graph, const Node* node,
                                   const std::vector<::dml::Expression>& inputs,
                                   std::vector<::dml::Expression>& outputs);
  absl::Status CreateSliceExpression(
      const GraphFloat32& graph, const Node* node,
      const std::vector<::dml::Expression>& inputs,
      std::vector<::dml::Expression>& outputs);
  absl::Status CreateClipExpression(
      const GraphFloat32& graph, const Node* min_node, const Node* max_node,
      const std::vector<::dml::Expression>& inputs,
      std::vector<::dml::Expression>& outputs);
  absl::Status CreateReLUExpression(
      const GraphFloat32& graph, const Node* node,
      const std::vector<::dml::Expression>& inputs,
      std::vector<::dml::Expression>& outputs);
  absl::Status CreateTanhExpression(
      const GraphFloat32& graph, const Node* node,
      const std::vector<::dml::Expression>& inputs,
      std::vector<::dml::Expression>& outputs);
  absl::Status CreateMaximumExpression(
      const GraphFloat32& graph, const Node* node,
      const std::vector<::dml::Expression>& inputs,
      std::vector<::dml::Expression>& outputs);
  absl::Status CreateMinimumExpression(
      const GraphFloat32& graph, const Node* node,
      const std::vector<::dml::Expression>& inputs,
      std::vector<::dml::Expression>& outputs);
  absl::Status CreateScaleBiasExpression(float scale, float bias,
                                   const std::vector<::dml::Expression>& inputs,
                                   std::vector<::dml::Expression>& outputs);
  absl::Status CreateMulExpression(const GraphFloat32& graph, const Node* node,
                                   const std::vector<::dml::Expression>& inputs,
                                   std::vector<::dml::Expression>& outputs);
  absl::Status CreateSubExpression(const GraphFloat32& graph, const Node* node,
                                   const std::vector<::dml::Expression>& inputs,
                                   std::vector<::dml::Expression>& outputs);


};

}  // namespace dml
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_DML_RUNTIME_H_
