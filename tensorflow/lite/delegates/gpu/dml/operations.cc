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

#include "tensorflow/lite/delegates/gpu/dml/runtime.h"
#include "tensorflow/lite/delegates/gpu/dml/Float16Compressor.h"

#define OPTIMIZE_NODE 1

using Microsoft::WRL::ComPtr;

namespace tflite {
namespace gpu {
namespace dml {

namespace {

bool Is4Aligned(const SliceAttributes& attr) {
  return attr.strides.c == 1 && attr.starts.c % 4 == 0;
}

int4 GetOffset(const SliceAttributes& attr, int src_width, int src_height,
               int src_channels, int src_batch) {
  int4 offset;
  if (attr.strides.w > 0) {
    offset.x = attr.starts.w;
  } else {
    if (attr.ends.w > 0) {
      offset.x = attr.ends.w;
    } else {
      offset.x = src_width + attr.ends.w;
    }
  }
  if (attr.strides.h > 0) {
    offset.y = attr.starts.h;
  } else {
    if (attr.ends.h > 0) {
      offset.y = attr.ends.h;
    } else {
      offset.y = src_height + attr.ends.h;
    }
  }
  if (attr.strides.c > 0) {
    offset.z = attr.starts.c;
  } else {
    if (attr.ends.c > 0) {
      offset.z = attr.ends.c;
    } else {
      offset.z = src_channels + attr.ends.c;
    }
  }
  if (Is4Aligned(attr)) {
    offset.z /= 4;
  }
  if (attr.strides.b > 0) {
    offset.w = attr.starts.b;
  } else {
    if (attr.ends.b > 0) {
      offset.w = attr.ends.b;
    } else {
      offset.w = src_batch + attr.ends.b;
    }
  }
  return offset;
}

void GetStrides(const uint32_t* sizes,
                /*TensorLayout layout,*/ uint32_t* stridesOut) {
  /*  switch (layout) {
      case TensorLayout::NHWC:
        stridesOut[0] = sizes[1] * sizes[2] * sizes[3];
        stridesOut[1] = 1;
        stridesOut[2] = sizes[1] * sizes[3];
        stridesOut[3] = sizes[1];
        break;

      default:*/
  stridesOut[0] = sizes[1] * sizes[2] * sizes[3];
  stridesOut[1] = sizes[2] * sizes[3];
  stridesOut[2] = sizes[3];
  stridesOut[3] = 1;
  //  }
}

}  // namespace

absl::Status Runtime::CreateOperator(const GraphFloat32& graph) {
  DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;
#if DML_MANAGED_WEIGHTS
  flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
#endif // DML_MANAGED_WEIGHTS
  ::dml::TensorPolicy policy = ::dml::TensorPolicy::Default();
//  ::dml::TensorPolicy policy = ::dml::TensorPolicy::InterleavedChannel();
  ::dml::Scope scope(device->dml_device.Get(), policy);

  std::map<ValueId, ::dml::Expression> expressions;
  ::dml::Expression last_output;

  // Create external input & output
  for (auto value : graph.values()) {
    if (graph.IsGraphInput(value->id)) {
      RETURN_IF_ERROR(CreateInputTensorExpression(scope, flags, policy, value,
                                                  last_output));
      expressions[value->id] = last_output;
    } else if (graph.IsGraphOutput(value->id)) {
      output_resources.push_back(
          external_objects_->FindResource(value->id));
    }
  }

  // Create expression from graph
  std::vector<Node*> graph_nodes = graph.nodes();
  const uint32_t num_graph_node = graph_nodes.size();
  for (uint32_t i = 0; i < num_graph_node; ++i) {
    // Get current and next nodes
    const Node* node = graph_nodes[i];
    const Node* next_node =
        (i + 1) < num_graph_node ? graph_nodes[i + 1] : nullptr;

    OperationType op_type = OperationTypeFromString(node->operation.type);
    OperationType next_op_type =
        next_node ? OperationTypeFromString(next_node->operation.type)
                  : OperationType::UNKNOWN;

    // Reserve and get expressions
    std::vector<::dml::Expression> inputs;
    std::vector<::dml::Expression> outputs;
    auto input_values = graph.FindInputs(node->id);
    auto output_values = graph.FindOutputs(node->id);
    inputs.reserve(input_values.size());
    outputs.reserve(output_values.size());
    for (auto value : input_values) {
      inputs.push_back(expressions[value->id]);
    }

    // Behavior for each operation
    switch (op_type) {
      case OperationType::ADD: {
        RETURN_IF_ERROR(CreateAddExpression(graph, node, inputs, outputs));
      } break;
      case OperationType::CONCAT: {
        RETURN_IF_ERROR(CreateConcatExpression(graph, node, inputs, outputs));
      } break;
      case OperationType::CONVOLUTION_2D: {
        OperationType activation_type = OperationType::UNKNOWN;
#if OPTIMIZE_NODE
        if (next_op_type == OperationType::RELU ||
            next_op_type == OperationType::TANH) {
          activation_type = next_op_type;
          i++;
          output_values = graph.FindOutputs(next_node->id);
        }
#endif  // OPTIMIZE_NODE
        RETURN_IF_ERROR(
            CreateConvolution2DExpression(scope, flags, policy, graph, node,
                                          activation_type, inputs, outputs));
      } break;
      case OperationType::MUL: {
#if OPTIMIZE_NODE
        if (next_op_type == OperationType::SUB) {
          i++;
          output_values = graph.FindOutputs(next_node->id);

          auto mul_attr =
              absl::any_cast<MultiplyAttributes>(node->operation.attributes);
          const float* scale = absl::get_if<float>(&mul_attr.param);
          if (scale == nullptr) {
            return absl::InvalidArgumentError("Not supported attribute.");
          }

          auto sub_attr = absl::any_cast<ElementwiseAttributes>(
              next_node->operation.attributes);
          const float* bias = absl::get_if<float>(&sub_attr.param);
          if (bias == nullptr) {
            return absl::InvalidArgumentError("Not supported attribute.");
          }

          RETURN_IF_ERROR(
              CreateScaleBiasExpression(*scale, -(*bias), inputs, outputs));
        } else
#endif  // OPTIMIZE_NODE
        {
          RETURN_IF_ERROR(CreateMulExpression(graph, node, inputs, outputs));
        }
      } break;
      case OperationType::PAD: {
        RETURN_IF_ERROR(CreatePadExpression(graph, node, inputs, outputs));
      } break;
      case OperationType::SLICE: {
        RETURN_IF_ERROR(CreateSliceExpression(graph, node, inputs, outputs));
      } break;
      case OperationType::RELU: {
        RETURN_IF_ERROR(CreateReLUExpression(graph, node, inputs, outputs));
      } break;
      case OperationType::TANH: {
        RETURN_IF_ERROR(CreateTanhExpression(graph, node, inputs, outputs));
      } break;
      case OperationType::MAXIMUM: {
        RETURN_IF_ERROR(CreateMaximumExpression(graph, node, inputs, outputs));
      } break;
      case OperationType::MINIMUM: {
#if OPTIMIZE_NODE
        if (next_op_type == OperationType::MAXIMUM) {
          i++;
          output_values = graph.FindOutputs(next_node->id);

          RETURN_IF_ERROR(
              CreateClipExpression(graph, node, next_node, inputs, outputs));
        } else
#endif  // OPTIMIZE_NODE
        {
          RETURN_IF_ERROR(CreateMinimumExpression(graph, node, inputs, outputs));
        }
      } break;
      case OperationType::SUB: {
        RETURN_IF_ERROR(CreateSubExpression(graph, node, inputs, outputs));
      } break;
      default:
        return absl::UnimplementedError(absl::StrCat(
            "No support of ", node->operation.type, " with this parameters"));
    }

    // Store outputs
    const int num_output = output_values.size();
    if (outputs.size() < num_output) {
      return absl::OutOfRangeError("Output inconsistency");
    }
    for (int i = 0; i < num_output; i++) {
      expressions[output_values[i]->id] = outputs[i];
    }
    last_output = outputs[0];
  }

  // Compile
  DML_EXECUTION_FLAGS execution_flags =
    allow_precision_loss_
      ? DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION
      : DML_EXECUTION_FLAG_NONE;
  compiled_operator = scope.Compile(execution_flags, {last_output});

  return absl::OkStatus();
}

absl::Status Runtime::CreateInputTensorExpression(
    ::dml::Scope& scope, DML_TENSOR_FLAGS flags,
    const ::dml::TensorPolicy& policy,
    const Value* value, ::dml::Expression& output) {
  DML_TENSOR_DATA_TYPE data_type = allow_precision_loss_
                                     ? DML_TENSOR_DATA_TYPE_FLOAT16
                                     : DML_TENSOR_DATA_TYPE_FLOAT32;
  uint32_t index = input_resources.size();
  input_resources.push_back(external_objects_->FindResource(value->id));

  const auto& shape = value->tensor.shape;
  UINT tensor_sizes[4] = {shape.b, shape.c, shape.h, shape.w};
  ::dml::TensorDesc::Dimensions dimensions(std::begin(tensor_sizes),
                                           std::end(tensor_sizes));
  output = ::dml::InputTensor(scope, index,
                              ::dml::TensorDesc(data_type, dimensions, policy));

  return absl::OkStatus();
}

absl::Status Runtime::CreateConstInputTensorExpression(
    ::dml::Scope& scope, DML_TENSOR_FLAGS flags,
    const ::dml::TensorPolicy& policy, const uint8_t* data,
    const uint32_t* sizes,
    ::dml::Expression& output) {
  DML_TENSOR_DATA_TYPE data_type = allow_precision_loss_
                                     ? DML_TENSOR_DATA_TYPE_FLOAT16
                                     : DML_TENSOR_DATA_TYPE_FLOAT32;
  uint32_t strides[4];
  GetStrides(sizes, /*m_tensorLayout,*/ strides);
  uint64_t buffer_size = DMLCalcBufferTensorSize(data_type, 4, sizes, strides);

  uint32_t index = input_resources.size();
  D3DResource* resource = AllocateConstObject(data, data_type, buffer_size);
  input_resources.push_back(resource);

  ::dml::TensorDesc::Dimensions dimensions = {sizes[0], sizes[1], sizes[2],
                                              sizes[3]};
  output = ::dml::InputTensor(
      scope, index, ::dml::TensorDesc(data_type, flags, dimensions, policy));

  return absl::OkStatus();
}

absl::Status Runtime::CreateAddExpression(
    const GraphFloat32& graph, const Node* node,
    const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
  if (inputs.size() >= 2) {
    outputs.push_back(::dml::Add(inputs[0], inputs[1]));
  } else {
    auto attr = absl::any_cast<AddAttributes>(node->operation.attributes);
    const float* scalar = absl::get_if<float>(&attr.param);
    if (scalar == nullptr) {
      return absl::InvalidArgumentError("Not supported attribute.");
    }

    DML_SCALE_BIAS scale_bias = {1.0f, *scalar};
    outputs.push_back(::dml::Identity(inputs[0], scale_bias));
  }

  return absl::OkStatus();
}

absl::Status Runtime::CreateConcatExpression(
    const GraphFloat32& graph, const Node* node,
    const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
  auto attr = absl::any_cast<ConcatAttributes>(node->operation.attributes);
  outputs.push_back(::dml::Join(::dml::Span<const ::dml::Expression>(inputs),
                                static_cast<uint32_t>(attr.axis)));

  return absl::OkStatus();
}

void SetValue(uint16_t& output, const float& input) {
  output = Float16Compressor::compress(input);
}

void SetValue(float& output, const float& input) {
  output = input;
}

template <typename T>
absl::Status CreateWeightsTensorExpression(
    Runtime* runtime, ::dml::Scope& scope, DML_TENSOR_FLAGS flags,
    const ::dml::TensorPolicy& policy,
    const Tensor<OHWI, DataType::FLOAT32>& weights, ::dml::Expression& output) {
  const auto& weights_shape = weights.shape;

  std::vector<T> gpu_data;
  gpu_data.resize(weights_shape.o * weights_shape.h * weights_shape.w *
                  weights_shape.i);
  for (uint32_t o = 0; o < weights_shape.o; o++) {
    for (uint32_t h = 0; h < weights_shape.h; h++) {
      for (uint32_t w = 0; w < weights_shape.w; w++) {
        for (uint32_t i = 0; i < weights_shape.i; i++) {
          uint32_t offset =
              o * weights_shape.i * weights_shape.h * weights_shape.w;
          uint32_t idx = w + h * weights_shape.w;

          SetValue(
              gpu_data[offset + idx + i * weights_shape.h * weights_shape.w],
              weights.data[offset + idx * weights_shape.i + i]);
        }
      }
    }
  }

  UINT weights_tensor_sizes[4] = {weights_shape.o, weights_shape.i,
                                  weights_shape.h, weights_shape.w};

  return runtime->CreateConstInputTensorExpression(
      scope, flags, policy, reinterpret_cast<const uint8_t*>(gpu_data.data()),
      weights_tensor_sizes, output);
}

absl::Status CreateBiasTensorExpressionHalf(
    Runtime* runtime, ::dml::Scope& scope, DML_TENSOR_FLAGS flags,
    const ::dml::TensorPolicy& policy,
    const Tensor<Linear, DataType::FLOAT32>& bias, ::dml::Expression& output) {
  const auto& bias_shape = bias.shape;

  std::vector<uint16_t> gpu_data;
  const UINT num_bias = bias.data.size();
  gpu_data.reserve(num_bias);
  for (auto bias : bias.data) {
    gpu_data.push_back(Float16Compressor::compress(bias));
  }

  UINT bias_tensor_sizes[4] = {1, bias_shape.v, 1, 1};

  return runtime->CreateConstInputTensorExpression(
      scope, flags, policy, reinterpret_cast<const uint8_t*>(gpu_data.data()),
      bias_tensor_sizes, output);
}

absl::Status CreateBiasTensorExpression(
    Runtime* runtime, ::dml::Scope& scope, DML_TENSOR_FLAGS flags,
    const ::dml::TensorPolicy& policy,
    const Tensor<Linear, DataType::FLOAT32>& bias, ::dml::Expression& output) {
  const auto& bias_shape = bias.shape;

  UINT bias_tensor_sizes[4] = {1, bias_shape.v, 1, 1};

  return runtime->CreateConstInputTensorExpression(
      scope, flags, policy,
      reinterpret_cast<const uint8_t*>(bias.data.data()),
      bias_tensor_sizes, output);
}

absl::Status Runtime::CreateConvolution2DExpression(
    ::dml::Scope& scope, DML_TENSOR_FLAGS flags,
    const ::dml::TensorPolicy& policy, const GraphFloat32& graph,
    const Node* node, OperationType activation_type,
    const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
  auto attr =
      absl::any_cast<Convolution2DAttributes>(node->operation.attributes);

#if 1
  // Weights & Bias Expression
  ::dml::Expression filter;
  ::dml::Expression bias;
  if (allow_precision_loss_) {
    CreateWeightsTensorExpression<uint16_t>(this, scope, flags, policy,
                                            attr.weights, filter);
    CreateBiasTensorExpressionHalf(this, scope, flags, policy, attr.bias, bias);
  } else {
    CreateWeightsTensorExpression<float>(this, scope, flags, policy,
                                            attr.weights, filter);
    CreateBiasTensorExpression(this, scope, flags, policy, attr.bias, bias);
  }

  // Parameters
  const uint32_t strides[2] = {attr.strides.h, attr.strides.w};
  const uint32_t dilations[2] = {attr.dilations.h, attr.dilations.w};
  const uint32_t start_padding[2] = {attr.padding.prepended.h,
                                     attr.padding.prepended.w};
  const uint32_t end_padding[2] = {attr.padding.appended.h,
                                   attr.padding.appended.w};

  // Convolution 2D Expression
  ::dml::FusedActivation activation;
  switch (activation_type) {
    case OperationType::RELU: {
      activation = ::dml::FusedActivation::Relu();
    } break;
    case OperationType::TANH: {
      activation = ::dml::FusedActivation::Tanh();
    }  break;
    default:
      activation = ::dml::FusedActivation::None();
      break;
  }
  outputs.push_back(
      ::dml::ConvolutionBuilder(inputs[0], filter, bias)
          .Strides(::dml::Span<const uint32_t>{strides})
          .Dilations(::dml::Span<const uint32_t>{dilations})
          .StartPadding(::dml::Span<const uint32_t>{start_padding})
          .EndPadding(::dml::Span<const uint32_t>{end_padding})
          .FusedActivation(activation)
          .Build());
#else
  outputs.push_back(::dml::Identity(inputs[0]);
#endif

  return absl::OkStatus();
}

absl::Status Runtime::CreatePadExpression(
    const GraphFloat32& graph, const Node* node,
    const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
  auto attr = absl::any_cast<PadAttributes>(node->operation.attributes);
  DML_PADDING_MODE padding_mode = attr.type == PaddingContentType::ZEROS
                                      ? DML_PADDING_MODE_CONSTANT
                                      : attr.type == PaddingContentType::REFLECT
                                            ? DML_PADDING_MODE_REFLECTION
                                            : DML_PADDING_MODE_EDGE;
  const uint32_t start_padding[4] = {attr.prepended.b, attr.prepended.c,
                                     attr.prepended.h, attr.prepended.w};
  const uint32_t end_padding[4] = {attr.appended.b, attr.appended.c,
                                   attr.appended.h, attr.appended.w};

#if 1
  outputs.push_back(::dml::Padding(inputs[0], padding_mode, 0.0f,
                                   ::dml::Span<const uint32_t>{start_padding},
                                   ::dml::Span<const uint32_t>{end_padding}));
#else
  outputs.push_back(::dml::Identity(inputs[0]));
#endif

  return absl::OkStatus();
}

absl::Status Runtime::CreateSliceExpression(
    const GraphFloat32& graph, const Node* node,
    const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
  auto input_values = graph.FindInputs(node->id);
  auto output_values = graph.FindOutputs(node->id);
  auto attr = absl::any_cast<SliceAttributes>(node->operation.attributes);

  const auto& input_shape = input_values[0]->tensor.shape;
  const auto& output_shape = output_values[0]->tensor.shape;
  int4 offset = GetOffset(attr, input_shape.w, input_shape.h, input_shape.c,
                          input_shape.b);

  const uint32_t offsets[4] = {offset.w, offset.z, offset.y, offset.x};
  const uint32_t sizes[4] = {output_shape.b, output_shape.c, output_shape.h,
                             output_shape.w};
  const uint32_t strides[4] = {attr.strides.b, attr.strides.c, attr.strides.h,
                               attr.strides.w};

  outputs.push_back(::dml::Slice(inputs[0],
                                 ::dml::Span<const uint32_t>{offsets},
                                 ::dml::Span<const uint32_t>{sizes},
                                 ::dml::Span<const uint32_t>{strides}));

  return absl::OkStatus();
}

absl::Status Runtime::CreateReLUExpression(
    const GraphFloat32& graph, const Node* node,
    const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
//  auto attr = absl::any_cast<ReLUAttributes>(node->operation.attributes);
  outputs.push_back(::dml::ActivationRelu(inputs[0]));

  return absl::OkStatus();
}

absl::Status Runtime::CreateTanhExpression(
    const GraphFloat32& graph, const Node* node,
    const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
  outputs.push_back(::dml::Tanh(inputs[0]));

  return absl::OkStatus();
}

absl::Status Runtime::CreateClipExpression(
    const GraphFloat32& graph, const Node* min_node, const Node* max_node,
    const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
  float min_value = FLT_MAX;
  if (min_node) {
    auto attr =
        absl::any_cast<ElementwiseAttributes>(min_node->operation.attributes);
    min_value = *absl::get_if<float>(&attr.param);
  }
  float max_value = FLT_MIN;
  if (max_node) {
    auto attr =
        absl::any_cast<ElementwiseAttributes>(max_node->operation.attributes);
    max_value = *absl::get_if<float>(&attr.param);
  }

  outputs.push_back(::dml::Clip(inputs[0], max_value, min_value));

  return absl::OkStatus();
}

absl::Status Runtime::CreateMaximumExpression(
    const GraphFloat32& graph, const Node* node,
    const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
  return CreateClipExpression(graph, nullptr, node, inputs, outputs);
}

absl::Status Runtime::CreateMinimumExpression(
    const GraphFloat32& graph, const Node* node,
    const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
  return CreateClipExpression(graph, node, nullptr, inputs, outputs);
}

absl::Status Runtime::CreateScaleBiasExpression(
    float scale, float bias, const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
  DML_SCALE_BIAS scale_bias = {scale, bias};
  outputs.push_back(::dml::Identity(inputs[0], scale_bias));

  return absl::OkStatus();
}

absl::Status Runtime::CreateMulExpression(
    const GraphFloat32& graph, const Node* node,
    const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
  auto attr = absl::any_cast<MultiplyAttributes>(node->operation.attributes);
  const float* scalar = absl::get_if<float>(&attr.param);
  if (scalar == nullptr) {
    return absl::InvalidArgumentError("Not supported attribute.");
  }

  return CreateScaleBiasExpression(*scalar, 0.0f, inputs, outputs);
}

absl::Status Runtime::CreateSubExpression(
    const GraphFloat32& graph, const Node* node,
    const std::vector<::dml::Expression>& inputs,
    std::vector<::dml::Expression>& outputs) {
  auto attr = absl::any_cast<ElementwiseAttributes>(node->operation.attributes);
  const float* scalar = absl::get_if<float>(&attr.param);
  if (scalar == nullptr) {
    return absl::InvalidArgumentError("Not supported attribute.");
  }

  return CreateScaleBiasExpression(1.0f, -(*scalar), inputs, outputs);
}

}  // namespace dml
}  // namespace gpu
}  // namespace tflite