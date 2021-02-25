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

#include "tensorflow/lite/delegates/gpu/dml/d3d_shader.h"
#include "absl/strings/str_cat.h"
#include <d3dcompiler.h>

using Microsoft::WRL::ComPtr;

namespace tflite {
namespace gpu {
namespace dml {

D3DShader::D3DShader() : shader(nullptr), init_srv(false) {}

D3DShader::~D3DShader() { Release(); }

absl::Status D3DShader::Compile(DMLDevice* device,
                                const std::string& shader_source,
                                bool has_sampler) {
  HRESULT result = S_OK;
  ComPtr<ID3DBlob> errors;
  result = D3DCompile(shader_source.c_str(), shader_source.size(), nullptr,
                      nullptr, nullptr, "main", "cs_5_1", 0, 0, &shader,
                      errors.GetAddressOf());
  if (!SUCCEEDED(result)) {
    void* error_buffer = errors ? errors->GetBufferPointer() : nullptr;
    return absl::UnknownError(absl::StrCat("D3DCompile Error ", error_buffer ? (char*)error_buffer : ""));
  }

  // Define root table layout
  CD3DX12_DESCRIPTOR_RANGE desc_range[3];
  desc_range[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // t0
  desc_range[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0
  if (has_sampler) {
    desc_range[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0);  // s0
  }

  CD3DX12_ROOT_PARAMETER root_parameters[4];
  root_parameters[0].InitAsConstants(3, 0);
  root_parameters[1].InitAsDescriptorTable(1, &desc_range[0], D3D12_SHADER_VISIBILITY_ALL);
  root_parameters[2].InitAsDescriptorTable(1, &desc_range[1], D3D12_SHADER_VISIBILITY_ALL);
  if (has_sampler) {
    root_parameters[3].InitAsDescriptorTable(1, &desc_range[2],
                                             D3D12_SHADER_VISIBILITY_ALL);
  }

  CD3DX12_ROOT_SIGNATURE_DESC root_signature_desc(has_sampler ? 4 : 3,
                                                  root_parameters);

  ComPtr<ID3DBlob> serialized_signature;
  DML_CHECK_SUCCEEDED(D3D12SerializeRootSignature(
   &root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1,
   serialized_signature.GetAddressOf(), nullptr));

  // Create the root signature
  DML_CHECK_SUCCEEDED(device->d3d_device->CreateRootSignature(
      0, serialized_signature->GetBufferPointer(),
      serialized_signature->GetBufferSize(),
      IID_PPV_ARGS(&root_signature)));

//  root_signature->SetName(L"Compute RS");

  // Create compute pipeline state
  D3D12_COMPUTE_PIPELINE_STATE_DESC compute_desc = {};
  compute_desc.pRootSignature = root_signature.Get();
  compute_desc.CS.pShaderBytecode = shader->GetBufferPointer();
  compute_desc.CS.BytecodeLength = shader->GetBufferSize();

  DML_CHECK_SUCCEEDED(device->d3d_device->CreateComputePipelineState(
      &compute_desc, IID_PPV_ARGS(&pipeline_stat)));
//  pipeline_stat->SetName(L"Compute PSO");

  // Create descriptor heaps.
  D3D12_DESCRIPTOR_HEAP_DESC descriptor_heap_desc{};
  descriptor_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  descriptor_heap_desc.NumDescriptors = 2;
  descriptor_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  DML_CHECK_SUCCEEDED(device->d3d_device->CreateDescriptorHeap(
      &descriptor_heap_desc, IID_PPV_ARGS(&descriptor_heap)));

  if (has_sampler) {
    descriptor_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    descriptor_heap_desc.NumDescriptors = 1;
    descriptor_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    DML_CHECK_SUCCEEDED(device->d3d_device->CreateDescriptorHeap(
        &descriptor_heap_desc, IID_PPV_ARGS(&sampler_descriptor_heap)));
  }
 
  return absl::OkStatus();
}

void D3DShader::Release() {
  if (shader) {
    shader->Release();
    shader = nullptr;
  }
  init_srv = false;
}

struct ImageLayoutCB {
  UINT Height;
  UINT Width;
  UINT Channels;
};

// Divide and round up
static UINT DivUp(UINT a, UINT b) { return (a + b - 1) / b; }

absl::Status D3DShader::Dispatch(DMLDevice* device, UINT width, UINT height,
                                 UINT channels,
                                 const DirectMlResource* input,
                                 const DirectMlResource* output) {

  if (init_srv == false) {
    init_srv = true;

    // Describe and create a SRV for the input tensor.
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.Format = input->data_type == DML_TENSOR_DATA_TYPE_FLOAT32
                          ? DXGI_FORMAT_R32_FLOAT
                          : DXGI_FORMAT_R16_FLOAT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    UINT data_size = input->data_type == DML_TENSOR_DATA_TYPE_FLOAT32
                    ? sizeof(float)
                    : sizeof(uint16_t);
    srv_desc.Buffer.NumElements =
        static_cast<UINT>(input->resource->GetDesc().Width / data_size);
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    D3D12_CPU_DESCRIPTOR_HANDLE handle =
        descriptor_heap->GetCPUDescriptorHandleForHeapStart();

    device->d3d_device->CreateShaderResourceView(input->resource, &srv_desc,
                                                 handle);

    // Describe and create a UAV for the output tensor.
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = output->data_type == DML_TENSOR_DATA_TYPE_FLOAT32
                          ? DXGI_FORMAT_R32_FLOAT
                          : DXGI_FORMAT_R16_FLOAT;
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.FirstElement = 0;
    data_size = output->data_type == DML_TENSOR_DATA_TYPE_FLOAT32
                         ? sizeof(float)
                         : sizeof(uint16_t);
    uav_desc.Buffer.NumElements =
        static_cast<UINT>(output->resource->GetDesc().Width / data_size);
    uav_desc.Buffer.StructureByteStride = 0;
    uav_desc.Buffer.CounterOffsetInBytes = 0;
    uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    UINT64 increment = device->d3d_device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    handle.ptr = static_cast<SIZE_T>(handle.ptr + UINT64(1) * increment);

    device->d3d_device->CreateUnorderedAccessView(output->resource, nullptr,
                                                  &uav_desc, handle);

  }

  ID3D12DescriptorHeap* descriptor_heaps[] = {descriptor_heap.Get()};
  device->command_list->SetDescriptorHeaps(ARRAYSIZE(descriptor_heaps),
                                           descriptor_heaps);

  device->command_list->SetComputeRootSignature(root_signature.Get());

  ImageLayoutCB imageLayoutCB = {};
  imageLayoutCB.Height = height;
  imageLayoutCB.Width = width;
  imageLayoutCB.Channels = channels;

  device->command_list->SetComputeRoot32BitConstants(0, 3, &imageLayoutCB, 0);

  D3D12_GPU_DESCRIPTOR_HANDLE handle =
      descriptor_heap->GetGPUDescriptorHandleForHeapStart();
  device->command_list->SetComputeRootDescriptorTable(1, handle);

  UINT64 increment = device->d3d_device->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  handle.ptr = static_cast<SIZE_T>(handle.ptr + UINT64(1) * increment);
  device->command_list->SetComputeRootDescriptorTable(2, handle);

  device->command_list->SetPipelineState(pipeline_stat.Get());
  device->command_list->Dispatch(DivUp(width, 32), DivUp(height, 16), 1);

  device->command_list->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));

  return absl::OkStatus();
}

absl::Status D3DShader::Dispatch(DMLDevice* device, UINT width, UINT height,
                                 UINT channels, const DirectMlTexture* input,
                                 const DirectMlResource* output) {

  // Describe and create a SRV for the input texture.
  D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
  srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv_desc.Format = input->format;
  srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  srv_desc.Texture2D.MostDetailedMip = 0;
  srv_desc.Texture2D.MipLevels = 1;
  srv_desc.Texture2D.PlaneSlice = 0;

  device->d3d_device->CreateShaderResourceView(input->resource, &srv_desc,
      descriptor_heap->GetCPUDescriptorHandleForHeapStart());

  if (init_srv == false) {
    init_srv = true;
    // Describe and create a UAV for the output tensor.
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = output->data_type == DML_TENSOR_DATA_TYPE_FLOAT32
                          ? DXGI_FORMAT_R32_FLOAT
                          : DXGI_FORMAT_R16_FLOAT;
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.FirstElement = 0;
    UINT data_size = output->data_type == DML_TENSOR_DATA_TYPE_FLOAT32
                         ? sizeof(float)
                         : sizeof(uint16_t);
    uav_desc.Buffer.NumElements =
        static_cast<UINT>(output->resource->GetDesc().Width / data_size);
    uav_desc.Buffer.StructureByteStride = 0;
    uav_desc.Buffer.CounterOffsetInBytes = 0;
    uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    D3D12_CPU_DESCRIPTOR_HANDLE handle =
        descriptor_heap->GetCPUDescriptorHandleForHeapStart();
    UINT64 increment = device->d3d_device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    handle.ptr = static_cast<SIZE_T>(handle.ptr + UINT64(1) * increment);

    device->d3d_device->CreateUnorderedAccessView(output->resource, nullptr,
                                                  &uav_desc, handle);
  
    // Describe and create a Sampler.
    D3D12_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc.MinLOD = 0;
    sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
    sampler_desc.MipLODBias = 0.0f;
    sampler_desc.MaxAnisotropy = 1;
    sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;

    device->d3d_device->CreateSampler(
        &sampler_desc,
        sampler_descriptor_heap->GetCPUDescriptorHandleForHeapStart());
  }

  ID3D12DescriptorHeap* descriptor_heaps[] = {descriptor_heap.Get(),
                                              sampler_descriptor_heap.Get()};
  device->command_list->SetDescriptorHeaps(ARRAYSIZE(descriptor_heaps),
                                           descriptor_heaps);

  device->command_list->SetComputeRootSignature(root_signature.Get());

  ImageLayoutCB imageLayoutCB = {};
  imageLayoutCB.Height = height;
  imageLayoutCB.Width = width;
  imageLayoutCB.Channels = channels;

  device->command_list->SetComputeRoot32BitConstants(0, 3, &imageLayoutCB, 0);

  D3D12_GPU_DESCRIPTOR_HANDLE handle =
      descriptor_heap->GetGPUDescriptorHandleForHeapStart();
  device->command_list->SetComputeRootDescriptorTable(1, handle);

  UINT64 increment = device->d3d_device->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  handle.ptr = static_cast<SIZE_T>(handle.ptr + UINT64(1) * increment);
  device->command_list->SetComputeRootDescriptorTable(2, handle);

  handle = sampler_descriptor_heap->GetGPUDescriptorHandleForHeapStart();
  device->command_list->SetComputeRootDescriptorTable(3, handle);

  device->command_list->SetPipelineState(pipeline_stat.Get());
  device->command_list->Dispatch(DivUp(width, 32), DivUp(height, 16), 1);

  device->command_list->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));

  return absl::OkStatus();
}


}  // namespace dml
}  // namespace gpu
}  // namespace tflite
