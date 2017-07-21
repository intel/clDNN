/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "kernel_selector_common.h"
#include "reorder_weights_kernel.h"
#include "kernel_selector_utils.h" 
 
namespace KernelSelector 
{
    ParamsKey ReorderWeightsKernel::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputWeightsType(WeightsType::F16);
        k.EnableOutputWeightsType(WeightsType::F32);
        k.EnableAllWeightsLayout();
        k.EnableDifferentTypes();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        return k;
    }

    KernelsData ReorderWeightsKernel::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::REORDER);

        KernelData kd = KernelData::Default<ReorderWeightsParams>(params);
        ReorderWeightsParams& newParams = *static_cast<ReorderWeightsParams*>(kd.params.get());

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto cldnn_jit = GetJitConstants(newParams);
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        const auto& out = newParams.reorderParams.output;
        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = { out.OFM().v, out.IFM().v, out.X().v*out.Y().v };
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit, entry_point, ROUND_ROBIN);
        kernel.arguments = GetArgsDesc(1, false, false);

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}