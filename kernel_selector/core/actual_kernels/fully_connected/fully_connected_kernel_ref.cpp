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

#include "fully_connected_kernel_ref.h"
#include "kernel_selector_utils.h"
 
namespace KernelSelector 
{
    ParamsKey FullyConnectedKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableInputLayout(DataLayout::bf);
        k.EnableOutputLayout(DataLayout::bf);
        k.EnableBiasPerFeature();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    KernelsData FullyConnectedKernelRef::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        KernelData kd = KernelData::Default<FullyConnectedParams>(params);

        FullyConnectedParams& newParams = *static_cast<FullyConnectedParams*>(kd.params.get());
        const std::string kernel_id = GetEntryPoint(kernelName, params.layerID, options);

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            { WeightsLayout::oiyx },
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        std::stringstream jit;
        jit << GetBaseJit(newParams, kernel_id)
            << GetFullyConnectedJit(newParams);

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = { out.Feature().v, out.Batch().v };
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit.str(), kernel_id);
        kernel.arguments = GetArgumentDesc(1, true, !newParams.bias.empty());

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}