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

#include "fully_connected_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace KernelSelector 
{
    JitConstants FullyConnectedKernelBase::GetJitConstants(const FullyConnectedParams& params, const FullyConnectedKernelBase::DispatchData& data) const
    {
        JitConstants mem_consts = MakeFullyConnectedJitConstants(params);

        if (data.vload_kernel_type)
        {
            const auto batches_per_work_item = GetBatchesPerWorkItem(params);

            mem_consts.AddConstant(MakeJitConstant("NEURONS_PER_WORK_ITEM", GetNeuronsPerWorkItem(params))); // how many neurons for a single batch will a single work item produce
            mem_consts.AddConstant(MakeJitConstant("BATCHES_PER_WORK_ITEM", batches_per_work_item));             // how many batches will a single work item compute
            mem_consts.AddConstant(MakeJitConstant("OUTPUT_ELEMENTS_COUNT", params.output.LogicalSize() / params.output.Batch().v));
        }

        return mem_consts;
    }

    FullyConnectedKernelBase::DispatchData FullyConnectedKernelBase::SetDefault(const FullyConnectedParams& params) const
    {
        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

        // Determine global work sizes.
        kd.gws0 = params.output.LogicalSize();
        kd.gws1 = kd.gws2 = 1;

        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = kd.lws2 = 1;
        kd.vload_kernel_type = false;

        return kd;
    }

    KernelsData FullyConnectedKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& options, DataLayout dl, std::vector<WeightsLayout> wl, float estimated_time) const
    {
        if (!Validate(params, options) ||
            wl.empty())
        {
            return KernelsData();
        }

        const auto& orgParams = static_cast<const FullyConnectedParams&>(params);
        const auto& orgOptParams = static_cast<const FullyConnectedOptionalParams&>(options);

        const bool bSupportedActivation = CheckActivationSupport(orgParams.activationFunc);

        bool bProperInput = orgParams.inputs[0].GetLayout() == dl;
        if (!bProperInput && !orgParams.inputs[0].PitchesDifferFromLogicalDims())
        {
            bProperInput =
                (dl == DataLayout::fb && orgParams.inputs[0].GetLayout() == DataLayout::fyxb) ||
                (dl == DataLayout::bf && orgParams.inputs[0].GetLayout() == DataLayout::bfyx);
        }

        const bool bSupportedInput = orgOptParams.allowReorderInput || bProperInput;

        if (!bSupportedActivation || 
            !bSupportedInput)
        {
            return KernelsData();
        }

        KernelData kd = KernelData::Default<FullyConnectedParams>(params);
        FullyConnectedParams& newParams = *static_cast<FullyConnectedParams*>(kd.params.get());

        if (!bProperInput)
        {
            newParams.inputs[0] = newParams.inputs[0].TransformIgnorePadding(dl);
            kd.reorderInput = true;
        }

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            wl,
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        kd.kernels.resize(1);
        
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);

        DispatchData runInfo = SetDefault(newParams);
        auto cldnn_jit = GetJitConstants(newParams, runInfo);
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point, true, !orgParams.bias.empty());

        kd.estimatedTime = estimated_time;

        return{ kd };
    }
}