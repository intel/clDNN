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

#include "softmax_kernel_base.h"

namespace KernelSelector 
{
    JitConstants SoftmaxKernelBase::GetJitConstants(const SoftmaxParams& params, SoftmaxKernelBase::DispatchData kd) const
    {
        JitConstants mem_consts = MakeSoftmaxJitConstants(params);

        mem_consts.AddConstants({
            MakeJitConstant("ITEMS_NUM",      kd.itemsNum),
            MakeJitConstant("LWS",            kd.lws0),
            MakeJitConstant("GWS",            kd.gws0),
            MakeJitConstant("DATA_SETS_COUNT",kd.dataSetsCount),
            MakeJitConstant("DATA_SET_SIZE",  kd.dataSetSize),
            MakeJitConstant("LEFTOVERS",      kd.leftovers),
        });

        return mem_consts;
    }

    static bool validate(const SoftmaxParams& params)
    {
        const auto& input = params.inputs[0];

        if (params.activationFunc != ActivationFunction::NONE)
        {
            return false;
        }
        
        if (input.GetLayout() == DataLayout::bf ||
            input.GetLayout() == DataLayout::fb)
        {
            return true;
        }

        switch (params.smParams.dim)
        {
        case SoftmaxDim::X:         return input.Y().v == 1 && input.Feature().v == 1;
        case SoftmaxDim::Y:         return input.X().v == 1 && input.Feature().v == 1;
        case SoftmaxDim::FEATURE:   return input.X().v == 1 && input.Y().v == 1;
        default:                    return false;
        }
    }

    SoftmaxKernelBase::DispatchData SoftmaxKernelBase::SetDefault(const SoftmaxParams& params, const OptionalParams&) const
    {
        const auto& input = params.inputs[0];

        DispatchData kd;
        
        kd.gws0 = 1;
        kd.gws1 = 1;
        kd.gws2 = 1;

        kd.lws0 = 1;
        kd.lws1 = 1;
        kd.lws2 = 1;


        kd.fp16UnitUsed = input.GetDType() == Datatype::F16;
        kd.leftovers = 0;
        kd.itemsNum = 0;
        kd.normIndex = 0;
        kd.dataSetsCount = 0;

        // currently all derived kernels support bf/fb only
        auto flatten_input = input.FlattenFeatureAndSpatials();
        kd.dataSetSize = flatten_input.Feature().v;
        kd.dataSetsCount = input.Batch().v;

        return kd;
    }

    KernelsData SoftmaxKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& options, float estimated_time) const
    {
        assert(params.GetType() == KernelType::SOFT_MAX);

        const SoftmaxParams& orgParams = static_cast<const SoftmaxParams&>(params);

        if (!validate(orgParams))
        {
            return{};
        }

        KernelData kd = KernelData::Default<SoftmaxParams>(params);

        auto runInfo = SetDefault(orgParams, options);
        auto cldnn_jit = GetJitConstants(orgParams, runInfo);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point);

        kd.estimatedTime = estimated_time;

        return{ kd };
    }
}