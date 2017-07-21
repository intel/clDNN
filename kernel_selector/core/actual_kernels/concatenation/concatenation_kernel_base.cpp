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

#include "tensor_type.h"
#include "concatenation_kernel_base.h"

namespace KernelSelector 
{
    static int32_t GetConcatChannelIndex(const ConcatenationParams& params)
    {
        Tensor::DataChannelName name = Tensor::DataChannelName::X;
        switch (params.concatParams.axis)
        {
        case ConcatAxis::X:         name = Tensor::DataChannelName::X; break;
        case ConcatAxis::Y:         name = Tensor::DataChannelName::Y; break;
        case ConcatAxis::FEATURE:   name = Tensor::DataChannelName::FEATURE; break;
        case ConcatAxis::BATCH:     name = Tensor::DataChannelName::BATCH; break;
        default: break;
        }

        return Tensor::Channelndex(params.inputs[0].GetLayout(), name);
    }

    bool ConcatenationKernelBase::Validate(const Params& p, const OptionalParams&) const
    {
        if (p.GetType() != KernelType::CONCATENATION)
        {
            return false;
        }

        const ConcatenationParams& params = static_cast<const ConcatenationParams&>(p);

        if (GetConcatChannelIndex(params) == -1)
        {
            return false;
        }

        if (params.activationFunc != ActivationFunction::NONE)
        {
            return false;
        }

        return true; 
    }

    JitConstants ConcatenationKernelBase::GetJitConstants(const ConcatenationParams& params) const
    {
        return MakeConcatenationJitConstants(params);
    }

    ConcatenationKernelBase::DispatchData ConcatenationKernelBase::SetDefault(const ConcatenationParams& params) const
    {
        DispatchData kd;

        const auto& dims = params.inputs[0].GetDims();
        // Determine global work sizes.
        kd.gws0 = dims.size() < 2 ? 1 : dims[1].v;
        kd.gws1 = dims.size() < 3 ? 1 : dims[2].v;
        kd.gws2 = dims.size() < 4 ? 1 : dims[3].v;

        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }

        kd.lws1 = 1;
        kd.lws2 = 1;
        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }

    KernelsData ConcatenationKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params,  options))
        {
            return{};
        }

        const ConcatenationParams& orgParams = static_cast<const ConcatenationParams&>(params);

        DispatchData runInfo = SetDefault(orgParams);
        KernelData kd = KernelData::Default<ConcatenationParams>(params);

        auto cldnnJit = GetJitConstants(orgParams);
        auto entryPoint = GetEntryPoint(kernelName, orgParams.layerID,  options);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entryPoint);

        kd.estimatedTime = runInfo.effiency;

        return{ kd };
    }
}