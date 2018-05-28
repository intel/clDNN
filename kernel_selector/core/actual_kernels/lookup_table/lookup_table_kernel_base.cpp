/*
// Copyright (c) 2018 Intel Corporation
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

#include "lookup_table_kernel_base.h"

namespace KernelSelector
{
    bool LookUpTableKernelBase::Validate(const Params& p, const OptionalParams& o) const
    {
        if (p.GetType() != KernelType::LOOKUP_TABLE ||
            o.GetType() != KernelType::LOOKUP_TABLE)
        {
            return false;
        }

        return true;
    }

    JitConstants LookUpTableKernelBase::GetJitConstants(const LookUpTableParams& params) const
    {
        JitConstants mem_consts = MakeLookUpTableJitConstants(params);
        return mem_consts;
    }

    LookUpTableKernelBase::DispatchData LookUpTableKernelBase::SetDefault(const LookUpTableParams& params) const
    {
        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

        // Determine global work sizes.
        kd.gws0 = params.lookUpTableParams.inputIndices.X().v;
        kd.gws1 = params.lookUpTableParams.inputIndices.Batch().v;                   // B
        kd.gws2 = 1;

        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = 1;
        kd.lws2 = 1;

        return kd;
    }

    KernelsData LookUpTableKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& options, float estimatedTime) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        const LookUpTableParams& orgParams = static_cast<const LookUpTableParams&>(params);

        DispatchData runInfo = SetDefault(orgParams);

        KernelData kd = KernelData::Default<LookUpTableParams>(params);

        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point, "", false, false, 2);

        kd.estimatedTime = estimatedTime;

        return{ kd };
    }
}