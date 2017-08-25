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

#include "concatenation_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace KernelSelector 
{

    ParamsKey ConcatenationKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnableConcatAxis(ConcatAxis::X);
        k.EnableConcatAxis(ConcatAxis::Y);
        k.EnableConcatAxis(ConcatAxis::FEATURE);
        k.EnableConcatAxis(ConcatAxis::BATCH);
        k.EnableConcatKernelPerInput();
        return k;
    }

    JitConstants ConcatenationKernelRef::GetJitConstants(const ConcatenationParams& params) const
    {
        auto cldnnJit = ConcatenationKernelBase::GetJitConstants(params);
        const ConcatenationParams& orgParams = static_cast<const ConcatenationParams&>(params);
        if (orgParams.inputs[0].X().v == 1 && orgParams.inputs[0].Y().v == 1)
        {
            cldnnJit.AddConstant(MakeJitConstant("CHECK_FEATURES", 1));
        }
        return cldnnJit;
    }

    KernelsData ConcatenationKernelRef::GetKernelsData(const Params& params, const OptionalParams& optParams) const
    {
        KernelsData kd = GetCommonKernelsData(params, optParams);
        for (int i = 0; i < (int)kd[0].kernels.size(); i++)
        {
            auto& kernel = kd[0].kernels[i];

            // to avoid cases when we execute with local work sizes 1x1x1
            if (kernel.workGroups.local[0] == 1 &&
                kernel.workGroups.global[1] != 1)
            {
                kernel.workGroups.global[1] = Align(kernel.workGroups.global[1], 32);
                kernel.workGroups.local[1] = 32;
            }
        }
        return kd;
    }
}